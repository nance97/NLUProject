import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from LM.part_A.model import LM_LSTM, RNN_cell


def build_model(cfg, vocab_size, pad_idx):
    if cfg["model_type"] == "RNN_cell":
        return RNN_cell(
            hidden_size=cfg["hid_size"],
            input_size=cfg["emb_size"],
            output_size=vocab_size,
            vocab_size=vocab_size,
            dropout=cfg.get("dropout", 0.0)
        )
    elif cfg["model_type"] == "LM_LSTM":
        return LM_LSTM(
            emb_size=cfg["emb_size"],
            hidden_size=cfg["hid_size"],
            output_size=vocab_size,
            pad_index=pad_idx,
            n_layers=cfg.get("n_layers", 1),
            fc_dropout=cfg.get("dropout", 0.0),
            emb_dropout=cfg.get("embed_dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {cfg['model_type']}")

def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.LSTM,)):
            for name, p in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias' in name:
                    p.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

def train_loop(loader, model, optimizer, criterion, clip=5.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch["source"])
        loss = criterion(logits, batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss  += loss.item() * batch["n_tokens"]
        total_tokens += batch["n_tokens"]
    return total_loss / total_tokens

def eval_loop(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["source"])
            loss = criterion(logits, batch["target"])
            total_loss  += loss.item() * batch["n_tokens"]
            total_tokens += batch["n_tokens"]
    ppl = math.exp(total_loss / total_tokens)
    return ppl

def find_best_lr(
    lrs: list,
    loader_train, loader_dev,
    model, criterion_train, criterion_dev,
    init_fn, clip=5.0
):
    best = {}
    for lr in lrs:
        init_fn(model)  # re‐init weights
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # single epoch for speed
        train_loop(loader_train, model, optimizer, criterion_train, clip)
        ppl = eval_loop(loader_dev, model, criterion_dev)
        best[lr] = ppl
    # return lr with minimal dev-PPL
    return min(best.items(), key=lambda x: x[1])
