import math
import torch
import torch.nn as nn
from model import LM_LSTM, LockedDropout


def build_model(cfg, vocab_size, pad_idx):
    model = LM_LSTM(
        emb_size=cfg["emb_size"],
        hidden_size=cfg["hid_size"],
        output_size=vocab_size,
        pad_index=pad_idx,
        n_layers=cfg.get("n_layers", 1),
        fc_dropout=cfg.get("dropout", 0.0),
        emb_dropout=cfg.get("embed_dropout", 0.0)
    )

    if cfg.get("weight_tying", False):
        assert cfg["emb_size"] == cfg["hid_size"], \
            "For weight tying, emb_size must equal hidden_size"
        # tie embedding <-> output projection
        model.output_layer.weight = model.embedding.weight

    if cfg.get("use_variational_dropout", False):
        # replace the nn.Dropout modules with LockedDropout
        model.emb_dropout = LockedDropout(cfg["embed_dropout"])
        model.fc_dropout = LockedDropout(cfg["dropout"])

    return model


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
