import math
import torch
import torch.nn as nn
from model import LM_LSTM, LockedDropout


def build_model(cfg, vocab_size, pad_idx):
    """
    Instantiate the language model based on experiment configuration.

    Uses the LSTM (LM_LSTM) class under the hood.

    Args:
        cfg (dict): Experiment config dict with keys like 'model_type', 'emb_size', 'weight_tying', 'use_var_drop', etc.
        vocab_size (int): Number of tokens in the vocabulary.
        pad_idx (int): Index of the <pad> token for embedding padding.

    Returns:
        nn.Module: An instance of LM_LSTM.
    """
    # LSTM: supports multiple layers, regular or variational dropout and weight-tying
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

    if cfg.get("use_var_drop", False):
        # replace the nn.Dropout modules with LockedDropout
        model.emb_dropout = LockedDropout(cfg["embed_dropout"])
        model.fc_dropout = LockedDropout(cfg["dropout"])

    return model


def init_weights(model: nn.Module):
    """
    Initialize model weights for reproducibility and stable training.

    Applies Xavier/orthogonal initialization for RNN/LSTM layers
    and uniform for Linear layers.
    """
    for module in model.modules():
        # RNN/LSTM weight initialization
        if isinstance(module, (nn.LSTM,)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
        # Linear projection initialization
        elif isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


def train_loop(loader, model, optimizer, criterion, clip=5.0):
    """
    Run one epoch of training.

    Args:
        loader (DataLoader): batches of source,target sequences
        model (nn.Module): language model to train
        optimizer (Optimizer): optimizer for parameter updates
        criterion (Loss): cross-entropy loss ignoring pad tokens
        clip (float): max norm for gradient clipping

    Returns:
        float: average token-level loss over the epoch
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        optimizer.zero_grad()
        # Forward: compute logits seq_len × vocab for each batch element
        logits = model(batch["source"])
        # Compute CE loss over non-pad tokens
        loss = criterion(logits, batch["target"])
        # Backprop and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Accumulate weighted by number of tokens in batch
        total_loss += loss.item() * batch["n_tokens"]
        total_tokens += batch["n_tokens"]

    return total_loss / total_tokens


def eval_loop(loader, model, criterion):
    """
    Evaluate model perplexity on dev/test set.

    Args:
        loader (DataLoader): batches to evaluate
        model (nn.Module): trained language model
        criterion (Loss): same loss used in training

    Returns:
        float: perplexity = exp(average loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["source"])
            loss = criterion(logits, batch["target"])
            total_loss += loss.item() * batch["n_tokens"]
            total_tokens += batch["n_tokens"]

    # convert average loss to perplexity
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl
