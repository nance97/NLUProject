import math
import torch
import torch.nn as nn
from model import LM_LSTM, LM_RNN


def build_model(cfg, vocab_size, pad_idx):
    """
    Instantiate the language model based on experiment configuration.

    Uses either the vanilla RNN (LM_RNN) or LSTM (LM_LSTM) class under the hood.

    Args:
        cfg (dict): Experiment config dict with keys like 'model_type', 'emb_size', etc.
        vocab_size (int): Number of tokens in the vocabulary.
        pad_idx (int): Index of the <pad> token for embedding padding.

    Returns:
        nn.Module: An instance of either LM_RNN or LM_LSTM.
    """
    model_type = cfg.get("model_type")
    if model_type == "LM_RNN":
        # Vanilla RNN: input_dim=emb_size, hidden_dim=hid_size
        return LM_RNN(
            emb_size=cfg["emb_size"],
            hidden_size=cfg["hid_size"],
            output_size=vocab_size,
            pad_index=pad_idx
        )
    elif model_type == "LM_LSTM":
        # LSTM: supports multiple layers and dropout
        return LM_LSTM(
            emb_size=cfg["emb_size"],
            hidden_size=cfg["hid_size"],
            output_size=vocab_size,
            pad_index=pad_idx,
            n_layers=cfg.get("n_layers", 1),
            emb_dropout=cfg.get("embed_dropout", 0.0),
            fc_dropout=cfg.get("dropout", 0.0)
        )
    else:
        # Safety catch for unknown model types
        raise ValueError(f"Unknown model type: {model_type}")


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
