import argparse, copy, importlib
from functions import (build_model, init_weights, train_loop, eval_loop)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import (ensure_ptb, read_file, Lang, PennTreebank, make_loader, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Name of config, e.g. exp3")
    args = parser.parse_args()

    # Load config from configs/expX.py
    cfg_mod = importlib.import_module(f"configs.{args.exp}")
    cfg = cfg_mod.CFG

    # 1) Load data
    ensure_ptb()
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw   = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw  = read_file("dataset/PennTreeBank/ptb.test.txt")

    # 2) Build vocab & datasets
    lang = Lang(train_raw, special_tokens=["<pad>","<eos>"])
    ptb_train = PennTreebank(train_raw, lang)
    ptb_dev   = PennTreebank(dev_raw,   lang)
    ptb_test  = PennTreebank(test_raw,  lang)

    # 3) DataLoaders
    pad_idx = lang.word2id["<pad>"]
    loader_train = make_loader(ptb_train, batch_size=64, pad_token=pad_idx, shuffle=True)
    loader_dev   = make_loader(ptb_dev,   batch_size=128, pad_token=pad_idx)
    loader_test  = make_loader(ptb_test,  batch_size=128, pad_token=pad_idx)

    # 4) Model + loss
    model = build_model(cfg, len(lang.word2id), pad_idx).to(DEVICE)
    init_weights(model)

    # 5) Optimizer
    if cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"])
    elif cfg["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(cfg["optimizer"])

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_dev   = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 6) Training loop with patience/early stopping
    n_epochs = 100
    patience = 3
    best_ppl = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs+1):
        train_loss = train_loop(loader_train, model, optimizer, criterion_train, clip=5.0)
        dev_ppl = eval_loop(loader_dev, model, criterion_dev)
        print(f"Epoch {epoch}: dev PPL = {dev_ppl:.2f}")

        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    # 7) Test
    if best_model is not None:
        model = best_model
    test_ppl = eval_loop(loader_test, model, criterion_dev)
    print(f"Test perplexity: {test_ppl:.2f}")

    # 8) Save the best model
    os.makedirs("bin", exist_ok=True)
    torch.save(model.state_dict(), f"bin/{args.exp}_best_model.pt")
    print(f"Best model saved to bin/{args.exp}_best_model.pt")
