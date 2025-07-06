import copy
import torch
from utils import (
    ensure_ptb, read_file, Lang,
    PennTreebank, make_loader, DEVICE
)
from functions import (
    init_weights,
    train_loop, eval_loop, build_model
)
import torch.nn as nn
import torch.optim as optim
import argparse, importlib
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Name of config, e.g. exp3")
    args = parser.parse_args()

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
    model = build_model(cfg, len(lang.word2id), pad_idx)
    model.to(DEVICE)
    init_weights(model)

    if cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(cfg["optimizer"])

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_dev = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 6) Final training with best lr + AdamW
    n_epochs  = 20
    best_dev_ppl = float('inf')
    best_model   = None

    for epoch in range(1, n_epochs+1):
        train_loop(loader_train, model, optimizer, criterion_train, clip=5.0)
        dev_ppl = eval_loop(loader_dev, model, criterion_dev)
        print(f"Epoch {epoch}: dev PPL = {dev_ppl:.2f}")

        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_model   = copy.deepcopy(model)

    # 7) Test
    test_ppl = eval_loop(loader_test, model, criterion_dev)
    print(f"Test perplexity: {test_ppl:.2f}")

    # 8) Save the best model
    os.makedirs("bin", exist_ok=True)
    to_save = best_model or model
    torch.save(to_save.state_dict(), "bin/partA_best_model.pt")
    print("Best model saved to bin/partA_best_model.pt")
