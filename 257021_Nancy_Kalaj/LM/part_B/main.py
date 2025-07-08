import argparse, copy, importlib
from functions import (build_model, init_weights, train_loop, eval_loop)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import (ensure_ptb, read_file, Lang, PennTreebank, make_loader, DEVICE)


if __name__ == "__main__":
    # Parse --exp to select which experiment configuration to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Name of config, e.g. exp3")
    parser.add_argument("--test", action="store_true", help="Skip training; load and evaluate saved model on test set")
    args = parser.parse_args()

    # Dynamically load the config module for this experiment
    cfg_mod = importlib.import_module(f"configs.{args.exp}")
    cfg = cfg_mod.CFG

    # Ensure PTB data files are present (downloads if missing)
    ensure_ptb()
    # Read raw lines, appending the <eos> token
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw   = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw  = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Build vocabulary and dataset objects
    lang = Lang(train_raw, special_tokens=["<pad>","<eos>"])
    ptb_train = PennTreebank(train_raw, lang)
    ptb_dev   = PennTreebank(dev_raw,   lang)
    ptb_test  = PennTreebank(test_raw,  lang)

    # Create data loaders with padding and batching
    pad_idx = lang.word2id["<pad>"]
    loader_train = make_loader(ptb_train, batch_size=64, pad_token=pad_idx, shuffle=True)
    loader_dev = make_loader(ptb_dev, batch_size=128, pad_token=pad_idx)
    loader_test = make_loader(ptb_test, batch_size=128, pad_token=pad_idx)

    # Build the model according to cfg (RNN or LSTM, with dropouts, tying, etc.), move to DEVICE
    model = build_model(cfg, len(lang.word2id), pad_idx).to(DEVICE)
    # Initialize weights for reproducibility and stability
    init_weights(model)

    # Test the best saved version of the model
    if args.test:
        # load checkpoint
        ckpt = f"bin/{args.exp}_best_model.pt"
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        test_ppl = eval_loop(loader_test, model, nn.CrossEntropyLoss(ignore_index=pad_idx))
        print(f"[TEST] {args.exp} → Test PPL = {test_ppl:.2f}")
        exit(0)

    # Set up optimizer: just AdamW (with weight decay)
    if cfg["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(cfg["optimizer"])

    # Cross-entropy loss, ignoring pad tokens
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_dev   = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training with early stopping based on dev set perplexity and (optionally) with avSGD
    n_epochs = 100
    patience = 3
    best_ppl = float('inf')
    best_model = None
    epochs_no_improve = 0
    using_asgd = False

    for epoch in range(1, n_epochs+1):
        # train for one epoch (returns average token-level loss)
        train_loss = train_loop(loader_train, model, optimizer, criterion_train, clip=5)
        # evaluate perplexity on the dev set
        dev_ppl = eval_loop(loader_dev, model, criterion_dev)
        print(f"Epoch {epoch}: dev PPL = {dev_ppl:.2f}")

        # update best_model if we improved on dev
        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # switch to avSGD if no improvement for `patience` epochs and train until you get no improvements for `patience` epochs again
        if epochs_no_improve >= patience:
            if cfg.get("use_avsgd", False) and not using_asgd:
                print(f"No improvement for {patience} epochs. Switching to ASGD (lr={cfg['asgd_lr']})…")
                optimizer = optim.ASGD(model.parameters(), lr=cfg["asgd_lr"])
                using_asgd = True
                epochs_no_improve = 0
            else:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

    # After training, use the best seen model for final test evaluation
    if best_model is not None:
        model = best_model
    test_ppl = eval_loop(loader_test, model, criterion_dev)
    print(f"Test perplexity: {test_ppl:.2f}")

    # Save the best model checkpoint for reproducibility
    os.makedirs("bin", exist_ok=True)
    torch.save(model.state_dict(), f"bin/{args.exp}_best_model.pt")
    print(f"Best model saved to bin/{args.exp}_best_model.pt")
