import argparse, numpy as np, torch.optim as optim, torch.nn as nn
from functions import train_model, build_model
from utils import DEVICE, PAD_TOKEN, ensure_atis, prepare_data


TRAIN_DATA_PATH = "dataset/ATIS/train.json"
TEST_DATA_PATH = "dataset/ATIS/test.json"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    cfg_mod = __import__(f"configs.{args.exp}", fromlist=['CFG'])
    cfg = cfg_mod.CFG

    ensure_atis()

    # Use default or config batch sizes as you prefer
    train_loader, dev_loader, test_loader, lang = prepare_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH
    )

    model = build_model(cfg, lang)
    model.to(DEVICE)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    results = train_model(
        train_loader, dev_loader, test_loader, lang,
        model, criterion_slots, criterion_intents, cfg
    )
