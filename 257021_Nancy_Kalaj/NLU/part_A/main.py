import argparse
import os
from utils import ensure_atis, prepare_loaders
from functions import *


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    cfg_mod = __import__(f"configs.{args.exp}", fromlist=['CFG'])
    cfg = cfg_mod.CFG

    ensure_atis()

    train_loader, dev_loader, test_loader, lang = prepare_loaders("dataset/ATIS/train.json", "dataset/ATIS/test.json")

    # Define the loss functions for slot filling (token-level) and intent classification (utterance-level)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    save_path = f"bin/{args.exp}_best_model.pt"

    # Load the best model from disk and run evaluation on the test set
    if args.test:
        ckpt = torch.load(save_path, map_location=DEVICE)

        lang = ckpt["lang"]
        train_loader, dev_loader, test_loader, _ = prepare_loaders(
            "dataset/ATIS/train.json",
            "dataset/ATIS/test.json",
            lang=lang
        )

        cfg = ckpt["cfg"]
        model = build_model(out_slot, out_int, vocab_len, cfg).to(DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        results_test, intent_report = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print(f"\nTest slot-filling F1: {results_test['total']['f']:.3f}")
        print(f"Test intent accuracy : {intent_report['accuracy']:.3f}")
        exit(0)
    
    # Train
    results = train_model(
        train_loader, dev_loader, test_loader, lang, out_int, out_slot, 
        vocab_len, criterion_slots, criterion_intents, cfg
    )

    # Save best model
    os.makedirs("bin", exist_ok=True)
    ckpt = {
        "model_state_dict": results["best_model"].state_dict(),
        "cfg": cfg,
        "lang": lang
    }
    torch.save(ckpt, save_path)
    print(f"Best model saved to {save_path}")
