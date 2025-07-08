import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import prepare_data
from model import BertJointNLU
from functions import set_seed, run_bert_multi
from datetime import datetime
import pickle

def main():
    # Parse command-line arguments for batch size, sequence length, epochs, learning rate, and test mode
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--test", action="store_true")
    p.add_argument("--model_path", type=str, default=None, help="Path to the saved .pt checkpoint (for --test mode)")
    args = p.parse_args()

    # Ensure deterministic results for reproducibility
    set_seed(42)

    # Load and preprocess data, returning tokenized DataLoaders and label mappings
    tokenizer, train_loader, dev_loader, test_loader, lang = prepare_data(
        "dataset/ATIS/train.json",
        "dataset/ATIS/test.json",
        tokenizer_name="bert-base-uncased",
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss functions for slot tagging and intent classification
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding in slot loss
    intent_criterion = nn.CrossEntropyLoss()

    # TEST MODE: load saved model and evaluate on test set
    if args.test:
        if args.model_path is None:
            raise ValueError("Must specify --model_path for testing")

        state = torch.load(args.model_path, map_location=device)
        lang_path = args.model_path.replace(".pt", "_lang.pkl")
        with open(lang_path, "rb") as f:
            lang = pickle.load(f)

        tokenizer, _, _, test_loader, _ = prepare_data(
            "dataset/ATIS/train.json",
            "dataset/ATIS/test.json",
            tokenizer_name="bert-base-uncased",
            max_length=args.max_length,
            batch_size=args.batch_size,
            lang=lang
        )

        model = BertJointNLU(
            "bert-base-uncased",
            n_intents=len(lang.intent2id),
            n_slots=len(lang.slot2id),
            dropout=0.1
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        from functions import eval_loop_bert
        slot_res, intent_acc = eval_loop_bert(test_loader, model, tokenizer, lang, device)
        print(f"\nTest slot-filling F1 : {slot_res['total']['f']:.4f}")
        print(f"Test intent accuracy  : {intent_acc:.4f}")
        exit(0)

    # Train and evaluate the model for multiple runs
    slot_f1s, intent_accs, model_states = run_bert_multi(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        lang=lang,
        pretrained_model_name="bert-base-uncased",
        slot_criterion=slot_criterion,
        intent_criterion=intent_criterion,
        device=device,
        runs=5,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=3,
    )

    # Report the mean and variance of results over all runs
    slot_arr, intent_arr = np.array(slot_f1s), np.array(intent_accs)
    print("\n=== Final over 5 runs ===")
    print(f"Slot F1 : {slot_arr.mean():.4f} ± {slot_arr.std():.4f}")
    print(f"Intent Acc: {intent_arr.mean():.4f} ± {intent_arr.std():.4f}")

    # Identify the best run by highest slot F1 and save the corresponding model
    best_run = int(slot_arr.argmax())
    best_state = model_states[best_run]
    model = BertJointNLU(
        "bert-base-uncased",
        n_intents=len(lang.intent2id),
        n_slots=len(lang.slot2id),
        dropout=0.1
    ).to(device)
    model.load_state_dict(best_state)

    os.makedirs("bin", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"bin/best_bert_{stamp}.pt"
    torch.save(model.state_dict(), save_path)
    with open(f"bin/lang_{stamp}.pkl", "wb") as f:
        pickle.dump(lang, f)
    print(f"Best BERT model saved to {save_path}")

if __name__ == "__main__": 
    main()
