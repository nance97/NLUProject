import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import prepare_data
from model import BertForJointIntentSlot
from functions import set_seed, run_bert_multi

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    # reproducibility
    set_seed(42)

    # data
    tokenizer, train_loader, dev_loader, test_loader, lang = prepare_data(
        "dataset/ATIS/train.json",
        "dataset/ATIS/test.json",
        tokenizer_name="bert-base-uncased",
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    intent_criterion = nn.CrossEntropyLoss()

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

    # Aggregate results
    slot_arr, intent_arr = np.array(slot_f1s), np.array(intent_accs)
    print("\n=== Final over 5 runs ===")
    print(f"Slot F1 : {slot_arr.mean():.4f} ± {slot_arr.std():.4f}")
    print(f"Intent Acc: {intent_arr.mean():.4f} ± {intent_arr.std():.4f}")

    # Save the best-of-5 model
    best_run = int(slot_arr.argmax())
    best_state = model_states[best_run]
    model = BertForJointIntentSlot(
        "bert-base-uncased",
        n_intents=len(lang.intent2id),
        n_slots=len(lang.slot2id),
        dropout=0.1
    ).to(device)
    model.load_state_dict(best_state)
    os.makedirs("bin", exist_ok=True)
    torch.save(model.state_dict(), "bin/best_bert.bin")

if __name__ == "__main__":
    main()
