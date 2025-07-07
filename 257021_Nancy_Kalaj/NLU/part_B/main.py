import os
import argparse
import torch
import torch.nn as nn
from utils import prepare_data
from model import BertForJointIntentSlot
from functions import set_seed, train_model, eval_loop_bert

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
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
    model = BertForJointIntentSlot(
        pretrained_model_name="bert-base-uncased",
        n_intents=len(lang.intent2id),
        n_slots=len(lang.slot2id)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    intent_criterion = nn.CrossEntropyLoss()

    if not args.test:
        model = train_model(
            train_loader, dev_loader, model, tokenizer, lang,
            optimizer, slot_criterion, intent_criterion,
            args.epochs, device
        )
        os.makedirs("bin", exist_ok=True)
        torch.save(model.state_dict(), "bin/best_model.bin")
    else:
        model.load_state_dict(torch.load("bin/best_model.bin", map_location=device))

    # final test
    slot_res, intent_acc = eval_loop_bert(test_loader, model, tokenizer, lang, device)
    print(f"\nTEST Slot F1 = {slot_res['total']['f']:.4f}, TEST Intent Acc = {intent_acc:.4f}")

if __name__ == "__main__":
    main()
