import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from model import BertJointNLU
from conll import evaluate
from torch.optim import AdamW

# Set seeds for reproducibility across numpy, random, and torch (CPU and GPU)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Evaluate the model on a dataset, returning slot metrics and intent accuracy
def eval_loop_bert(data_loader, model, tokenizer, lang, device):
    model.eval()
    all_ref, all_hyp = [], []     # For slot evaluation (CoNLL style)
    ref_intents, hyp_intents = [], []  # For intent classification accuracy

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_slots = batch["slot_labels"]
            gt_intents = batch["intent_label"].to(device)

            slot_logits, intent_logits = model(input_ids, attention_mask)
            # Intent predictions (argmax over classes)
            preds_int = intent_logits.argmax(dim=1)
            ref_intents.extend([lang.id2intent[i.item()] for i in gt_intents.cpu()])
            hyp_intents.extend([lang.id2intent[p.item()] for p in preds_int.cpu()])

            # Slot predictions for each token (argmax over slot types)
            preds_slots = slot_logits.argmax(dim=2).cpu()
            for ids, word_ids, gts, preds in zip(
                input_ids.cpu(), batch["attention_mask"], gt_slots, preds_slots
            ):
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
                ref_seq, hyp_seq = [], []
                for tok, wid, gt, p in zip(tokens, word_ids.tolist(), gts.tolist(), preds.tolist()):
                    # Skip padding and special tokens
                    if wid is None or gt == -100:
                        continue
                    ref_seq.append((tok, lang.id2slot[gt]))
                    hyp_seq.append((tok, lang.id2slot[p]))
                all_ref.append(ref_seq)
                all_hyp.append(hyp_seq)

    slot_res = evaluate(all_ref, all_hyp)
    intent_acc = accuracy_score(ref_intents, hyp_intents)
    return slot_res, intent_acc

# Trains the model, performing early stopping based on dev slot F1; returns the best model
def train_model(
    train_loader, dev_loader, model, tokenizer, lang,
    optimizer, slot_criterion, intent_criterion,
    num_epochs, device,
    patience: int = 3
):
    best_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels    = batch["slot_labels"].to(device)
            intent_labels  = batch["intent_label"].to(device)

            # Forward pass, compute joint loss (intent + slot)
            slot_logits, intent_logits = model(input_ids, attention_mask)
            loss_slot   = slot_criterion(
                slot_logits.view(-1, slot_logits.size(-1)),
                slot_labels.view(-1)
            )
            loss_intent = intent_criterion(intent_logits, intent_labels)
            loss = loss_slot + loss_intent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validation (dev set) evaluation phase
        slot_res, intent_acc = eval_loop_bert(dev_loader, model, tokenizer, lang, device)
        dev_f1 = slot_res["total"]["f"]
        print(f"Ep{epoch} — train_loss: {avg_train:.4f}, dev_f1: {dev_f1:.4f}, dev_int_acc: {intent_acc:.4f}")

        # Early stopping check: save best model state if F1 improves
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Stopping early at epoch {epoch} (no improvement in {patience} epochs).")
                break

    # Restore the best weights from early stopping
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# Runs multiple independent fine-tuning experiments and aggregates results
def run_bert_multi(
    train_loader,
    dev_loader,
    test_loader,
    tokenizer,
    lang,
    pretrained_model_name: str,
    slot_criterion,
    intent_criterion,
    device,
    runs: int = 5,
    lr: float = 3e-5,
    num_epochs: int = 100,
    patience: int = 3,
):

    slot_f1s = []
    intent_accs = []
    best_model_states = []

    for run in range(runs):
        seed = 42 + run
        set_seed(seed)

        # Instantiate a fresh model and optimizer for each run
        model = BertJointNLU(
            pretrained_model_name,
            n_intents=len(lang.intent2id),
            n_slots=len(lang.slot2id),
            dropout=0.1
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Train with early stopping on dev F1
        model = train_model(
            train_loader, dev_loader,
            model, tokenizer, lang,
            optimizer,
            slot_criterion, intent_criterion,
            num_epochs, device,
            patience=patience
        )

        # Final evaluation on test set
        slot_res, acc = eval_loop_bert(test_loader, model, tokenizer, lang, device)
        slot_f1 = slot_res["total"]["f"]
        slot_f1s.append(slot_f1)
        intent_accs.append(acc)
        best_model_states.append(model.state_dict())

        print(f"Run {run+1}/{runs}: slot-F1={slot_f1:.4f}, intent-acc={acc:.4f}")

    return slot_f1s, intent_accs, best_model_states
