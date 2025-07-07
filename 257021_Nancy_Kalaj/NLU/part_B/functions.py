import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from conll import evaluate

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def eval_loop_bert(data_loader, model, tokenizer, lang, device):
    model.eval()
    all_ref, all_hyp = [], []
    ref_intents, hyp_intents = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_slots       = batch["slot_labels"]
            gt_intents     = batch["intent_label"].to(device)

            slot_logits, intent_logits = model(input_ids, attention_mask)
            # intents
            preds_int = intent_logits.argmax(dim=1)
            ref_intents.extend([lang.id2intent[i.item()] for i in gt_intents.cpu()])
            hyp_intents.extend([lang.id2intent[p.item()] for p in preds_int.cpu()])

            # slots
            preds_slots = slot_logits.argmax(dim=2).cpu()
            for ids, word_ids, gts, preds in zip(
                input_ids.cpu(), batch["attention_mask"], gt_slots, preds_slots
            ):
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
                ref_seq, hyp_seq = [], []
                for tok, wid, gt, p in zip(tokens, word_ids.tolist(), gts.tolist(), preds.tolist()):
                    if wid is None or gt == -100:
                        continue
                    ref_seq.append((tok, lang.id2slot[gt]))
                    hyp_seq.append((tok, lang.id2slot[p]))
                all_ref.append(ref_seq)
                all_hyp.append(hyp_seq)

    slot_res    = evaluate(all_ref, all_hyp)
    intent_acc  = accuracy_score(ref_intents, hyp_intents)
    return slot_res, intent_acc

def train_model(
    train_loader, dev_loader, model, tokenizer, lang,
    optimizer, slot_criterion, intent_criterion,
    num_epochs, device
):
    best_f1 = 0.0
    best_model = None

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels    = batch["slot_labels"].to(device)
            intent_labels  = batch["intent_label"].to(device)

            slot_logits, intent_logits = model(input_ids, attention_mask)
            loss_slot   = slot_criterion(
                slot_logits.view(-1, slot_logits.size(-1)),
                slot_labels.view(-1)
            )
            loss_intent = intent_criterion(intent_logits, intent_labels)
            loss = loss_slot + loss_intent
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        slot_res, intent_acc = eval_loop_bert(dev_loader, model, tokenizer, lang, device)
        dev_f1 = slot_res["total"]["f"]
        print(f"Ep{epoch} — train_loss: {avg_train:.4f}, dev_f1: {dev_f1:.4f}, dev_int_acc: {intent_acc:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_model = model.state_dict()

    # load best
    model.load_state_dict(best_model)
    return model
