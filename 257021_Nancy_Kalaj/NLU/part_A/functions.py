import torch, torch.nn as nn, math
from sklearn.metrics import classification_report
from model import ModelIAS

def build_model(cfg, lang):
    return ModelIAS(
      vocab_size = len(lang.word2id),
      emb_size   = cfg["emb_size"],
      hid_size   = cfg["hid_size"],
      n_slots    = len(lang.slot2id),
      n_intents  = len(lang.intent2id),
      pad_idx    = lang.word2id["pad"],
      n_layers   = cfg.get("n_layers",1),
      drop       = cfg.get("dropout",0.0)
    )

def init_weights(model: nn.Module):
    """
    Initialize RNN/LSTM with Xavier/orthogonal and Linear uniformly.
    """
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

def train_epoch(loader, model, opt, slot_cr, intent_cr, clip):
    model.train()
    losses = []
    for b in loader:
        opt.zero_grad()
        s_logits, i_logits = model(b["utterances"], b["slots_len"])
        l_slot   = slot_cr(s_logits, b["y_slots"])
        l_intent = intent_cr(i_logits,  b["intents"])
        (l_slot + l_intent).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append((l_slot + l_intent).item())
    return sum(losses) / len(losses)

def eval_loop(loader, criterion_slots, criterion_intents, model, lang):
    from conll import evaluate

    model.eval()
    loss_array   = []
    ref_intents  = []
    hyp_intents  = []
    ref_slots    = []
    hyp_slots    = []

    with torch.no_grad():
        for sample in loader:
            # forward pass
            slots, intents = model(sample["utterances"], sample["slots_len"])
            # losses (if you need them)
            loss_slot   = criterion_slots(slots,  sample["y_slots"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_array.append((loss_slot + loss_intent).item())

            # intent inference
            out_ints = torch.argmax(intents, dim=1).tolist()
            gt_ints  = sample["intents"].tolist()
            ref_intents.extend([lang.id2intent[i] for i in gt_ints])
            hyp_intents.extend([lang.id2intent[i] for i in out_ints])

            # slot inference
            output_slots = torch.argmax(slots, dim=1)  # (B, T)
            for idx in range(len(sample["slots_len"])):
                L        = sample["slots_len"][idx].item()
                utt_ids  = sample["utterances"][idx, :L].tolist()
                gt_ids   = sample["y_slots"][idx, :L].tolist()
                # map IDs back to tags/words
                gt_tags  = [lang.id2slot[i] for i in gt_ids]
                words    = [lang.id2word[i] for i in utt_ids]
                pr_ids   = output_slots[idx, :L].tolist()
                pr_tags  = [lang.id2slot[i] for i in pr_ids]

                # build seq of (word, tag)
                ref_slots.append(list(zip(words, gt_tags)))
                hyp_slots.append(list(zip(words, pr_tags)))

    # compute metrics exactly as in notebook
    slot_res = evaluate(ref_slots, hyp_slots)
    intent_rep = classification_report(
        ref_intents, hyp_intents,
        zero_division=False, output_dict=True
    )
    return slot_res, intent_rep, loss_array
