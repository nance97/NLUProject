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

def eval_model(loader, model, slot_cr, intent_cr, lang):
    from conll import evaluate

    model.eval()
    all_ref, all_hyp = [], []
    ref_ints, hyp_ints = [], []
    with torch.no_grad():
        for b in loader:
            s_logits, i_logits = model(b["utterances"], b["slots_len"])
            preds_int = i_logits.argmax(-1).tolist()
            ref_ints += [lang.id2intent[i] for i in b["intents"].tolist()]
            hyp_ints += [lang.id2intent[p] for p in preds_int]

            sl = s_logits.argmax(1)
            for i in range(len(b["slots_len"])):
                L        = b["slots_len"][i].item()
                toks     = [lang.id2word[idx] for idx in b["utterances"][i,:L].tolist()]
                gold_ids = b["y_slots"][i,:L].tolist()
                pred_ids = sl[i,:L].tolist()

                ref_seq, hyp_seq = [], []
                for tok, gid, pid in zip(toks, gold_ids, pred_ids):
                    # look up strings
                    gold_tag = lang.id2slot.get(gid, None)
                    pred_tag = lang.id2slot.get(pid, None)

                    # if something’s wrong, print it
                    if gold_tag is None or pred_tag is None or gold_tag == "pad":
                        print("  ⚠️ skipping bad tag:", tok, gid, gold_tag, pid, pred_tag)
                        continue

                    ref_seq.append((tok, gold_tag))
                    hyp_seq.append((tok, pred_tag))

                if ref_seq and hyp_seq:
                    all_ref.append(ref_seq)
                    all_hyp.append(hyp_seq)

    slot_res   = evaluate(all_ref, all_hyp)
    intent_rep = classification_report(ref_ints, hyp_ints,
                                       zero_division=False,
                                       output_dict=True)
    return slot_res, intent_rep
