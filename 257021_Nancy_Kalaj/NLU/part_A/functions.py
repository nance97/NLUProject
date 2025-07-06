import torch, torch.nn as nn, math
from sklearn.metrics import classification_report
from utils import DEVICE, PAD_TOKEN
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
    # delay import so conll.py has been downloaded
    from conll import evaluate  # type: ignore

    model.eval()
    all_ref_slots, all_hyp_slots = [], []
    ref_ints, hyp_ints = [], []
    with torch.no_grad():
        for b in loader:
            s_logits, i_logits = model(b["utterances"], b["slots_len"])

            # Intent preds
            preds_int = i_logits.argmax(-1).tolist()
            ref_ints += [lang.id2intent[i.item()] for i in b["intents"]]
            hyp_ints += [lang.id2intent[p]         for p in preds_int]

            # Slot preds
            sl = s_logits.argmax(1)  # (B, T)
            for i in range(len(b["slots_len"])):
                L      = b["slots_len"][i].item()
                tokens = [lang.id2word[idx] for idx in b["utterances"][i,:L].tolist()]
                gold_ids= b["y_slots"][i,:L].tolist()
                pred_ids= sl[i,:L].tolist()

                ref_seq, hyp_seq = [], []
                for tok, gid, pid in zip(tokens, gold_ids, pred_ids):
                    gold_tag = lang.id2slot[gid]
                    pred_tag = lang.id2slot[pid]
                    # skip padding or None tags
                    if gold_tag == "pad" or gold_tag is None:
                        continue
                    ref_seq.append((tok, gold_tag))
                    hyp_seq.append((tok, pred_tag))

                all_ref_slots.append(ref_seq)
                all_hyp_slots.append(hyp_seq)

    slot_res   = evaluate(all_ref_slots, all_hyp_slots)
    intent_rep = classification_report(ref_ints, hyp_ints,
                                       zero_division=False,
                                       output_dict=True)
    return slot_res, intent_rep
