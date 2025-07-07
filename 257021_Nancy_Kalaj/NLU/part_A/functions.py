import torch
import copy
import numpy as np
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import classification_report
from utils import PAD_TOKEN


def build_model(cfg, lang):
    from model import ModelIAS
    return ModelIAS(
        vocab_size=len(lang.word2id), emb_size=cfg['emb_size'],
        hid_size=cfg['hid_size'], n_slots=len(lang.slot2id),
        n_intents=len(lang.intent2id), pad_idx=lang.word2id['pad'],
        n_layers=cfg.get('n_layers',1), drop=cfg.get('dropout',0.0)
    )


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def train_epoch(loader, model, optimizer, slot_cr, intent_cr, clip=5):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        slots, intents = model(batch['utterances'], batch['slots_len'])
        loss = slot_cr(slots, batch['y_slots']) + intent_cr(intents, batch['intents'])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def eval_model(loader, model, slot_cr, intent_cr, lang):
    from conll import evaluate 
    
    model.eval()
    refs_slots, hyps_slots = [], []
    refs_int,   hyps_int   = [], []
    with torch.no_grad():
        for batch in loader:
            slots, intents = model(batch['utterances'], batch['slots_len'])
            # Intent predictions
            pred_int = torch.argmax(intents, dim=1).tolist()
            refs_int.extend([lang.id2intent[i] for i in batch['intents'].tolist()])
            hyps_int.extend([lang.id2intent[i] for i in pred_int])
            # Slot predictions
            pred_slots = torch.argmax(slots, dim=1)
            for i, seq in enumerate(pred_slots):
                L = batch['slots_len'][i].item() if isinstance(batch['slots_len'][i], torch.Tensor) else batch['slots_len'][i]
                utt_ids = batch['utterances'][i][:L].tolist()
                gt_ids  = batch['y_slots'][i][:L].tolist()
                pred_ids= seq[:L].tolist()
                ref_seq = []
                hyp_seq = []
                for w, g, p in zip(utt_ids, gt_ids, pred_ids):
                    if g == PAD_TOKEN:
                        continue
                    word = lang.id2word.get(w, '<unk>')
                    ref_label = lang.id2slot.get(g, 'pad')
                    hyp_label = lang.id2slot.get(p, 'pad')
                    ref_seq.append((word, ref_label))
                    hyps_slots_inner = hyp_seq.append((word, hyp_label))
                refs_slots.append(ref_seq)
                hyps_slots.append(hyp_seq)
    slot_res = evaluate(refs_slots, hyps_slots)
    intent_res = classification_report(refs_int, hyps_int, output_dict=True, zero_division=False)
    return slot_res, intent_res
