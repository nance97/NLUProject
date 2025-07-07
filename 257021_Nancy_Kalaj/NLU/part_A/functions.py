import torch
import copy
import numpy as np
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import classification_report


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

    """
    Evaluate joint slot and intent model, mirroring original notebook logic.
    """
    model.eval()
    refs_slots, hyp_slots = [], []
    refs_int,   hyp_int   = [], []
    with torch.no_grad():
        for batch in loader:
            slots, intents = model(batch['utterances'], batch['slots_len'])
            # Intent inference
            pred_int = torch.argmax(intents, dim=1).tolist()
            refs_int.extend([lang.id2intent[i] for i in batch['intents'].tolist()])
            hyp_int.extend([lang.id2intent[i] for i in pred_int])
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for idx_seq, seq in enumerate(output_slots):
                length = batch['slots_len'][idx_seq].item()
                utt_ids = batch['utterances'][idx_seq][:length].tolist()
                gt_ids  = batch['y_slots'][idx_seq][:length].tolist()
                # Build reference slot sequence
                gt_slots = [lang.id2slot[i] for i in gt_ids]
                utterance = [lang.id2word[i] for i in utt_ids]
                # Build hypothesis sequence with mapping unknown tags to 'O'
                pred_tags = seq[:length].tolist()
                hyp_seq = []
                ref_set = set(gt_slots)
                for p in pred_tags:
                    tag = lang.id2slot[p]
                    if tag not in ref_set:
                        tag = 'O'
                    hyp_seq.append(tag)

                # Combine into (word, tag) tuples
                refs_slots.append(list(zip(utterance, gt_slots)))
                hyp_slots.append(list(zip(utterance, hyp_seq)))

    # Use conlleval for slot metrics
    slot_res = evaluate(refs_slots, hyp_slots)
    # Classification report for intents
    intent_res = classification_report(refs_int, hyp_int, output_dict=True, zero_division=False)
    return slot_res, intent_res
