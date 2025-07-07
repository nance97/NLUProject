import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import classification_report
from conll import evaluate


def build_model(cfg, lang):
    from model import ModelIAS
    return ModelIAS(
        vocab_size=len(lang.word2id), emb_size=cfg['emb_size'],
        hid_size=cfg['hid_size'], n_slots=len(lang.slot2id),
        n_intents=len(lang.intent2id), pad_idx=lang.word2id['pad'],
        n_layers=cfg.get('n_layers',1), drop=cfg.get('dropout',0.0)
    )

# Initializes the weights of the model layers
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


# Trains the model for one epoch over the provided data
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train() # Set model to training mode
    loss_array = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len']) # Forward pass
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # Compute loss; since it's joint training, it is the sum of the individual losses
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # clip the gradient to avoid exploding gradients
        optimizer.step() # Update the weights

    return loss_array

# Evaluates the model over the provided data
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    
    model.eval() # Set the model to evaluation mode
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): # Disable gradient computation
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len']) # Forward pass
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot # Compute loss
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterances'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])

                if not (len(utterance) == len(gt_slots) == len(to_decode)):
                    print(f"[BAD LENGTH] utt: {len(utterance)}, gt_slots: {len(gt_slots)}, pred: {len(to_decode)}")
                for elem in gt_ids[:length]:
                    if lang.id2slot[elem] == 'pad':
                        print(f"[GT PAD] idx={elem}, id2slot={lang.id2slot[elem]}")
                for elem in to_decode:
                    if lang.id2slot[elem] == 'pad':
                        print(f"[PRED PAD] idx={elem}, id2slot={lang.id2slot[elem]}")
                if len(ref_slots[-1]) == 0 or len(hyp_slots[-1]) == 0:
                    print(f"[EMPTY SEQ] at batch {id_seq}")

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    # … right after you build ref_slots & hyp_slots, before evaluate() …
    for sent_idx, seq in enumerate(hyp_slots):
        for tok_idx, pair in enumerate(seq):
            word, tag = pair            # unpack the tuple!
            if tag.startswith('I-'):
                # look at the previous predicted tag
                prev_tag = seq[tok_idx-1][1] if tok_idx > 0 else 'O'
                _, cur_type = tag.split('-', 1)
                ok_prev = prev_tag.endswith(cur_type) and prev_tag.startswith(('B-','I-'))
                if not ok_prev:
                    print(f"[INVALID IOB] sent={sent_idx}, tok={tok_idx}:")
                    print(f"  WORD={word!r}, GOLD={ref_slots[sent_idx][tok_idx][1]!r}, PRED={tag!r}")
                    print("  full GOLD seq:", [t for (_,t) in ref_slots[sent_idx]])
                    print("  full PRED seq:", [t for (_,t) in seq])
                    break
        else:
            continue
        break

    # now call your normal evaluator
    results = evaluate(ref_slots, hyp_slots)

        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
