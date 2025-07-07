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
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    # DEBUG: print sizes
    print(f"[DEBUG] #sentences: ref={len(ref_slots)}, hyp={len(hyp_slots)}")
    print("[DEBUG] first 5 sentence lengths (ref vs hyp):",
          [(len(r), len(h)) for r, h in zip(ref_slots[:5], hyp_slots[:5])])

    # Let it crash so you actually see the error and stack trace
    results = evaluate(ref_slots, hyp_slots)
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
