import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import copy
from conll import evaluate
from sklearn.metrics import classification_report
import csv
import matplotlib.pyplot as plt

from model import *
from utils import PAD_TOKEN


def build_model(cfg, lang):
    from model import ModelIAS
    return ModelIAS(
        vocab_size=len(lang.word2id), emb_size=cfg['emb_size'],
        hid_size=cfg['hid_size'], n_slots=len(lang.slot2id),
        n_intents=len(lang.intent2id), pad_idx=lang.word2id['pad'],
        n_layers=cfg.get('n_layers',1), drop=cfg.get('dropout',0.0)
    )

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
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s)) 
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# Runs the whole training process for the model for multiple times (= runs) and provides a final evaluation on its average performances
def train_model(train_loader, dev_loader, test_loader, lang, model, criterion_slots, criterion_intents, cfg):
    results = {
        "best_model": None,
        "slot_f1": 0,
        "int_acc": 0,
        "losses_dev": [],
        "losses_train": [],
        "sampled_epochs": [],
    }
    slot_f1s, intent_acc, best_models = [], [], []

    for run in tqdm(range(0, 5)):
        model.apply(init_weights)
        optimizer_cls = getattr(torch.optim, cfg.get("optimizer", "Adam"))
        optimizer = optimizer_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None

        for epoch in tqdm(range(1,200)):
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=5)
                if epoch % 5 == 0: 
                    sampled_epochs.append(epoch)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                    losses_dev.append(np.asarray(loss_dev).mean())
                    
                    f1 = results_dev['total']['f']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = 0
                    else:
                        patience -= 1
                    if patience <= 0:
                        break 
        
        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)   
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        best_models.append((best_model, best_f1))
        results["losses_dev"].append(losses_dev)
        results["losses_train"].append(losses_train)
        results["sampled_epochs"].append(sampled_epochs)

        print('Slot F1', results_test['total']['f'])
        print('Intent Acc', intent_test['accuracy'])

    # Compute and store mean for slot_f1s and intent accuracy
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    results["slot_f1"] = round(slot_f1s.mean(),3)
    results["int_acc"] = round(intent_acc.mean(), 3)

    best_model, _ = max(best_models, key=lambda x: x[1])
    results["best_model"] = copy.deepcopy(best_model).to('cpu')

    print('Slot F1', results['slot_f1'], '+-', round(slot_f1s.std(),3))
    print('Intent Acc', results['int_acc'], '+-', round(slot_f1s.std(), 3))

    return results

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
