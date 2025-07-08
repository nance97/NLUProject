import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import copy
from conll import evaluate
from sklearn.metrics import classification_report

from model import *
from utils import DEVICE, PAD_TOKEN


# Custom weight initialization for all supported modules in the model
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
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

# Constructs and returns a ModelIAS instance with the given hyperparameters and configuration
def build_model(out_slot, out_int, vocab_len, cfg):
    from model import ModelIAS
    model = ModelIAS(
        vocab_size=vocab_len, emb_size=cfg['emb_size'],
        hid_size=cfg['hid_size'], n_slots=out_slot,
        n_intents=out_int, pad_idx=PAD_TOKEN,
        drop=cfg['dropout']
    ).to(DEVICE)
    return model

# Performs one full pass over the training set, optimizing the model weights
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        # Combine intent and slot losses for joint optimization
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Prevent exploding gradients
        optimizer.step()

    return loss_array

# Evaluates the model
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])  # Model inference
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot

            # Collect ground-truth and predicted intents for reporting
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Collect ground-truth and predicted slot labels for CoNLL evaluation
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
    return results, report_intent

# Main experiment runner: Trains and validates the model across multiple runs, tracking metrics and best-performing model
def train_model(train_data, val_data, test_data, vocab, intent_count, slot_count, vocab_dim, loss_slots, loss_intents, config):
    stats = {
        "best_model": None,
        "slot_f1": 0.0,
        "int_acc": 0.0,
    }
    slot_scores = []
    intent_scores = []
    run_models = []

    # run multiple times to check stability
    for run_no in tqdm(range(5)):
        # initialize model and optimizer
        model = build_model(slot_count, intent_count, vocab_dim, config)
        model.apply(init_weights)
        OptimizerCls = getattr(optim, config["optimizer"])
        optimizer = OptimizerCls(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))

        best_val_f1 = 0.0
        best_val_model = None
        patience = 3

        # epoch loop with early stopping on validation slot F1
        for epoch in tqdm(range(1, 100), desc=f"Run {run_no+1} Epochs"):
            losses = train_loop(train_data, optimizer, loss_slots, loss_intents, model, clip=5)

            if epoch % 5 == 0:
                val_res, _ = eval_loop(val_data, loss_slots, loss_intents, model, vocab)
                val_f1 = val_res["total"]["f"]

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_model = copy.deepcopy(model).cpu()
                    patience = 3
                else:
                    patience -= 1
                    if patience == 0:
                        break

        # test the best validation model
        best_val_model.to(DEVICE)
        test_res, test_intent_rep = eval_loop(test_data, loss_slots, loss_intents, best_val_model, vocab)
        slot_scores.append(test_res["total"]["f"])
        intent_scores.append(test_intent_rep["accuracy"])
        run_models.append((best_val_model, best_val_f1))

        print(f"Run {run_no+1}: Test Slot F1={test_res['total']['f']:.3f}, Intent Acc={test_intent_rep['accuracy']:.3f}")

    # aggregate across runs
    slot_arr = np.array(slot_scores)
    intent_arr = np.array(intent_scores)
    stats["slot_f1"] = round(slot_arr.mean(), 3)
    stats["int_acc"] = round(intent_arr.mean(), 3)

    # pick the top model
    top_model, _ = max(run_models, key=lambda x: x[1])
    stats["best_model"] = copy.deepcopy(top_model).cpu()

    print(f"Aggregated Slot F1: {stats['slot_f1']} ± {round(slot_arr.std(),3)}")
    print(f"Aggregated Intent Acc: {stats['int_acc']} ± {round(intent_arr.std(),3)}")

    return stats

