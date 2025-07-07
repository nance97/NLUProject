import argparse, sys, os, copy, numpy as np, torch, torch.optim as optim, torch.nn as nn
from utils import ensure_atis, prepare_loaders, DEVICE, PAD_TOKEN
from functions import build_model, eval_loop, init_weights, train_loop

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    cfg_mod = __import__(f"configs.{args.exp}", fromlist=['CFG'])
    cfg = cfg_mod.CFG

    ensure_atis()
    sys.path.insert(0, os.path.join(os.getcwd(), 'dataset/ATIS'))
    train_loader, dev_loader, test_loader, lang, _, _, _ = prepare_loaders('dataset/ATIS/train.json', 'dataset/ATIS/test.json')

    ckpt = f"bin/{args.exp}_best.pt"
    slot_cr = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    intent_cr = nn.CrossEntropyLoss()

    results = {
        "best_model": None,
        "slot_f1": 0,
        "int_acc": 0,
        "losses_dev": [],
        "losses_train": [],
        "sampled_epochs": [],
    }
    slot_f1s, intent_acc, best_models = [], [], []

    for run in range(5):
        model = build_model(cfg, lang)
        model.apply(init_weights)
        Optim = getattr(optim, cfg['optimizer'])
        optimizer = Optim(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay',0.0))
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None

        for epoch in range(1, 200):
                loss = train_loop(train_loader, optimizer, slot_cr, intent_cr, model, clip=5)
                if epoch % 5 == 0: 
                    sampled_epochs.append(epoch)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, _, loss_dev = eval_loop(dev_loader, slot_cr, intent_cr, model, lang)
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
        results_test, intent_test, _ = eval_loop(test_loader, slot_cr, intent_cr, best_model, lang)   
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
