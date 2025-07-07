import argparse, sys, os, copy, numpy as np, torch, torch.optim as optim, torch.nn as nn
from utils import ensure_atis, prepare_splits, Lang, make_loader, DEVICE, PAD_TOKEN
from functions import build_model, init_weights, train_epoch, eval_model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    cfg_mod = __import__(f"configs.{args.exp}", fromlist=['CFG'])
    cfg = cfg_mod.CFG

    ensure_atis()
    sys.path.insert(0, os.path.join(os.getcwd(), 'dataset/ATIS'))
    train, dev, test = prepare_splits('dataset/ATIS/train.json', 'dataset/ATIS/test.json')
    lang = Lang(train, dev, test)

    loaders = {
        'train': make_loader(train, lang, bs=128, shuffle=True),
        'dev': make_loader(dev, lang, bs=64, shuffle=False),
        'test': make_loader(test, lang, bs=64, shuffle=False),
    }

    ckpt = f"bin/{args.exp}_best.pt"
    slot_cr = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    intent_cr = nn.CrossEntropyLoss()

    if args.test:
        model = build_model(cfg, lang).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        slot_res, intent_res = eval_model(loaders['test'], model, slot_cr, intent_cr, lang)
        print(f"TEST {args.exp}: Slot F1={slot_res['total']['f']:.4f}, Intent Acc={intent_res['accuracy']:.4f}")
        sys.exit(0)

    best_dev_f1 = 0.0
    best_model = None
    slot_scores, intent_scores = [], []

    for run in range(5):
        model = build_model(cfg, lang).to(DEVICE)
        init_weights(model)
        Optim = getattr(optim, cfg['optimizer'])
        optimizer = Optim(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay',0.0))

        no_improve = 0
        for epoch in range(1, 200):
            train_epoch(loaders['train'], model, optimizer, slot_cr, intent_cr, clip=5)
            slot_dev,_ = eval_model(loaders['dev'], model, slot_cr, intent_cr, lang)
            f1 = slot_dev['total']['f']
            if f1 > best_dev_f1:
                best_dev_f1, best_model, no_improve = f1, copy.deepcopy(model), 0
            else:
                no_improve += 1
            if no_improve >= 3:
                break

        slot_test,_ = eval_model(loaders['test'], best_model, slot_cr, intent_cr, lang)
        intent_test = eval_model(loaders['test'], best_model, slot_cr, intent_cr, lang)[1]
        slot_scores.append(slot_test['total']['f'])
        intent_scores.append(intent_test['accuracy'])

    print(f"Slot F1 = {np.mean(slot_scores):.4f} ± {np.std(slot_scores):.4f}")
    print(f"Intent Acc = {np.mean(intent_scores):.4f} ± {np.std(intent_scores):.4f}")

    os.makedirs('bin', exist_ok=True)
    if best_model is None:
        raise RuntimeError("No model was trained!")
    torch.save(best_model.state_dict(), ckpt)