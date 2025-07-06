# main.py
import argparse, sys, os, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from utils import ensure_atis, prepare_splits, Lang, make_loader, DEVICE, PAD_TOKEN
from functions import build_model, init_weights, train_epoch, eval_model

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp",      required=True)
    p.add_argument("--test",action="store_true")
    args = p.parse_args()

    # load config
    cfg_mod = __import__(f"configs.{args.exp}", fromlist=["CFG"])
    cfg = cfg_mod.CFG

    # make sure ATIS JSONs + conll.py are downloaded
    ensure_atis()
    sys.path.insert(0, os.path.join(os.getcwd(), "dataset/ATIS"))
    # data + loaders
    tr, dv, te = prepare_splits("dataset/ATIS/train.json","dataset/ATIS/test.json")
    lang = Lang(tr, dv, te)
    loaders = {
      "train": make_loader(tr, lang, bs=128, shuffle=True),
      "dev":   make_loader(dv, lang, bs=64,  shuffle=False),
      "test":  make_loader(te, lang, bs=64,  shuffle=False),
    }

    ckpt_path = f"bin/{args.exp}_best.pt"
    slot_cr = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    intent_cr = nn.CrossEntropyLoss()

    # test?
    if args.test:
        model = build_model(cfg, lang).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        slot_res, intent_res = eval_model(loaders["test"], model, slot_cr, intent_cr, lang)
        print(f"TEST {args.exp}: Slot F1={slot_res['total']['f']:.4f}, Intent Acc={intent_res['accuracy']:.4f}") # type: ignore
        exit(0)

    # multi-run + early stopping
    slot_f1s, intent_accs = [], []
    best_dev_f1 = 0.0
    best_model  = None

    for run in range(5):
        model = build_model(cfg, lang).to(DEVICE)
        init_weights(model)
        Optim = getattr(optim, cfg["optimizer"])
        optimizer = Optim(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0.0))

        epochs_no_improve = 0
        for epoch in range(1, 200):
            train_epoch(loaders["train"], model, optimizer, slot_cr, intent_cr, clip=5)
            slot_dev, _ = eval_model(loaders["dev"], model, slot_cr, intent_cr, lang)
            dev_f1 = slot_dev["total"]["f"]

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                break

        # evaluate this run’s best on test
        slot_test, intent_test = eval_model(loaders["test"], best_model, slot_cr, intent_cr, lang)
        slot_f1s.append(slot_test["total"]["f"])
        if not isinstance(intent_test, dict):
            raise RuntimeError(f"Expected dict from eval_model, got {type(intent_test)}")
        intent_accs.append(intent_test["accuracy"])

    # summarize
    print("Slot F1 = %.4f ± %.4f" % (np.mean(slot_f1s), np.std(slot_f1s)))
    print("Intent Acc = %.4f ± %.4f" % (np.mean(intent_accs), np.std(intent_accs)))

    # save best model
    os.makedirs("bin", exist_ok=True)
    if best_model is None:
        raise RuntimeError("No model was ever trained—best_model is still None!")
    torch.save(best_model.state_dict(), ckpt_path)
    print(f"Best model saved to {ckpt_path}")
