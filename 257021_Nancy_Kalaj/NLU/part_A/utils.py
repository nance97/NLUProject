# utils.py
import os, urllib.request
import json
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE    = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PAD_TOKEN = 0

# URLs to fetch ATIS + conll.py
ATIS_URLS = {
    "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
    "test.json":  "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json",
    "conll.py":   "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py",
}

def ensure_atis(atis_dir="dataset/ATIS"):
    os.makedirs(atis_dir, exist_ok=True)
    for fname, url in ATIS_URLS.items():
        dest = os.path.join(atis_dir, fname) if fname.endswith('.json') else os.path.join(os.getcwd(), fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, dest)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def prepare_splits(train_path, test_path, dev_frac=0.1, seed=42):
    raw_train = load_json(train_path)
    raw_test  = load_json(test_path)

    # stratify by intent, hold out 10% as dev
    intents = [ex["intent"] for ex in raw_train]
    freq    = Counter(intents)
    X, singles, y = [], [], []
    for ex in raw_train:
        if freq[ex["intent"]] > 1:
            X.append(ex); y.append(ex["intent"])
        else:
            singles.append(ex)

    X_tr, X_dev, _, _ = train_test_split(
        X, y, test_size=dev_frac, stratify=y,
        random_state=seed, shuffle=True
    )
    X_tr += singles
    return X_tr, X_dev, raw_test

class Lang:
    def __init__(self, train_exs, dev_exs, test_exs):
        # collect all words, slots, intents
        words = sum((ex["utterance"].split() for ex in train_exs), [])
        slots = sum((ex["slots"].split()     for ex in train_exs+dev_exs+test_exs), [])
        intents = [ex["intent"] for ex in train_exs+dev_exs+test_exs]

        self.word2id   = {"pad":PAD_TOKEN, "unk":1}
        self.slot2id   = {"pad":PAD_TOKEN}
        self.intent2id = {}
        idx_w = 2
        for w in words:
            if w not in self.word2id:
                self.word2id[w] = idx_w; idx_w+=1

        idx_s = 1
        for s in slots:
            if s not in self.slot2id:
                self.slot2id[s] = idx_s; idx_s+=1

        idx_i = 0
        for i in sorted(set(intents)):
            self.intent2id[i] = idx_i; idx_i+=1

        # reverse maps
        self.id2word   = {i:w for w,i in self.word2id.items()}
        self.id2slot   = {i:s for s,i in self.slot2id.items()}
        self.id2intent = {i:i_ for i_,i in self.intent2id.items()}

class ATISDataset(Dataset):
    def __init__(self, examples, lang):
        self.exs  = examples
        self.lang = lang

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        ex   = self.exs[i]
        words = ex["utterance"].split()
        slots = ex["slots"].split()
        # map to IDs (with unk fall-back for words)
        wids = [ self.lang.word2id.get(w, self.lang.word2id["unk"]) 
                 for w in words ]
        sids = [ self.lang.slot2id[t] for t in slots ]
        iid  = self.lang.intent2id[ex["intent"]]
        return {
            "utterance": torch.LongTensor(wids),
            "slots":     torch.LongTensor(sids),
            "intent":    iid
        }

def collate_fn(batch):
    # sort by utterance length
    batch.sort(key=lambda ex: ex["utterance"].size(0), reverse=True)

    utts    = [ex["utterance"] for ex in batch]
    slots   = [ex["slots"]     for ex in batch]
    intents = [ex["intent"]     for ex in batch]
    lengths = [u.size(0)        for u in utts]
    maxlen  = max(lengths)

    # pad utterances & slots
    utt_padded  = torch.full((len(batch), maxlen), PAD_TOKEN, dtype=torch.long)
    slots_padded= torch.full((len(batch), maxlen), PAD_TOKEN, dtype=torch.long)
    for i, (u, s) in enumerate(zip(utts, slots)):
        L = lengths[i]
        utt_padded[i, :L]    = u
        slots_padded[i, :L]  = s

    intent_tensor = torch.tensor(intents, dtype=torch.long)

    return {
        "utterances": utt_padded.to(DEVICE),
        "y_slots":    slots_padded.to(DEVICE),
        "intents":    intent_tensor.to(DEVICE),
        "slots_len":  torch.tensor(lengths, dtype=torch.long).to(DEVICE)
    }

def make_loader(exs, lang, bs=64, shuffle=False):
    ds = ATISDataset(exs, lang)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn)
                 