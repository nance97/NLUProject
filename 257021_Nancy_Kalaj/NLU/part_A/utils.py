import os
import urllib.request
import json
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = 0

# URLs for ATIS and the CoNLL script
ATIS_URLS = {
    "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
    "test.json":  "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json",
    "conll.py":   "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py",
}

def ensure_atis(atis_dir="dataset/ATIS"):
    """
    Download ATIS JSON files and the conll.py evaluation script
    if they aren’t already present.
    """
    os.makedirs(atis_dir, exist_ok=True)
    for fname, url in ATIS_URLS.items():
        dest = os.path.join(atis_dir, fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, dest)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def prepare_splits(train_path, test_path, dev_frac=0.1, seed=42):
    raw = load_json(train_path)
    test = load_json(test_path)
    # stratify on intent
    intents = [ex["intent"] for ex in raw]
    freq = Counter(intents)
    X, singles = [], []
    y = []
    for ex in raw:
        if freq[ex["intent"]] > 1:
            X.append(ex)
            y.append(ex["intent"])
        else:
            singles.append(ex)
    X_train, X_dev, _, _ = train_test_split(
        X, y, test_size=dev_frac, stratify=y,
        random_state=seed
    )
    X_train += singles
    return X_train, X_dev, test

class Lang:
    def __init__(self, train_exs, dev_exs, test_exs, cutoff=0):
        words = [w for ex in train_exs for w in ex["utterance"].split()]
        slots = set(tok for ex in train_exs+dev_exs+test_exs
                         for tok in ex["slots"].split())
        intents = {ex["intent"] for ex in train_exs+dev_exs+test_exs}
        self.word2id = self._build(words, pad=True, unk=True, cutoff=cutoff)
        self.slot2id = self._build(list(slots), pad=True, unk=False)
        self.intent2id = self._build(list(intents), pad=False, unk=False)
        self.id2word = {i:w for w,i in self.word2id.items()}
        self.id2slot = {i:s for s,i in self.slot2id.items()}
        self.id2intent= {i:c for c,i in self.intent2id.items()}

    def _build(self, items, pad, unk, cutoff=0):
        idx = 0
        vocab = {}
        if pad: vocab["pad"] = idx; idx+=1
        if unk: vocab["unk"] = idx; idx+=1
        counts = Counter(items)
        for w,count in counts.items():
            if count > cutoff:
                vocab[w] = idx; idx+=1
        return vocab

class ATISDataset(Dataset):
    def __init__(self, exs, lang):
        self.exs = exs
        self.lang = lang

    def __len__(self): return len(self.exs)
    def __getitem__(self,i):
        ex = self.exs[i]
        # map words, slots, intent
        wids = [ self.lang.word2id.get(w,"unk") 
                 for w in ex["utterance"].split() ]
        sids = [ self.lang.slot2id[t] 
                 for t in ex["slots"].split() ]
        iid  = self.lang.intent2id[ex["intent"]]
        return {
            "wids": torch.LongTensor(wids),
            "sids": torch.LongTensor(sids),
            "iid": torch.LongTensor([iid])
        }

def collate_fn(data):
    # data: list of examples from your ATISJointDataset, each with 
    #   .utt_ids, .slot_ids, .intent_ids, .lengths

    # 1) sort by length descending
    data.sort(key=lambda x: len(x.utt_ids), reverse=True)

    # 2) extract lists
    wids_list    = [ex.utt_ids    for ex in data]
    sids_list    = [ex.slot_ids   for ex in data]
    intent_list  = [ex.intent_ids for ex in data]
    lengths      = [len(x) for x in wids_list]
    maxlen       = max(lengths)

    # 3) pad wids and sids
    def pad(seqs, pad_value=PAD_TOKEN):
        tensor = torch.full((len(seqs), maxlen), pad_value, dtype=torch.long)
        for i, seq in enumerate(seqs):
            tensor[i, :len(seq)] = torch.LongTensor(seq)
        return tensor.to(DEVICE)

    wids    = pad(wids_list)
    sids    = pad(sids_list)
    iid     = torch.LongTensor(intent_list).to(DEVICE)
    lengths = torch.LongTensor(lengths)

    return {
        "wids":    wids,      # (B, T)
        "sids":    sids,      # (B, T)
        "iid":     iid,       # (B,)
        "lengths": lengths    # (B,)
    }

def make_loader(exs, lang, bs, shuffle):
    ds = ATISDataset(exs, lang)
    return DataLoader(
      ds, batch_size=bs, shuffle=shuffle,
      collate_fn=collate_fn
    )
