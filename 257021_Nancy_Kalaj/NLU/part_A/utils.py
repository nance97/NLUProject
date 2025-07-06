import os, json, torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = 0

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
            "wids":  torch.LongTensor(wids),
            "sids":  torch.LongTensor(sids),
            "iid":   torch.LongTensor([iid])
        }

def pad_collate(batch):
    # pads wids and sids to longest in batch, records lengths
    wseqs = [b["wids"] for b in batch]
    sseqs = [b["sids"] for b in batch]
    lengths = [len(x) for x in wseqs]
    maxlen = max(lengths)
    def pad(seqs):
        out = torch.full((len(seqs), maxlen), PAD_TOKEN, dtype=torch.long)
        for i,s in enumerate(seqs): out[i,:len(s)] = s
        return out
    return {
      "wids": pad(wseqs).to(DEVICE),
      "sids": pad(sseqs).to(DEVICE),
      "iid":  torch.cat([b["iid"] for b in batch],0).to(DEVICE),
      "lengths": torch.LongTensor(lengths).to(DEVICE)
    }

def make_loader(exs, lang, bs, shuffle):
    ds = ATISDataset(exs, lang)
    return DataLoader(
      ds, batch_size=bs, shuffle=shuffle,
      collate_fn=pad_collate
    )
