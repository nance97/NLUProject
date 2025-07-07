import os
import json
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizerFast

# where to download ATIS if missing
ATIS_URLS = {
    "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
    "test.json":  "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json",
}

def ensure_atis(atis_dir="dataset/ATIS"):
    os.makedirs(atis_dir, exist_ok=True)
    for fname, url in ATIS_URLS.items():
        dest = os.path.join(atis_dir, fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, dest)

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def create_raws(tmp_train_raw, dev_size=0.10, seed=42):
    # Count how often each intent occurs
    intents     = [ex["intent"] for ex in tmp_train_raw]
    count_y     = Counter(intents)
    inputs      = []     # examples with intent freq > 1
    labels      = []     # their intents (for stratification)
    mini_train  = []     # singleton-intent examples

    # Split off singletons
    for ex, intent in zip(tmp_train_raw, intents):
        if count_y[intent] > 1:
            inputs.append(ex)
            labels.append(intent)
        else:
            mini_train.append(ex)

    # Stratified split on the frequent-intent examples
    X_train, X_dev, _, _ = train_test_split(
        inputs,
        labels,
        test_size=dev_size,
        stratify=labels,
        random_state=seed,
        shuffle=True
    )

    # Put all singletons back into the training set
    X_train.extend(mini_train)
    return X_train, X_dev

class Lang:
    """ build vocab for words, slots, intents """
    PAD_TOKEN = 0
    def __init__(self, corpus):
        # corpus is list of examples with 'utterance','slots','intent'
        words = sum([ex["utterance"].split() for ex in corpus], [])
        slots = sum([ex["slots"].split()     for ex in corpus], [])
        intents= [ex["intent"]                for ex in corpus]
        self.word2id   = self._make_w2id(words)
        self.slot2id   = self._make_lab2id(slots)
        self.intent2id = self._make_lab2id(intents, pad=False)
        self.id2slot   = {v:k for k,v in self.slot2id.items()}
        self.id2intent = {v:k for k,v in self.intent2id.items()}

    def _make_w2id(self, elems, unk=True, cutoff=0):
        vocab = {"pad": self.PAD_TOKEN}
        if unk:
            vocab["unk"] = len(vocab)
        ctr = Counter(elems)
        for w,c in ctr.items():
            if c>cutoff:
                vocab[w] = len(vocab)
        return vocab

    def _make_lab2id(self, elems, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = self.PAD_TOKEN
        for e in elems:
            if e not in vocab:
                vocab[e] = len(vocab)
        return vocab

class ATISJointDataset(Dataset):
    """ tokenizes and aligns slot labels for BERT """
    def __init__(self, examples, tokenizer: BertTokenizerFast, lang: Lang, max_length=50):
        self.examples = examples
        self.tokenizer = tokenizer
        self.slot2id = lang.slot2id
        self.intent2id = lang.intent2id
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["utterance"].split()
        slots = ex["slots"].split()
        intent = self.intent2id[ex["intent"]]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        word_ids = enc.word_ids()

        # align slots
        slot_label_ids = []
        for i, wid in enumerate(word_ids):
            if wid is None:
                slot_label_ids.append(-100)
            elif i>0 and wid==word_ids[i-1]:
                slot_label_ids.append(-100)
            else:
                slot_label_ids.append(self.slot2id[slots[wid]])
        slot_label_ids = torch.tensor(slot_label_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_labels": slot_label_ids,
            "intent_label": torch.tensor(intent, dtype=torch.long),
        }

def atis_collate_fn(batch):
    return {
        "input_ids":      torch.stack([ex["input_ids"]      for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch]),
        "slot_labels":    torch.stack([ex["slot_labels"]    for ex in batch]),
        "intent_label":   torch.stack([ex["intent_label"]   for ex in batch]),
    }

def prepare_data(
    train_path: str,
    test_path: str,
    tokenizer_name: str="bert-base-uncased",
    max_length: int=50,
    batch_size: int=128
):
    ensure_atis(os.path.dirname(train_path))
    train_raw = load_data(train_path)
    test_raw  = load_data(test_path)
    train_raw, dev_raw = create_raws(train_raw)

    corpus = train_raw + dev_raw + test_raw
    lang = Lang(corpus)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    train_ds = ATISJointDataset(train_raw, tokenizer, lang, max_length)
    dev_ds   = ATISJointDataset(dev_raw,   tokenizer, lang, max_length)
    test_ds  = ATISJointDataset(test_raw,  tokenizer, lang, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=atis_collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, collate_fn=atis_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=atis_collate_fn)

    return tokenizer, train_loader, dev_loader, test_loader, lang
