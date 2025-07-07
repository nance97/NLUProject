import os
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import urllib.request

PAD_TOKEN = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def load_data(path):
    with open(path) as f:
        return json.load(f)


def prepare_splits(train_path, test_path, dev_portion=0.1, seed=42):
    raw_train = load_data(train_path)
    raw_test = load_data(test_path)

    # stratify by intent
    intents = [x['intent'] for x in raw_train]
    train_split = []
    rare = []
    for x in raw_train:
        if intents.count(x['intent']) > 1:
            train_split.append(x)
        else:
            rare.append(x)
    X, dev, _, _ = train_test_split(
        train_split, [x['intent'] for x in train_split],
        test_size=dev_portion, random_state=seed, stratify=[x['intent'] for x in train_split]
    )
    X.extend(rare)
    return X, dev, raw_test


class Lang:
    def __init__(self, train, dev, test, cutoff=0):
        words = sum([x['utterance'].split() for x in train], [])
        slots = set(sum([x['slots'].split() for x in (train+dev+test)], []))
        intents = set([x['intent'] for x in (train+dev+test)])
        self.word2id = self._w2id(words, cutoff)
        self.slot2id = self._l2id(slots)
        self.intent2id = self._l2id(intents, pad=False)
        self.id2word = {v:k for k,v in self.word2id.items()}
        self.id2slot = {v:k for k,v in self.slot2id.items()}
        self.id2intent = {v:k for k,v in self.intent2id.items()}

    def _w2id(self, elems, cutoff=0, unk=True):
        count = Counter(elems)
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        for w, c in count.items():
            if c > cutoff:
                vocab[w] = len(vocab)
        return vocab

    def _l2id(self, elems, pad=True):
        vocab = {'pad': PAD_TOKEN} if pad else {}
        for e in sorted(elems):
            vocab[e] = len(vocab)
        return vocab
    

def collate_fn(batch):
    """
    Custom collate to pad utterances and slots and preserve lengths
    """
    def merge(sequences, pad_value=PAD_TOKEN):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :lengths[i]] = torch.tensor(seq, dtype=torch.long)
        return padded, torch.tensor(lengths, dtype=torch.long)

    # sort by utterance length desc
    batch.sort(key=lambda x: len(x['utterance']), reverse=True)
    utts = [x['utterance'] for x in batch]
    slots = [x['slots'] for x in batch]
    intents = [x['intent'] for x in batch]

    utt_tensor, utt_lengths = merge(utts, PAD_TOKEN)
    slot_tensor, _ = merge(slots, PAD_TOKEN)
    intent_tensor = torch.tensor(intents, dtype=torch.long)

    return {
        'utterances': utt_tensor.to(DEVICE),
        'slots_len': utt_lengths.to(DEVICE),
        'y_slots': slot_tensor.to(DEVICE),
        'intents': intent_tensor.to(DEVICE)
    }


def make_loader(dataset, lang, bs=32, shuffle=False, collate_fn=collate_fn):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn)