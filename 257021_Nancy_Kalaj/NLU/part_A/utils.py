import os
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_atis():
    """
    Make sure the ATIS JSONs and conll.py are downloaded
    """
    os.makedirs("dataset/ATIS", exist_ok=True)
    if not os.path.exists("dataset/ATIS/train.json"):
        os.system("wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json")
    if not os.path.exists("dataset/ATIS/test.json"):
        os.system("wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json")
    if not os.path.exists("dataset/ATIS/conll.py"):
        os.system("wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py")


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


def make_loader(dataset, lang, bs=32, shuffle=False, collate_fn=None):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn)