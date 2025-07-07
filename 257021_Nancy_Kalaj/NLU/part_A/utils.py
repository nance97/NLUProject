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
    """
    Builds vocabularies for words, slots, and intents over train+dev+test splits.
    """
    def __init__(self, train, dev, test, cutoff=0):
        # collect tokens
        words = sum([ex['utterance'].split() for ex in (train+dev+test)], [])
        slots = sum([ex['slots'].split()     for ex in (train+dev+test)], [])
        intents = [ex['intent']              for ex in (train+dev+test)]
        # build mappings
        self.word2id   = self._build_w2id(words, cutoff)
        self.slot2id   = self._build_l2id(slots, pad=True)
        self.intent2id = self._build_l2id(intents, pad=False)
        # reverse mappings
        self.id2word   = {i:w for w,i in self.word2id.items()}
        self.id2slot   = {i:s for s,i in self.slot2id.items()}
        self.id2intent = {i:c for c,i in self.intent2id.items()}

    def _build_w2id(self, tokens, cutoff, unk=True):
        count = Counter(tokens)
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        for w,f in count.items():
            if f > cutoff:
                vocab[w] = len(vocab)
        return vocab

    def _build_l2id(self, labels, pad=True):
        unique = []
        seen = set()
        for lab in labels:
            if lab not in seen:
                unique.append(lab)
                seen.add(lab)
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for lab in unique:
            vocab[lab] = len(vocab)
        return vocab

class IntentsAndSlotsDataset(Dataset):
    """
    Wraps raw ATIS examples into token ID sequences using Lang.
    """
    def __init__(self, examples, lang):
        self.examples = examples
        self.lang = lang
        self.utt_ids = [self._map_seq(ex['utterance'].split(), lang.word2id) for ex in examples]
        self.slot_ids= [self._map_seq(ex['slots'].split(),    lang.slot2id) for ex in examples]
        self.int_ids = [lang.intent2id[ex['intent']]                       for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'utterance': self.utt_ids[idx],
            'slots':     self.slot_ids[idx],
            'intent':    self.int_ids[idx]
        }

    def _map_seq(self, tokens, mapper):
        return [mapper.get(tok, mapper.get('unk', PAD_TOKEN)) for tok in tokens]

# notebook-style collate_fn
import torch

def collate_fn(data):
    """
    Notebook-style collate: returns only processed tensors for training/evaluation.
    """
    # inner merge same as in notebook
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded = torch.full((len(sequences), max_len), PAD_TOKEN, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :lengths[i]] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths

    # extract raw lists
    utterances_list = [d['utterance'] for d in data]
    slots_list      = [d['slots']     for d in data]
    intents_list    = [d['intent']    for d in data]

    # sort by utterance length descending
    sorted_idx = sorted(range(len(utterances_list)), key=lambda i: len(utterances_list[i]), reverse=True)
    utterances_list = [utterances_list[i] for i in sorted_idx]
    slots_list      = [slots_list[i]      for i in sorted_idx]
    intents_list    = [intents_list[i]    for i in sorted_idx]

    # merge into padded tensors
    src_utt, lengths   = merge(utterances_list)
    y_slots, _         = merge(slots_list)
    intent_tensor      = torch.tensor(intents_list, dtype=torch.long)

    # return processed batch
    return {
        'utterances': src_utt.to(DEVICE),
        'slots_len':  torch.tensor(lengths, dtype=torch.long).to(DEVICE),
        'y_slots':    y_slots.to(DEVICE),
        'intents':    intent_tensor.to(DEVICE)
    }


def make_loader(dataset, lang, bs=32, shuffle=False, collate_fn=collate_fn):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn)