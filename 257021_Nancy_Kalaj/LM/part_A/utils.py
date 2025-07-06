import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import os
import urllib.request

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PTB_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt":  "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt",
}

def ensure_ptb(download_dir="dataset/PennTreeBank"):
    os.makedirs(download_dir, exist_ok=True)
    for fname, url in PTB_URLS.items():
        local_path = os.path.join(download_dir, fname)
        if not os.path.exists(local_path):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, local_path)

def read_file(path, eos_token="<eos>"):
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line.strip() + " " + eos_token)
    return lines

def get_vocab(corpus, special_tokens=None):
    vocab = {}
    idx = 0
    if special_tokens:
        for tok in special_tokens:
            vocab[tok] = idx; idx += 1
    for sent in corpus:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = idx; idx += 1
    return vocab

class Lang:
    def __init__(self, corpus, special_tokens=None):
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {i:w for w,i in self.word2id.items()}

class PennTreebank(Dataset):
    def __init__(self, corpus, lang):
        self.src = [s.split()[:-1] for s in corpus]
        self.tgt = [s.split()[1:]  for s in corpus]
        self.lang = lang

        self.src_ids = [[lang.word2id[w] for w in sent] for sent in self.src]
        self.tgt_ids = [[lang.word2id[w] for w in sent] for sent in self.tgt]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, i):
        return {
            "source": torch.LongTensor(self.src_ids[i]),
            "target": torch.LongTensor(self.tgt_ids[i])
        }

def collate_fn(batch, pad_token):
    # pad to max length in batch
    sources = [b["source"] for b in batch]
    targets = [b["target"] for b in batch]
    lengths = [len(x) for x in sources]
    maxlen  = max(lengths)

    def pad(seqs):
        res = torch.full((len(seqs), maxlen), pad_token, dtype=torch.long)
        for i, s in enumerate(seqs):
            res[i, :len(s)] = s
        return res

    src_padded = pad(sources).to(DEVICE)
    tgt_padded = pad(targets).to(DEVICE)
    n_tokens   = sum(lengths)

    return {
        "source":      src_padded,
        "target":      tgt_padded,
        "n_tokens":    n_tokens,
        "lengths":     lengths
    }

def make_loader(dataset, batch_size, pad_token, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, pad_token=pad_token)
    )
