# utils.py
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
import os
import urllib.request

# Automatically use GPU if available
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# URLs for Penn Treebank splits to ensure reproducibility
PTB_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt":  "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt",
}

def ensure_ptb(download_dir="dataset/PennTreeBank"):
    """
    Ensure that PTB train/valid/test files exist locally.
    Downloads them into `download_dir` if missing.

    Args:
        download_dir (str): directory path to store PTB files.
    """
    os.makedirs(download_dir, exist_ok=True)
    for fname, url in PTB_URLS.items():
        local_path = os.path.join(download_dir, fname)
        if not os.path.exists(local_path):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, local_path)


def read_file(path, eos_token="<eos>"):
    """
    Read raw text lines from a PTB file, appending an <eos> token.

    Args:
        path (str): path to the PTB text file.
        eos_token (str): token to append at end of each sentence.

    Returns:
        List[str]: lines with appended eos_token.
    """
    lines = []
    with open(path, "r") as f:
        for line in f:
            # strip trailing newline and add EOS marker
            lines.append(line.strip() + " " + eos_token)
    return lines


def get_vocab(corpus, special_tokens=None):
    """
    Build vocabulary mapping from tokens to integer IDs.

    Args:
        corpus (List[str]): list of sentences (strings).
        special_tokens (List[str] or None): tokens to reserve at the front (e.g., '<pad>', '<eos>').

    Returns:
        dict: token -> unique integer ID.
    """
    vocab = {}
    idx = 0
    # assign IDs to special tokens first, if any
    if special_tokens:
        for tok in special_tokens:
            vocab[tok] = idx
            idx += 1
    # then map every token in corpus
    for sent in corpus:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


class Lang:
    """
    Simple vocabulary holder: token->id and id->token.
    """
    def __init__(self, corpus, special_tokens=None):
        # build forward and reverse mappings
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {i: w for w, i in self.word2id.items()}


class PennTreebank(Dataset):
    """
    PyTorch Dataset for PTB language modeling.

    Stores source/target token ID sequences for next-word prediction.
    """
    def __init__(self, corpus, lang):
        # split each sentence into tokens, dropping EOS for source and BOS for target
        self.src = [s.split()[:-1] for s in corpus]
        self.tgt = [s.split()[1:]  for s in corpus]
        self.lang = lang

        # convert words to IDs
        self.src_ids = [[lang.word2id[w] for w in sent] for sent in self.src]
        self.tgt_ids = [[lang.word2id[w] for w in sent] for sent in self.tgt]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, i):
        # returns a dict with LongTensors for source and target
        return {
            "source": torch.LongTensor(self.src_ids[i]),
            "target": torch.LongTensor(self.tgt_ids[i])
        }


def collate_fn(batch, pad_token):
    """
    Custom collate function to pad variable-length batches.

    Pads all sequences in the batch to the same max length.
    """
    # extract lists of tensors
    sources = [b["source"] for b in batch]
    targets = [b["target"] for b in batch]
    lengths = [len(x) for x in sources]
    maxlen  = max(lengths)

    def pad(seqs):
        # create a (batch_size, maxlen) tensor filled with pad_token
        res = torch.full((len(seqs), maxlen), pad_token, dtype=torch.long)
        for i, s in enumerate(seqs):
            res[i, :len(s)] = s
        return res

    # pad sources and targets, move to DEVICE
    src_padded = pad(sources).to(DEVICE)
    tgt_padded = pad(targets).to(DEVICE)
    n_tokens   = sum(lengths)

    return {
        "source":   src_padded,
        "target":   tgt_padded,
        "n_tokens": n_tokens,
        "lengths":  lengths
    }


def make_loader(dataset, batch_size, pad_token, shuffle=False):
    """
    Convenience wrapper for creating DataLoaders with padding.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, pad_token=pad_token)
    )
