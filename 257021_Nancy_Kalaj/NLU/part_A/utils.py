import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import urllib.request

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constant for the padding token's ID in all vocabularies
PAD_TOKEN = 0

ATIS_URLS = {
    "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
    "test.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json",
}

# Downloads ATIS files into the specified directory if they are not already present
def ensure_atis(atis_dir="dataset/ATIS"):
    os.makedirs(atis_dir, exist_ok=True)
    for fname, url in ATIS_URLS.items():
        dest = os.path.join(atis_dir, fname) if fname.endswith('.json') else os.path.join(os.getcwd(), fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, dest)

# Handles vocabulary building and provides mappings for words, slots, and intents
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        # Build mappings for words (with UNK and PAD), slots, and intents
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    # Maps words to IDs, using cutoff for frequency filtering and adding unk/pad tokens
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    # Maps labels (slots or intents) to IDs, optionally including the pad token
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

# Dataset class: represents a list of utterances/intents/slots as torch tensors with mapped IDs
class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        # Store raw utterances, slot sequences, and intent labels
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # Map all data to their corresponding IDs
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        # Each sample is a dictionary of IDs for utterance, slots, and intent
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        # Maps labels to their integer IDs (uses 'unk' for out-of-vocab)
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        # Maps sequences (utterances or slot label sequences) to integer ID sequences
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

# Pads batch of variable-length sequences and stacks them for efficient GPU processing
def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # Pad each sequence as needed
        padded_seqs = padded_seqs.detach()  # No grad tracking for batching
        return padded_seqs, lengths
    # Sort batch by sequence length (descending) for efficient packing in LSTM
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

# Loads data from a JSON file and returns as a list of dicts
def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

# Splits the training data into actual train and validation/dev splits,
# ensuring at least one example per intent label in train
def prepare_splits(tmp_train_raw):
    labels = []
    inputs = []
    mini_train = []

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    portion = 0.10

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion,
        random_state=42, shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw

def prepare_loaders(train_path, test_path, batch_size=128):
    raw_train = load_data(train_path)
    raw_test = load_data(test_path)

    # split off a dev set (singleton-intent examples stay in train)
    train_raw, dev_raw = prepare_splits(raw_train)

    # build vocab on all words, slots and intents across splits
    all_words = sum([ex["utterance"].split() for ex in train_raw], [])
    corpus = train_raw + dev_raw + raw_test
    all_slots = set(sum([ex["slots"].split() for ex in corpus], []))
    all_ints = set([ex["intent"] for ex in corpus])
    lang = Lang(all_words, all_ints, all_slots, cutoff=0)

    train_ds = IntentsAndSlots(train_raw, lang)
    dev_ds = IntentsAndSlots(dev_raw, lang)
    test_ds = IntentsAndSlots(raw_test, lang)

    # build loaders with padding collation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, lang
