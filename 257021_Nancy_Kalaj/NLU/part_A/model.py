import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, n_slots, n_intents,
                 pad_idx=0, n_layers=1, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(drop)
        self.encoder = nn.LSTM(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=(drop if n_layers>1 else 0.0)
        )
        self.slot_out  = nn.Linear(hid_size*2, n_slots)
        self.intent_out = nn.Linear(hid_size*2, n_intents)

    def forward(self, wids, lengths):
        emb = self.dropout(self.embedding(wids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.encoder(packed)
        seq_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        seq_out = self.dropout(seq_out)
        # combine forward/back final states for intent:
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        slot_logits = self.slot_out(seq_out).permute(0,2,1)
        intent_logits = self.intent_out(h_cat)
        return slot_logits, intent_logits
