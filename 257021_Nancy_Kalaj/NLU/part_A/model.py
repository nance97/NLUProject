import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, n_slots, n_intents,
                 pad_idx=0, n_layers=1, drop=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(drop)
        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=drop if n_layers>1 else 0.0
        )
        self.slot_out = nn.Linear(hid_size*2, n_slots)
        self.intent_out = nn.Linear(hid_size*2, n_intents)

    def forward(self, utterance, lengths):
        emb = self.embedding(utterance)
        emb = self.dropout(emb)
        packed = pack_padded_sequence(emb, lengths.cpu().numpy(), batch_first=True)
        packed_out, (h_n, _) = self.utt_encoder(packed)
        utt_encoded, _ = pad_packed_sequence(packed_out, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)
        # intent: last forward + backward
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        h_cat = torch.cat([forward_h, backward_h], dim=1)
        slots_logits = self.slot_out(utt_encoded).permute(0,2,1)
        intent_logits = self.intent_out(h_cat)
        return slots_logits, intent_logits