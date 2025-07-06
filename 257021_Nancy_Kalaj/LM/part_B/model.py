import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super(LockedDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        # x has shape: (batch_size, seq_len, feature_dim)
        if not self.training or self.dropout == 0:
            return x
        # Create dropout mask of shape (batch_size, 1, feature_dim)
        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask

class LM_LSTM(nn.Module):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1,
        emb_dropout: float = 0.1,
        fc_dropout:  float = 0.1
    ):
        super().__init__()
        # embeddings → LSTM → projection
        self.embedding  = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout: nn.Module = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc_dropout: nn.Module = nn.Dropout(fc_dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T)
        emb, _ = self.lstm(self.emb_dropout(self.embedding(x)))
        # emb: (B, T, H)
        out = self.fc_dropout(emb)
        # project & permute to (B, V, T)
        return self.output_layer(out).permute(0, 2, 1)
