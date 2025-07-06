import torch.nn as nn

class LockedDropout(nn.Module):
    """
    Applies the same dropout mask at every time-step ("locked" or "variational" dropout).
    This encourages consistent feature-level regularization across the sequence dimension.

    Attributes:
        dropout (float): probability of dropping each feature channel.
    """
    def __init__(self, dropout: float):
        super(LockedDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass for locked dropout.

        Args:
            x (Tensor): input activations of shape (batch_size, seq_len, feature_dim).
        Returns:
            Tensor: same shape as x, with a fixed dropout mask applied across timesteps.
        """
        # In evaluation mode or if no dropout, return input unchanged
        if not self.training or self.dropout == 0:
            return x
        # Create a dropout mask of shape (batch_size, 1, feature_dim)
        # each channel is dropped/kept consistently over the entire sequence
        mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(1 - self.dropout)
        # scale mask to preserve expectation
        mask = mask.div_(1 - self.dropout)
        # expand mask to full sequence length
        mask = mask.expand_as(x)
        # apply mask
        return x * mask

class LM_LSTM(nn.Module):
    """
    LSTM-based language model with optional variational dropout.

    Architecture:
        Embedding -> Embedding Dropout -> LSTM -> Output Dropout -> Linear -> Permute

    Args:
        emb_size (int): dimensionality of token embeddings
        hidden_size (int): size of LSTM hidden state
        output_size (int): vocabulary size
        pad_index (int): padding token index
        n_layers (int): number of stacked LSTM layers
        emb_dropout (float): dropout probability on embeddings
        fc_dropout (float): dropout probability after LSTM
    """
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
        # Token embedding layer with padding
        self.embedding  = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Apply dropout to embeddings to regularize input (variational dropout applied here if use_var_drop = True)
        self.emb_dropout: nn.Module = nn.Dropout(emb_dropout)
        # LSTM processes the entire sequence        
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        # Dropout on LSTM outputs for regularization (variational dropout applied here if use_var_drop = True)
        self.fc_dropout: nn.Module = nn.Dropout(fc_dropout)
        # Linear projection to vocabulary space
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM language model.

        Args:
            x (Tensor): shape (batch_size, seq_len) of token indices
        Returns:
            Tensor: shape (batch_size, vocab_size, seq_len) of logits
        """
        # Embed and apply dropout: (B, T, emb_size)
        emb, _ = self.lstm(self.emb_dropout(self.embedding(x)))
        # Apply dropout to LSTM outputs
        out = self.fc_dropout(emb)
        # Project to vocab logits and permute axes to (B, V, T)
        return self.output_layer(out).permute(0, 2, 1)
