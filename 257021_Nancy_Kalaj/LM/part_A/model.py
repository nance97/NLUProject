import torch.nn as nn

class LM_RNN(nn.Module):
    """
    Vanilla RNN-based language model (Elman RNN).

    Args:
        emb_size (int): dimensionality of token embeddings
        hidden_size (int): size of RNN hidden state
        output_size (int): vocabulary size (number of tokens including pad/eos)
        pad_index (int): index of the padding token in vocabulary
        out_dropout (float): dropout probability (not used here but retained for API consistency)
        emb_dropout (float): dropout probability for embeddings (not used in this simple RNN)
        n_layers (int): number of stacked RNN layers
    """
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=0.1,
        emb_dropout=0.1,
        n_layers=1
    ):
        super(LM_RNN, self).__init__()
        # Embedding layer maps token IDs to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=emb_size,
            padding_idx=pad_index
        )
        # Single-direction RNN layer processes the sequence
        self.rnn = nn.RNN(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True
        )
        # Store padding index for reference
        self.pad_token = pad_index
        # Final linear layer projects hidden state to vocabulary logits
        self.output = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, input_sequence):
        """
        Forward pass for the RNN language model.

        Args:
            input_sequence (Tensor): shape (batch_size, seq_len) of token indices
        Returns:
            Tensor: shape (batch_size, vocab_size, seq_len) of logits for each time-step
        """
        # Convert token IDs to embeddings: (B, T, emb_size)
        emb = self.embedding(input_sequence)
        # Run through RNN: (B, T, hidden_size)
        rnn_out, _ = self.rnn(emb)
        # Project to vocabulary logits and permute to (B, V, T)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(nn.Module):
    """
    LSTM-based language model with optional dropout.

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
        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=emb_size,
            padding_idx=pad_index
        )
        # Apply dropout to embeddings to regularize input
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        # LSTM processes the entire sequence
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        # Dropout on LSTM outputs for regularization
        self.fc_dropout = nn.Dropout(p=fc_dropout)
        # Linear projection to vocabulary space
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, x):
        """
        Forward pass for the LSTM language model.

        Args:
            x (Tensor): shape (batch_size, seq_len) of token indices
        Returns:
            Tensor: shape (batch_size, vocab_size, seq_len) of logits
        """
        # Embed and apply dropout: (B, T, emb_size)
        emb = self.embedding(x)
        emb = self.emb_dropout(emb)
        # LSTM layer: (B, T, hidden_size)
        lstm_out, _ = self.lstm(emb)
        # Apply dropout to LSTM outputs
        out = self.fc_dropout(lstm_out)
        # Project to vocab logits and permute axes to (B, V, T)
        logits = self.output_layer(out).permute(0, 2, 1)
        return logits
