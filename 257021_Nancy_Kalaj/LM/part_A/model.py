import torch.nn as nn

class RNN_cell(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()
        self.W       = nn.Linear(input_size,  hidden_size, bias=False)
        self.U       = nn.Linear(hidden_size, hidden_size)
        self.V       = nn.Linear(hidden_size, vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden, word_emb):
        h_in  = self.W(word_emb)           # (batch, hidden_size)
        h_prev= self.U(prev_hidden)        # (batch, hidden_size)
        h_t   = self.sigmoid(h_in + h_prev)
        y_t   = self.V(h_t)               # (batch, vocab_size)
        return h_t, y_t
    
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output 

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
        self.embedding    = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout  = nn.Dropout(emb_dropout)
        self.lstm         = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc_dropout   = nn.Dropout(fc_dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T)
        emb, _ = self.lstm(self.emb_dropout(self.embedding(x)))
        # emb: (B, T, H)
        out = self.fc_dropout(emb)
        # project & permute to (B, V, T)
        return self.output_layer(out).permute(0, 2, 1)
