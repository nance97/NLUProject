# exp2: bidirectional LSTM with added dropout
CFG = {
    "model_type":   "ModelIAS", 
    "hid_size":     200,
    "emb_size":     300,
    "dropout":      0.1,
    "optimizer":    "SGD",
    "lr":           1e-4,
    "weight_decay": 0.0,
    "n_runs":       5,
}