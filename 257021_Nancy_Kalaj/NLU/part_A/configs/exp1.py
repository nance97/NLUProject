# exp1: bidirectional LSTM, no dropout
CFG = {
    "model_type":   "ModelIAS", 
    "hid_size":     200,
    "emb_size":     300,
    "dropout":      0.0,
    "optimizer":    "Adam",
    "lr":           1e-4,
    "weight_decay": 0.0,
}