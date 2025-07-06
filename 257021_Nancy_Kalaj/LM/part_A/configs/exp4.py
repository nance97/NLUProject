# exp4: switch to AdamW, lr=1e-3
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      200,
    "dropout":       0.1,
    "embed_dropout": 0.1,
    "optimizer":     "AdamW",
    "lr":            1e-3,
    "weight_decay":  0.01,
}
