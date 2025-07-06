# exp3: add standard dropout p=0.1
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      200,
    "dropout":       0.1,
    "embed_dropout": 0.1,
    "optimizer":     "SGD",
    "lr":            2.0,
    "weight_decay":  0.0,
    "use_avsgd":     False,
    "asgd_lr":       None,
    "weight_tying":  False,
}
