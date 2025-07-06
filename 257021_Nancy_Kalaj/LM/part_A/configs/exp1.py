# exp1: RNN with SGD, lr=2.0, no dropout
CFG = {
    "model_type":    "RNN_cell",
    "emb_size":      300,
    "hid_size":      200,
    "dropout":       0.0,
    "embed_dropout": 0.0,
    "optimizer":     "SGD",
    "lr":            2.0,
    "weight_decay":  0.0,
    "use_avsgd":     False,
    "asgd_lr":       None,
    "weight_tying":  False,
}