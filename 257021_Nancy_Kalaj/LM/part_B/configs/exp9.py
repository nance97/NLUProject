# exp8: same as exp7 but ASGD lr tuned to 2e-3
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      300,
    "dropout":       0.5,
    "embed_dropout": 0.5,
    "optimizer":     "AdamW",
    "lr":            1e-3,
    "weight_decay":  0.05,
    "use_var_drop":  True,
    "use_avsgd":     True,
    "asgd_lr":       2e-3,
    "weight_tying":  True,
}
