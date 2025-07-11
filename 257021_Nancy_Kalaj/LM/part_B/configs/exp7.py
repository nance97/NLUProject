# exp7: variational (locked) dropout p=0.5 + weight tying
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      300,
    "dropout":       0.5,     # LockedDropout
    "embed_dropout": 0.5,     # LockedDropout
    "optimizer":     "AdamW",
    "lr":            1e-3,
    "weight_decay":  0.05,
    "weight_tying":  True,
    "use_var_drop":  True,
    "use_avsgd":     False,
    "asgd_lr":       None,
}
