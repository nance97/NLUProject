# exp7: NT‐AvSGD (switch after patience=3) with asgd_lr=1e-3
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      300,
    "dropout":       0.5,
    "embed_dropout": 0.5,
    "optimizer":     "AdamW",    # start with AdamW
    "lr":            1e-3,
    "weight_decay":  0.05,
    "use_var_drop":  True,
    "use_avsgd":     True,
    "asgd_lr":       1e-3,       # ASGD kick‐in lr
    "weight_tying":  True,
}
