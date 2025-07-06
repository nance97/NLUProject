# exp5: weight tying (hid_size==emb_size), tune wd=0.05
CFG = {
    "model_type":    "LM_LSTM",
    "emb_size":      300,
    "hid_size":      300,
    "dropout":       0.1,
    "embed_dropout": 0.1,
    "optimizer":     "AdamW",
    "lr":            1e-3,
    "weight_decay":  0.05,
    "use_avsgd":     False,
    "asgd_lr":       None,
    "weight_tying":  True,
}
