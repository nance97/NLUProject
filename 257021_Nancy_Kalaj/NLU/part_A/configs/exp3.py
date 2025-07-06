# exp3: switched optimizer to AdamW with weight decay
CFG = {
    "model_type":   "ModelIAS", 
    "hid_size":     200,
    "emb_size":     300,
    "dropout":      0.1,
    "optimizer":    "AdamW",
    "lr":           1e-4,
    "weight_decay": 1e-2,
    "n_runs":       5,
}