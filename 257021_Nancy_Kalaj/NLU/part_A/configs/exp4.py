# exp4: increased learning rate from 1e-4 to 1e-3
CFG = {
    "model_type":   "ModelIAS", 
    "hid_size":     200,
    "emb_size":     300,
    "dropout":      0.1,
    "optimizer":    "AdamW",
    "lr":           1e-3,
    "weight_decay": 1e-2,
}