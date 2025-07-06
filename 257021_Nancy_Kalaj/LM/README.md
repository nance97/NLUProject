# Neural Language Modeling

First NLU assignment (parts A and B):

* **Part A**:
1. Replace RNN with a Long-Short Term Memory (LSTM) network
2. Add two dropout layers:
3. Replace SGD with AdamW
* **Part B**: 
1. Weight Tying
2. Variational Dropout
3. Non-monotonically Triggered AvSGD

---

### How to Train & Evaluate

Run any experiment by name. For example:

```bash
python main.py --exp exp3
```

This will:

* download PTB data if missing
* build vocab, datasets, and DataLoaders
* instantiate the model per `configs/exp3.py`
* train with early stopping on dev perplexity
* evaluate and save the best model to `bin/exp3_best_model.pt`

To skip training and only evaluate a saved checkpoint:

```bash
python LM/part_A/main.py --exp exp3 --test
```

### Configurations

All experiment-specific settings live under `configs/expX.py`. You can inspect or duplicate them to define new variants.

---
