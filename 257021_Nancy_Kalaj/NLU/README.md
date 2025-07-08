# Joint Intent Classification & Slot Filling

Second NLU assignment (parts A and B):

* **Part A**: Modify the baseline architecture Model IAS by:
1. Adding bidirectionality 
2. Adding dropout layers
* **Part B**: Fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling.

---

## Part A: How to Train & Evaluate

### How to Train

```bash
python main.py --exp exp1
```

This will:
- Download ATIS JSONs if missing  
- Build your vocabulary, `DataLoader`s, and model per `configs/exp1.py`  
- Train with early stopping (patience = 3) on dev-set slot F1  
- Save best weights to:
  ```
  bin/exp1_best_model.pt
  ```

### How to Evaluate a Saved Checkpoint

```bash
python main.py --exp exp1 --test
```

This skips training, loads  
```
bin/exp1_best_model.pt
```  
and prints slot-F1 and intent accuracy on the ATIS test set.

---

###  Part B: How to Train & Evaluate

### How to Train

```bash
python main.py
```

If you want to modify some of the training parameters you can use the following flags (default values are shown):

```bash
python main.py   --batch_size 16   --max_length 50   --epochs 100   --lr 3e-5
```

This will:
- Download ATIS data if missing  
- Tokenize, align slot labels, and build `DataLoader`s  
- Fine-tune BERT over 5 runs (each with its own seed)  
- Report mean ± std of slot-F1 and intent accuracy  
- Save the best run’s weights to:
  ```
  bin/best_bert_<YYYYMMDD-HHMMSS>.pt
  ```

### How to Evaluate a Saved Checkpoint

```bash
python bert_main.py --test --model_path bin/best_bert_<timestamp>.pt
```

You **must** pass the exact path via `--model_path`. The script will:
- Rebuild the same BERT architecture  
- Load weights from the given `.pt`  
- Run `eval_loop_bert` on ATIS test set  
- Print slot-F1 and intent accuracy  

---

## Configurations

- **Part A:** live under `configs/expX.py`.
- **Part B:** (batch size, learning rate, max sequence length, etc.) are passed via CLI flags in `main.py`.  

---