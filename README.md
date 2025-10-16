# 🧠 NLUProject – Language Modeling and Joint Intent–Slot Understanding

This project was developed for the **Natural Language Understanding (NLU)** course at the **University of Trento**, and is organized into two complementary parts:  
(1) **Language Modeling (LM)** on the Penn Treebank dataset, and  
(2) **Joint Intent Classification and Slot Filling (NLU)** on the ATIS dataset.  

Together, they form a complete exploration of recurrent and transformer-based architectures for understanding and generating natural language.

---

## 📘 Part 1 – Language Modeling (LM)

**Goal:** progressively enhance a vanilla RNN language model to achieve **test perplexity below 100** on the Penn Treebank corpus.

### 🔧 Key Techniques
- **LSTM replacement** for long-range dependency modeling  
- **Dropout regularization** to mitigate overfitting  
- **AdamW optimizer** with weight decay tuning  
- **Weight tying** between embedding and output projection layers  
- **Variational (Locked) Dropout** for consistent regularization across timesteps  
- **Non-Monotonic Triggered Averaged SGD (NT-AvSGD)** for smoother convergence  

### 📈 Results
| Stage | Key Change | Test Perplexity |
|--------|-------------|----------------|
| Vanilla RNN | Baseline | > 6000 |
| + LSTM | Long-term dependencies | 142.6 |
| + Dropout | Regularization | 123.3 |
| + AdamW (wd=0.05) | Optimizer tuning | 113.1 |
| + Weight tying | Fewer parameters | 103.4 |
| + Locked dropout | Temporal consistency | 91.1 |
| + NT-AvSGD | Optimized convergence | **89.93** |

**Final Test Perplexity:** 89.93  
**Dataset:** Penn Treebank  
**Reference:** Merity et al., *Regularizing and Optimizing LSTM Language Models*, 2017.

---

## 💬 Part 2 – Joint Intent Classification & Slot Filling (NLU)

**Goal:** refine and compare recurrent and transformer-based models for joint intent detection and slot tagging on the **ATIS** dataset.

### 🔧 Experimental Setup
- **Part A – LSTM Enhancements**
  - Bidirectional LSTM encoding  
  - Dropout regularization (p=0.1)  
  - AdamW optimizer with LR tuning  
- **Part B – BERT Fine-Tuning**
  - `bert-base-uncased` backbone  
  - Subtoken-aware slot label alignment  
  - Multi-task fine-tuning (slot + intent losses)  
  - Early stopping on slot-F1 validation  

### 📈 Results

| Model | Slot F1 (mean ± std) | Intent Accuracy (mean ± std) | Best Single Run |
|--------|----------------------|-------------------------------|------------------|
| BiLSTM + Dropout + AdamW + LR Tuning | 0.946 ± 0.002 | 0.951 ± 0.004 | 0.945 |
| **BERT-base (joint fine-tuning)** | **0.9548 ± 0.0015** | **0.9709 ± 0.0050** | **0.9564 / 0.9608** |

**Dataset:** ATIS  
**Reference:** Chen et al., *BERT for Joint Intent Classification and Slot Filling*, 2019.

---

## 🧩 Implementation Highlights

- Modular design: shared evaluation and preprocessing pipeline  
- Early stopping, multi-run aggregation, and CoNLL evaluation script integration  
- Subtoken alignment for robust slot labeling in transformer fine-tuning  
- Gradient clipping and optimizer scheduling for stability  

---

## 🧠 Insights

- Architectural refinements and regularization can still yield strong results for LSTMs when tuned carefully.  
- BERT significantly outperforms recurrent baselines but requires careful slot–token alignment to reach full potential.  
- Both projects reinforce the importance of **representation learning** and **task-specific optimization** in modern NLU.

---

## 📎 References

- S. Merity, N. S. Keskar, and R. Socher, *Regularizing and Optimizing LSTM Language Models*, arXiv:1708.02182 (2017)  
- Q. Chen, Z. Zhuo, and W. Wang, *BERT for Joint Intent Classification and Slot Filling*, arXiv:1902.10909 (2019)

---

**Author:** Nancy Kalaj  
**University of Trento – MSc in Artificial Intelligence Systems**  
📧 nancy.kalaj@studenti.unitn.it
