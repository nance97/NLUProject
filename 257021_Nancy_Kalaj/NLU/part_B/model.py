import torch.nn as nn
from transformers import BertModel

class BertJointNLU(nn.Module):
    def __init__(self, pretrained_model_name: str, n_intents: int, n_slots: int, dropout=0.1):
        super().__init__()
        # Load a pretrained BERT backbone for contextualized embeddings
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        h = self.bert.config.hidden_size  # Dimensionality of BERT's output
        self.drop = nn.Dropout(dropout)
        
        self.intent = nn.Linear(h, n_intents)   # Output layer for intent classification (uses [CLS] pooled output)
        self.slots = nn.Linear(h, n_slots)    # Output layer for slot filling (applied to every token)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)   
        seq = outputs.last_hidden_state    # sequence output: contextualized representations for each token
        pooled = outputs.pooler_output    # èooled output: [CLS] token embedding for utterance-level prediction

        seq = self.drop(seq)
        pooled = self.drop(pooled)
        # Predict slot labels for each token (for sequence tagging)
        slot_logits = self.slots(seq)
        # Predict intent label for the whole utterance (using [CLS] embedding)
        intent_logits = self.intent(pooled)
        return slot_logits, intent_logits
