import torch.nn as nn
from transformers import BertModel

class BertForJointIntentSlot(nn.Module):
    def __init__(self, pretrained_model_name: str, n_intents: int, n_slots: int, dropout=0.1):
        super().__init__()
        self.bert   = BertModel.from_pretrained(pretrained_model_name)
        h = self.bert.config.hidden_size
        self.drop  = nn.Dropout(dropout)
        self.intent = nn.Linear(h, n_intents)
        self.slots  = nn.Linear(h, n_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq, pooled = outputs.last_hidden_state, outputs.pooler_output
        seq = self.drop(seq)
        pooled = self.drop(pooled)
        slot_logits   = self.slots(seq)     # (B, L, n_slots)
        intent_logits = self.intent(pooled) # (B, n_intents)
        return slot_logits, intent_logits
