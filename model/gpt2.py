# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import GPT2Model,  GPT2Tokenizer
# from transformers import XLMRobertaModel, XLMRobertaTokenizer
from conf import config
from conf import model_config_bert as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_config.pretrain_model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_config.pretrain_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        outputs = self.gpt2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        hidden_states = outputs[0]
        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        mean_pool = torch.mean(h12, 1)
        logits = self.classifier(mean_pool)
        return mean_pool,logits
