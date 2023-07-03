# coding=utf-8
# author=yphacker

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

from conf import model_config_bert as model_config

from conf import config






class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_config.pretrain_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.classifier = nn.Linear(768, config.num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        cls_output = outputs[1]
        cls_output=torch.flatten(cls_output,1)
        logits = self.classifier(cls_output)
        # print(cls_output.size(),logits.size(),len(outputs))
        return cls_output,logits

