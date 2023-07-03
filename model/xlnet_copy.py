# # coding=utf-8
# # author=yphacker
# import os
# import csv
# import sys
# sys.path.append("..")
import torch
# from transformers import XLNetTokenizer, XLNetModel
#
import torch.nn as nn
#
# from conf import model_config_bert as model_config
#
# from conf import config
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = XLNetModel.from_pretrained(model_config.pretrain_model_path)
#         self.classifier = nn.Linear(38400, config.num_classes)
#
#
#     def forward(self, input_ids):
#         res=[]
#         for i in range(len(input_ids)):
#             input=input_ids[i]
#             emb_set = []
#             for item in input:
#                 outputs = self.model(item.unsqueeze(0).long())
#                 cls_output = torch.mean(outputs[0], axis=1)
#                 emb_set.append(cls_output)
#             if len(emb_set)>=50:
#                 emb_set=emb_set[:50]
#             else:
#                 deficit=50-len(emb_set)
#                 for i in range(deficit):
#                     emb_set.append(torch.zeros((1,768)))
#             emb_set=torch.cat(emb_set,dim=0)     #torch.Size([50, 768])
#             res.append(emb_set)
#         res=torch.stack(res,dim=0).float()  #torch.Size([16, 50, 768])
#         pre_cls=torch.flatten(res,1)    # torch.Size([16, 38400])
#
#         logits = self.classifier(pre_cls)
#         return pre_cls,logits
#
#
#
# if __name__ == '__main__':
#     input=torch.rand(16,34,80)
#     model=Model()
#     pre,logist=model(input)
#     print(pre.size())
#     print(logist.size())



import nltk
from transformers import XLNetTokenizer, XLNetModel
from conf import model_config_bert as model_config

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = XLNetModel.from_pretrained(model_config.pretrain_model_path)
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_config.pretrain_model_path)
#         self.classifer = nn.Linear(3840, 2)
#
#     def forward(self, input_ids):
#         res_emb = []
#
#         for i in range(len(input_ids)):
#             inputs = input_ids[i]
#             emb_set = []
#             for item in inputs:
#                 input = item.cuda().unsqueeze(0)
#                 outputs = self.model(input)
#                 cls_output = torch.mean(outputs[0], axis=1)
#                 emb_set.append(cls_output)
#             emb_set = torch.cat(emb_set, dim=0)
#             emb_set = torch.flatten(emb_set, 0).unsqueeze(0)
#             res_emb.append(emb_set)
#         res_emb = torch.cat(res_emb, dim=0)
#         # res_emb = torch.flatten(res_emb, 1)
#         logist = self.classifer(res_emb)
#         return res_emb, logist

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = XLNetModel.from_pretrained(model_config.pretrain_model_path)
        self.tokenizer = XLNetTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.classifer = nn.Linear(768, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        cls_output = torch.mean(outputs[0], axis=1)
        cls_output = torch.flatten(cls_output, 1)
        logits = self.classifier(cls_output)
        # print(cls_output.size(),logits.size(),len(outputs))
        return cls_output, logits

    # def forward(self, input_ids):
    #     res_emb=[]
    #
    #     for i in range(len(input_ids)):
    #         print('len input_ids:',len(input_ids))
    #         inputs=input_ids[i]
    #         emb_set = []
    #         for item in inputs:
    #             input=item.cuda().unsqueeze(0)
    #             print('input size:',input.size())
    #             outputs= self.model(input)
    #             cls_output = torch.mean(outputs[0], axis=1)
    #             print('cls_output size:',cls_output.size())
    #             emb_set.append(cls_output)
    #         if len(emb_set) >= 10:
    #             emb_set=emb_set[:10]
    #         else:
    #             deficit=10-len(emb_set)
    #             for i in range(deficit):
    #                 emb_set.append(torch.zeros((1,768)))
    #         emb_set=torch.cat(emb_set, dim=0)
    #         emb_set = torch.flatten(emb_set, 0).unsqueeze(0)
    #         res_emb.append(emb_set)
    #     res_emb = torch.cat(res_emb, dim=0)
    #     print('res_emb size:',res_emb.size())
    #     res_emb=torch.flatten(res_emb,1)
    #     logist=self.classifer(res_emb)
    #     return res_emb, logist





    # def forward(self, input_ids):
    #     res_emb=[]
    #
    #     for i in range(len(input_ids)):
    #         inputs=input_ids[i]
    #         res_item=self.text_embedding(inputs)
    #         res_emb.append(res_item)
    #     res_emb = torch.stack(res_emb, dim=0)
    #     res_emb=torch.flatten(res_emb,1)
    #     logist=self.classifer(res_emb)
    #     return res_emb, logist
    #
    # def text_embedding(self, text):
    #     emb_set=[]
    #     sentences = nltk.sent_tokenize(text)
    #     for sent in sentences:
    #         input_ids = torch.tensor(self.tokenizer.encode(sent)).unsqueeze(0)
    #         outputs= self.model(input_ids)
    #         cls_output = torch.mean(outputs[0], axis=1)
    #         emb_set.append(cls_output)
    #         if len(emb_set)>=50:
    #             emb_set=emb_set[:50]
    #         else:
    #             deficit=50-len(emb_set)
    #             for i in range(deficit):
    #                 emb_set.append(torch.zeros((1,768)))
    #     emb_set = torch.cat(emb_set, dim=0)
    #     # emb_set= torch.flatten(emb_set,0).unsqueeze(0)
    #     return emb_set