# coding=utf-8
# author=yphacker
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from conf import config
from conf import model_config_bert
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 128), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    trans=transforms.Compose(transform_train_list)
    img=trans(img)
    res_img = img.float()
    return res_img

class MyDataset(Dataset):

    def __init__(self, df, mode='train', task='1'):
        self.mode = mode
        self.task = task
        self.tokenizer = BertTokenizer.from_pretrained(model_config_bert.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.x_data = []
        self.img_data=[]
        self.y_data = []
        for i, row in df.iterrows():
            x,img_path, y,flag = self.row_to_tensor(self.tokenizer, row)
            if flag==True:
                self.x_data.append(x)
                self.img_data.append(read_img(img_path))
                self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.img_data[index],self.y_data[index]

    def contact(self, str1, str2):
        if pd.isnull(str2):
            return str1
        return str1 + str2

    def row_to_tensor(self, tokenizer, row):
        flag=True
        if self.task == '0':
            text = row['content']
        elif self.task == '1':
            text = self.contact(row['content'], row['comment_2c'])
        else:
            text = self.contact(row['content'], row['comment_all'])
        x_encode = tokenizer.encode(text)
        if len(x_encode) > config.max_seq_len[self.task]:
            text_len = int(config.max_seq_len[self.task] / 2)
            x_encode = x_encode[:text_len] + x_encode[-text_len:]
        else:
            padding = [0] * (config.max_seq_len[self.task] - len(x_encode))
            x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        img_name=row['img']
        img_path=''
        path = 'train_images/train_pictures/' + img_name
        if os.path.exists(path):
            img_path=path
        else:
            flag = False
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * len(config.label_columns), dtype=torch.long)
        else:
            # y_data = row[config.label_columns]
            y_data = row['label']
            y_tensor = torch.tensor(y_data, dtype=torch.long)
        return x_tensor, img_path, y_tensor,flag

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
