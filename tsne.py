# coding=utf-8

import matplotlib

matplotlib.use('Agg')
import gc
import sys
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from conf import model_config_bert as model_config
import matplotlib.pyplot as plt
from helper import *
from model.block import Block
from torch.utils.data import Dataset
from transformers import BertTokenizer
from model.vgg import vgg19
from model.bert import Model
from conf import config
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import matplotlib
# matplotlib.use('Agg')
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
# import tsne
# from tsne import bh_sne
from MulticoreTSNE import MulticoreTSNE as TSNE


def plot_embedding(data, label):
    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
    type1 = plt.scatter(type1_x, type1_y, s=10, c='b')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='m')
    # plt.legend((type1, type2),
    #            ('fake',  'real'),
    #            loc=(0.97, 0.5))
    plt.legend((type1, type2),
               ('fake',  'real'),
               )
    plt.xticks()
    plt.yticks()
    ax.spines['right'].set_visible(False)  # 去除右边框
    ax.spines['top'].set_visible(False)  # 去除上边框
    return fig



class MMRumor(object):

    def __init__(self, root='data', **kwargs):
        self.train_dir1 = os.path.join(root, 'train_rumor.txt')
        self.train_dir2 = os.path.join(root, 'train_nonrumor.txt')
        self.test_dir1 = os.path.join(root, 'test_rumor.txt')
        self.test_dir2 = os.path.join(root, 'test_nonrumor.txt')

        train_rumor = self.process_data(self.train_dir1, 0)
        train_nonrumor = self.process_data(self.train_dir2, 1)
        test_rumor = self.process_data(self.test_dir1, 0)
        test_nonrumor = self.process_data(self.test_dir2, 1)

        print("=> MMRumor loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # rumor | # nonrumor")
        print("  ------------------------------")
        print("  train    | {:5d}   | {:8d}".format(len(train_rumor), len(train_nonrumor)))
        print("  test     | {:5d}   | {:8d}".format(len(test_rumor), len(test_nonrumor)))

        print("  ------------------------------")
        print("  total    | {:5d}   | {:8d}".format(len(train_rumor) + len(test_rumor),
                                                    len(train_nonrumor) + len(test_nonrumor)))
        print("  ------------------------------")

        self.train = train_rumor + train_nonrumor
        self.test = test_rumor + test_nonrumor

    def process_data(self, file_name, label):
        with open('{}'.format(file_name), encoding="utf-8") as f:
            lines = f.readlines()
            count = 0
            res, tmp, datas, dataset = [], [], [], []
            for line in lines:
                res.append(line)
            for item in res:
                if count < 3:
                    tmp += [item]
                    count += 1
                if count == 3:
                    datas.append(tmp)
                    count = 0
                    tmp = []
            for data in datas:
                img_name = data[1].split('|')[0].split('/')[-1]
                content = data[2].split('\n')[0]
                if label == 0:
                    if os.path.exists('weibo_img/rumor_images/' + img_name):
                        dataset.append((img_name, content, 0))
                if label == 1:
                    if os.path.exists('weibo_img/nonrumor_images/' + img_name):
                        dataset.append((img_name, content, 1))
            return dataset


class RumorDataset(Dataset):
    def __init__(self, dataset, mode='train', k=16384):
        self.dataset = dataset
        self.mode = mode
        self.k = k
        self.tokenizer = BertTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.x_data = []
        self.img_data = []
        self.y_data = []
        self.cls_positive = [[] for _ in range(2)]
        self.cls_negative = [[] for _ in range(2)]

        for idx, data in enumerate(dataset):
            img_name, content, label = data
            x = self.row_to_tensor(self.tokenizer, content)
            self.x_data.append(x)
            self.img_data.append(self.read_img(img_name, label))
            self.y_data.append(label)
            self.cls_positive[label].append(idx)
        for i in range(2):
            for j in range(2):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(2)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(2)]
        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def row_to_tensor(self, tokenizer, content):
        x_encode = tokenizer.encode(content)
        if len(x_encode) > config.max_seq_len:
            text_len = int(config.max_seq_len / 2)
            x_encode = x_encode[:text_len] + x_encode[-text_len:]
        else:
            padding = [0] * (config.max_seq_len - len(x_encode))
            x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        return x_tensor

    def read_img(self, img_name, label):
        if label == 0:
            img_path = 'weibo_img/rumor_images/' + img_name
        else:
            img_path = 'weibo_img/nonrumor_images/' + img_name
        img = Image.open(img_path).convert('RGB')
        transform_train_list = [
            # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_test_list = [
            transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        tran_trains = transforms.Compose(transform_train_list)
        tran_tests = transforms.Compose(transform_test_list)
        if self.mode == 'train':
            img = tran_trains(img)
        else:
            img = tran_tests(img)
        res_img = img.float()
        return res_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pos_idx = index
        replace = True if self.k > len(self.cls_negative[self.y_data[index]]) else False
        neg_idx = np.random.choice(self.cls_negative[self.y_data[index]], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return self.x_data[index], self.img_data[index], self.y_data[index], index, sample_idx


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=60, type=int, help="epochs num")
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--distill', type=str, default='nst', choices=['kd', 'nst'])
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.8, help='weight balance for other losses')

    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--crd_op', default=0, type=int, help='choice of crd or crd_softmax')

    # parser.add_argument("-img_path", type=str,
    #                     default='data/log/roberta/vgg/img_best_crd_mutual.pth',
    #                     metavar='PATH')
    parser.add_argument("-text_path", type=str,
                        default='data/log/roberta/vgg/test_for_k/50/text_best_crd_mutual.pth',
                        metavar='PATH')
    # parser.add_argument("-fc_path", type=str,
    #                     default='data/log/roberta/vgg/fc_crd_mutual.pth',
    #                     metavar='PATH')
    # parser.add_argument("-block_path", type=str,
    #                     default='data/log/roberta/vgg/block_crd_mutual.pth',
    #                     metavar='PATH')

    args = parser.parse_args()
    # args.save_folder=os.path.join(config.save_folder,'baseline')
    return args


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


# make the distribution fit [0; 1] by dividing by its range


def main():
    best_acc = 0

    args = parse_option()

    dataset = MMRumor(root='data')

    train_dataset = RumorDataset(dataset.train, 'train', args.nce_k)
    args.n_data = len(train_dataset)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = RumorDataset(dataset.test, 'val', args.nce_k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model_text = Model()
    model_img = vgg19(num_classes=2)
    fc = LinearEmbed(25856, 2)
    block = Block([25088, 768], 2)

    # if args.img_path:
    #     print("=> loading img model:")
    #     checkpoint_1 = torch.load(args.img_path)
    #     model_img.load_state_dict(checkpoint_1['model'])

    if args.text_path:
        print("=> loading text model:")
        checkpoint_2 = torch.load(args.text_path)
        model_text.load_state_dict(checkpoint_2['model'])
    #
    # if args.fc_path:
    #     print("=> loading fc model:")
    #     checkpoint_3 = torch.load(args.fc_path)
    #     fc.load_state_dict(checkpoint_3['model'])

    # if args.block_path:
    #     print("=> loading block model:")
    #     checkpoint_4 = torch.load(args.block_path)
    #     block.load_state_dict(checkpoint_4['model'])

    module_list = torch.nn.ModuleList([])
    module_list.append(model_img)
    module_list.append(model_text)
    module_list.append(fc)
    module_list.append(block)
    for model in module_list:
        model.eval()

    if torch.cuda.is_available():
        module_list.cuda()
        cudnn.benchmark = True

    target = []
    feature_img_set, feature_text_set, feature_fc_set, feature_block_set = [], [], [], []
    for idx, data in enumerate(val_loader):
        batch_x, batch_img, batch_y, _, _ = data
        batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
        with torch.no_grad():
            feat_img, logits_img = model_img(batch_img)
            feat_img = feat_img.detach()
            feat_text, logits_text = model_text(batch_x)
            feat_text = feat_text.detach()
            feat_fc = torch.cat((feat_img, feat_text), 1)
            feat_block = block([feat_img,feat_text])

            feature_img_set.append(feat_img.cpu().numpy())
            feature_text_set.append(feat_text.cpu().numpy())
            feature_fc_set.append(feat_fc.cpu().numpy())
            feature_block_set.append(feat_block.cpu().numpy())
            target_np = batch_y.cpu().numpy()
            target.append(target_np)

    array_img = np.concatenate(feature_img_set, axis=0)
    array_text = np.concatenate(feature_text_set, axis=0)
    array_fc = np.concatenate(feature_fc_set, axis=0)
    array_block = np.concatenate(feature_block_set, axis=0)
    array_target = np.concatenate(target, axis=0)
    # print(array_img.shape)
    # print(array_target.shape)

    # embeddings = TSNE(n_jobs=4).fit_transform(array_img)
    embeddings = TSNE(n_jobs=4).fit_transform(array_text)
    fig = plot_embedding(embeddings, array_target)
    # fig.subplots_adjust(right=0.8)
    plt.savefig('text3.png', bbox_inches='tight')


if __name__ == '__main__':
    main()


