# coding=utf-8
#
#
import gc
import sys
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from conf import model_config_bert as model_config
import os
import pandas as pd
import torch
import numpy as np
from helper import *

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

        for i, data in enumerate(dataset):
            img_name, content, label = data
            x = self.row_to_tensor(self.tokenizer, content)
            self.x_data.append(x)
            self.img_data.append(self.read_img(img_name, label))
            self.y_data.append(label)
            self.cls_positive[label].append(i)
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
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=30, type=int, help="epochs num")
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--distill', type=str, default='nst', choices=['kd', 'nst'])
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.3, help='weight balance for other losses')

    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--crd_op', default=0, type=int, help='choice of crd or crd_softmax')

    args = parser.parse_args()
    # args.save_folder=config.save_folder
    return args


#

def main():
    best_acc = 0
    best_text_acc = 0
    best_prec_0, best_rec_0, best_f_0 = 0, 0, 0
    best_prec_1, best_rec_1, best_f_1 = 0, 0, 0
    full_metric = None

    args = parse_option()
    sys.stdout = Logger(os.path.join(config.save_folder, 'vgg.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset = MMRumor(root='data')

    train_dataset = RumorDataset(dataset.train, 'train', args.nce_k)
    args.n_data = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = RumorDataset(dataset.test, 'val', args.nce_k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)


    model = vgg19(num_classes=2)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9, weight_decay=5e-4)



    scheduler_img = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)


    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        cudnn.benchmark = True

    # log_writer = LogWriter(config.save_folder, sync_cycle=10)
    # with log_writer.mode("train") as logger:
    #     scalar_train_img_acc = logger.scalar("img_acc")
    #     scalar_train_text_acc = logger.scalar("text_acc")
    #     # scalar_train_loss = logger.scalar("loss")
    # with log_writer.mode("test") as logger:
    #     # scalar_test_acc = logger.scalar("acc")
    #     # scalar_test_loss = logger.scalar("loss")
    #     scalar_test_img_acc = logger.scalar("img_acc")
    #     scalar_test_text_acc = logger.scalar("text_acc")

    for epoch in range(1, args.epochs_num + 1):

        print("==> training...")

        time1 = time.time()
        train(epoch, train_loader, model, criterion, optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        acc, prec_0, rec_0, f_0, prec_1, rec_1, f_1 = validate(val_loader, model, criterion)

        if acc > best_acc:
            best_acc = acc
            full_metric = (acc, prec_0, rec_0, f_0, prec_1, rec_1, f_1)
            print('best acc in epoch:{}'.format(epoch), best_acc)
            print('full metric when best acc: prec_0, rec_0, f_0, prec_1, rec_1, f_1')
            print(full_metric)
            print('\n')
        if prec_0 > best_prec_0:
            best_prec_0 = prec_0
            print('best prec_0 in epoch {}:'.format(epoch), best_prec_0)
        if rec_0 > best_rec_0:
            best_rec_0 = rec_0
            print('best rec_0 in epoch {}:'.format(epoch), best_rec_0)
        if f_0 > best_f_0:
            best_f_0 = f_0
            print('best f_0 in epoch {}:'.format(epoch), best_f_0)

        if prec_1 > best_prec_1:
            best_prec_1 = prec_1
            print('best prec_1 in epoch {}:'.format(epoch), best_prec_1)
        if rec_1 > best_rec_1:
            best_rec_1 = rec_1
            print('best rec_1 in epoch {}:'.format(epoch), best_rec_1)
        if f_1 > best_f_1:
            best_f_1 = f_1
            print('best f_1 in epoch {}:'.format(epoch), best_f_1)

    print('best accuracy :', best_acc)
    print('full metric under best accuracy: prec, rec, F1, prec_fake, rec_fake, F1_fake')
    print(full_metric)

    print('best prec_0 :', best_prec_0)
    print('best rec_0 :', best_rec_0)
    print('best f_0 :', best_f_0)
    print('best prec_1 :', best_prec_1)
    print('best rec_1 :', best_rec_1)
    print('best f_1 :', best_f_1)

    # save the best model
    #     if test_img_acc > best_img_acc:
    #         best_img_acc = test_img_acc
    #         state = {
    #             'epoch': epoch,
    #             'model': model.state_dict(),
    #             'best_acc': best_img_acc,
    #         }
    #         save_file = os.path.join(config.save_folder, 'img_best_baseline.pth')
    #         print('saving the best model for img!')
    #         torch.save(state, save_file)
    #
    # print('best accuracy for image model :', best_img_acc)


def train(epoch, train_loader, model, criterion, optimizer):

    model.train()


    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, data in enumerate(train_loader):
        _, batch_img, batch_y, _, _= data
        batch_img, batch_y = batch_img.cuda(), batch_y.cuda()

        # added
        # print('index: ',index)
        # print('contrast_idx',contrast_idx)

        feat_s, logits_img = model(batch_img)

        loss= criterion(logits_img, batch_y)


        acc1_img = accuracy(logits_img, batch_y)
        losses.update(loss.item(), batch_img.size(0))
        top1.update(float(acc1_img[0]), batch_img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if idx % config.train_print_step == 0:
            print('Image Model:')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses, top1=top1))

    print(' Train image model : Acc@1 {top1.avg:.3f}'.format(top1=top1))


    # return top1, top1_text.avg


def validate(val_loader, model, criterion):
    """validation"""

    losses = AverageMeter()
    top1 = AverageMeter()
    count_0_lists = []
    correct_0_lists = []
    target_0_lists = []
    count_1_lists = []
    correct_1_lists = []
    target_1_lists = []

    model.eval()

    with torch.no_grad():
        cur_step = 0
        for _, batch_img, batch_y, _, _ in val_loader:
            batch_img, batch_y = batch_img.cuda(), batch_y.cuda()

            batch_img = batch_img.float()

            # compute output
            _,logits = model(batch_img)
            loss = criterion(logits, batch_y)

            # measure accuracy and record loss
            acc1 = accuracy(logits, batch_y)

            count_0, count_correct_0, count_target_0 = metric(logits, batch_y, for_fake=True)
            count_1, count_correct_1, count_target_1 = metric(logits, batch_y, for_fake=False)
            count_0_lists.append(count_0)
            correct_0_lists.append(count_correct_0)
            target_0_lists.append(count_target_0)

            count_1_lists.append(count_1)
            correct_1_lists.append(count_correct_1)
            target_1_lists.append(count_target_1)

            losses.update(loss.item(), batch_img.size(0))
            top1.update(float(acc1[0]), batch_img.size(0))

            cur_step += 1
            if cur_step % config.train_print_step == 0:
                print('test for image model: ')
                print('test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    cur_step, len(val_loader), loss=losses,
                    top1=top1))

        epoch_count_0 = sum(count_0_lists)
        epoch_correct_0 = sum(correct_0_lists)
        epoch_target_0 = sum(target_0_lists)

        epoch_count_1 = sum(count_1_lists)
        epoch_correct_1 = sum(correct_1_lists)
        epoch_target_1 = sum(target_1_lists)

        try:
            prec_0 = float(epoch_correct_0) * (100.0 / float(epoch_count_0))
        except ZeroDivisionError:
            prec_0 = 0
        try:
            rec_0 = float(epoch_correct_0) * (100.0 / float(epoch_target_0))
        except ZeroDivisionError:
            rec_0 = 0
        try:
            f_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0)
        except ZeroDivisionError:
            f_0 = 0

        try:
            prec_1 = float(epoch_correct_1) * (100.0 / float(epoch_count_1))
        except ZeroDivisionError:
            prec_1 = 0
        try:
            rec_1 = float(epoch_correct_1) * (100.0 / float(epoch_target_1))
        except ZeroDivisionError:
            rec_1 = 0
        try:
            f_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1)
        except ZeroDivisionError:
            f_1 = 0

        # prec_1 = float(epoch_correct_1) * (100.0 / float(epoch_count_1))
        # rec_1 = float(epoch_correct_1) * (100.0 / float(epoch_target_1))
        # f_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1)
        print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
        print('metrics: prec_0, rec_0, f_0, prec_1, rec_1, f_1')
        print(prec_0, rec_0, f_0, prec_1, rec_1, f_1)

    return top1.avg, prec_0, rec_0, f_0, prec_1, rec_1, f_1

    # print('val img model * Acc@1 {top1.avg:.3f} '.format(top1=top1))
    #
    #     return top1.avg, losses.avg


if __name__ == '__main__':
    main()

