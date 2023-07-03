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
from crd.criterion import CRDLoss, CRDsoftmax
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
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=30, type=int, help="epochs num")
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    # parser.add_argument('--distill', type=str, default='nst', choices=['kd', 'nst'])
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.3, help='weight balance for other losses')

    parser.add_argument('--nce_k', default=1500, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--crd_op', default=0, type=int, help='choice of crd or crd_softmax')

    args = parser.parse_args()
    # args.save_folder=os.path.join(config.save_folder,'crd_op_0')
    return args


#

def main():
    best_img_acc = 0
    best_text_acc = 0

    args = parse_option()
    sys.stdout = Logger(os.path.join(config.save_folder, 'crd_mutual_baseline.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset = MMRumor(root='data')

    train_dataset = RumorDataset(dataset.train, 'train', args.nce_k)
    args.n_data = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = RumorDataset(dataset.test, 'val', args.nce_k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    model_text = Model()
    model_img = vgg19(num_classes=2)

    module_list = nn.ModuleList([])
    module_list.append(model_img)

    criterion_cls = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(args.kd_T)
    criterion_crd = CRDLoss(args)
    # crd_softmax=CRDsoftmax(args)

    module_list.append(criterion_crd.embed_s)
    module_list.append(criterion_crd.embed_t)
    # module_list.append(crd_softmax.embed_s)
    # module_list.append(crd_softmax.embed_t)
    module_list.append(model_text)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    # criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_crd)
    # criterion_list.append(crd_softmax)

    optimizers = []

    optimizer_img = optim.SGD(model_img.parameters(), lr=0.0003, momentum=0.9, weight_decay=5e-4)

    optimizer_text = torch.optim.Adam(model_text.parameters(), lr=2e-5)

    optimizers.append(optimizer_img)
    optimizers.append(optimizer_text)

    # schedulers=[]
    # scheduler_img = optim.lr_scheduler.ExponentialLR(optimizer_img, gamma=0.1)
    # scheduler_text = optim.lr_scheduler.ExponentialLR(optimizer_text, gamma=0.1)
    # schedulers.append(scheduler_img)
    # schedulers.append(scheduler_text)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    time1 = time.time()
    for epoch in range(1, args.epochs_num + 1):

        print("==> training...")


        train_crd_mutual(epoch, train_loader, module_list, criterion_list, optimizers, args)

        # print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_img_acc, test_img_loss = validate_crd(val_loader, module_list, criterion_cls, 0)
        test_text_acc, test_text_loss = validate_crd(val_loader, module_list, criterion_cls, 1)

        # save the best model
        if test_img_acc > best_img_acc:
            best_img_acc = test_img_acc
            state = {
                'epoch': epoch,
                'model': model_img.state_dict(),
                'best_acc': best_img_acc,
            }
            save_file = os.path.join(config.save_folder, 'img_best_crd_mutual.pth'.format(args.crd_op))
            print('saving the best model for img!')
            torch.save(state, save_file)
        if test_text_acc > best_text_acc:
            best_text_acc = test_text_acc
            state = {
                'epoch': epoch,
                'model': model_text.state_dict(),
                'best_acc': best_text_acc,
            }
            save_file = os.path.join(config.save_folder, 'text_best_crd_mutual.pth'.format(args.crd_op))
            print('saving the best model for text!')
            torch.save(state, save_file)
    time2 = time.time()
    print('Total time:', time2-time1)
    print('best accuracy for image model :', best_img_acc)
    print('best accuracy for text model :', best_text_acc)





def train_crd_mutual(epoch, train_loader, module_list, criterion_list, optimizers, opt):
    for module in module_list:
        module.train()

    criterion_cls = criterion_list[0]
    # criterion_div = criterion_list[1]
    criterion_kd = criterion_list[1]
    # crdsoftmax=criterion_list[2]

    model_img = module_list[0]
    model_text = module_list[-1]

    losses_img = AverageMeter()
    losses_text = AverageMeter()
    top1_img = AverageMeter()
    top1_text = AverageMeter()

    for idx, data in enumerate(train_loader):
        batch_x, batch_img, batch_y, index, contrast_idx = data
        batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
        index, contrast_idx = index.cuda(), contrast_idx.cuda()

        feat_s, logits_img = model_img(batch_img)
        print('size of img feat_s:', feat_s.size())
        cls_output, logits_text = model_text(batch_x)

        loss_cls_img = criterion_cls(logits_img, batch_y)
        loss_cls_text = criterion_cls(logits_text, batch_y)

        # loss_div_img = criterion_div(logits_img, logits_text)
        # loss_div_text = criterion_div(logits_text, logits_img)

        if opt.crd_op == 0:
            loss_kd = criterion_kd(feat_s, cls_output, index, contrast_idx)
        else:

            loss_kd = crdsoftmax(feat_s, cls_output, index, contrast_idx)

        # train_loss_img = opt.gamma * loss_cls_img + opt.alpha * loss_div_img+ opt.beta * loss_kd
        # train_loss_text = opt.gamma * loss_cls_text + opt.alpha * loss_div_text + opt.beta * loss_kd
        train_loss_img = opt.gamma * loss_cls_img + opt.beta * loss_kd
        train_loss_text = opt.gamma * loss_cls_text + opt.beta * loss_kd

        acc1_img = accuracy(logits_img, batch_y)
        losses_img.update(train_loss_img.item(), batch_img.size(0))
        top1_img.update(float(acc1_img[0]), batch_img.size(0))

        optimizers[0].zero_grad()
        train_loss_img.backward(retain_graph=True)
        optimizers[0].step()

        acc1_text = accuracy(logits_text, batch_y)
        losses_text.update(train_loss_text.item(), batch_x.size(0))
        top1_text.update(float(acc1_text[0]), batch_x.size(0))

        optimizers[1].zero_grad()
        train_loss_text.backward()
        optimizers[1].step()

        if idx % config.train_print_step == 0:
            print('Image Model:')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses_img, top1=top1_img))
            # print('img_cls_loss：{}'.format(loss_cls_img))
            # print('img_dv_loss：{}'.format(loss_div))

            print('Text Model:')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses_text, top1=top1_text))
            # print('text_cls_loss：{}'.format(loss_cls_text))
            # print('text_dv_loss：{}'.format(loss_div))
    print(' Train image model : Acc@1 {top1.avg:.3f}'.format(top1=top1_img))
    print(' Train text model : Acc@1 {top1.avg:.3f}'.format(top1=top1_text))
    # sys.stdout.flush()


def validate_crd(val_loader, module_list, criterion, opt):
    """validation"""
    if opt == 0:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        for model in module_list:
            model.eval()
        model_img = module_list[0]

        with torch.no_grad():
            cur_step = 0
            for _, batch_img, batch_y, _, _ in val_loader:
                batch_img, batch_y = batch_img.cuda(), batch_y.cuda()

                batch_img = batch_img.float()

                # compute output
                _, logits = model_img(batch_img)
                loss = criterion(logits, batch_y)

                # measure accuracy and record loss
                acc1 = accuracy(logits, batch_y)
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

            print('val img model * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        return top1.avg, losses.avg

    elif opt == 1:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        # model.eval()
        # model_text=model[-1]
        for model in module_list:
            model.eval()
        model_text = module_list[-1]

        with torch.no_grad():
            cur_step = 0
            for batch_x, _, batch_y, _, _ in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                # compute output
                cls_output, logits = model_text(batch_x)
                loss = criterion(logits, batch_y)

                # measure accuracy and record loss
                acc1 = accuracy(logits, batch_y)
                losses.update(loss.item(), batch_x.size(0))
                top1.update(float(acc1[0]), batch_x.size(0))

                cur_step += 1
                if cur_step % config.train_print_step == 0:
                    print('test for text model: ')
                    print('test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        cur_step, len(val_loader), loss=losses,
                        top1=top1))

            print('val text model * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        return top1.avg, losses.avg


if __name__ == '__main__':
    main()

