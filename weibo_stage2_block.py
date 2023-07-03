# coding=utf-8


import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from conf import model_config_bert as model_config
import matplotlib.pyplot as plt
from helper import *
import numpy as np
from crd.criterion import CRDLoss
from torch.utils.data import Dataset
from transformers import BertTokenizer
from model.vgg import vgg19
from model.bert import Model
from model.block import Block
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
    def __init__(self, dataset, mode='train', k=1586):
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
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=30, type=int, help="epochs num")
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument("-img_path", type=str,
                        default='data/log/roberta/vgg/test_for_k/1500/img_best_crd_mutual.pth',
                        metavar='PATH')
    parser.add_argument("-text_path", type=str,
                        default='data/log/roberta/vgg/test_for_k/1500/text_best_crd_mutual.pth',
                        metavar='PATH')


    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.8, help='weight balance for other losses')

    parser.add_argument('--nce_k', default=250, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--crd_op', default=0, type=int, help='choice of crd or crd_softmax')

    args = parser.parse_args()
    args.save_folder = os.path.join(config.save_folder, 'baseline')
    return args


def main():
    best_acc = 0
    best_prec_0, best_rec_0, best_f_0 = 0, 0, 0
    best_prec_1, best_rec_1, best_f_1 = 0, 0, 0
    full_metric = None

    args = parse_option()
    sys.stdout = Logger(os.path.join(config.save_folder, 'block_fc.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset = MMRumor(root='data')

    train_dataset = RumorDataset(dataset.train, 'train', args.nce_k)
    args.n_data = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = RumorDataset(dataset.test, 'val', args.nce_k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model_text = Model().eval()
    model_img = vgg19(num_classes=2).eval()

    if args.img_path:
        print("=> loading img model:")
        checkpoint = torch.load(args.img_path)
        model_img.load_state_dict(checkpoint['model'])

    if args.text_path:
        print("=> loading text model:")
        checkpoint = torch.load(args.text_path)
        model_text.load_state_dict(checkpoint['model'])

    block = Block([25088, 768], 2)

    # fc = LinearEmbed(128, 2)

    module_list = torch.nn.ModuleList([])
    module_list.append(model_img)
    module_list.append(model_text)
    module_list.append(block)
    # module_list.append(fc)

    criterion_cls = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(module_list[-1:].parameters(), lr=2e-5)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_cls.cuda()
        cudnn.benchmark = True

    time1 = time.time()
    for epoch in range(1, args.epochs_num + 1):

        print("==> training...")


        train(epoch, train_loader, module_list, criterion_cls, optimizer, args)



        acc, prec_0, rec_0, f_0, prec_1, rec_1, f_1  = validate_2_stage(val_loader, module_list, criterion_cls)

        if acc > best_acc:
            best_acc = acc
            full_metric = (acc, prec_0, rec_0, f_0, prec_1, rec_1, f_1)
            print('best acc in epoch:{}'.format(epoch), best_acc)
            print('full metric when best acc: prec_0, rec_0, f_0, prec_1, rec_1, f_1')
            print(full_metric)
            state = {
                'epoch': epoch,
                'model': block.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(config.save_folder, 'block_crd_mutual.pth'.format(args.crd_op))
            print('saving the best model for block!')
            torch.save(state, save_file)
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

    time2 = time.time()
    print('Total time:', time2 - time1)
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
    #     if acc > best_acc:
    #         best_acc = acc
    #         state = {
    #             'epoch': epoch,
    #             'model': block.state_dict(),
    #             'best_acc': best_acc,
    #         }
    #         save_file = os.path.join(config.save_folder, 'block_fc.pth')
    #         print('saving  for fc!')
    #         torch.save(state, save_file)
    #
    # print('best accuracy :', best_acc)


def train(epoch, train_loader, module_list, criterion_cls, optimizer, opt):
    model_img = module_list[0]
    model_text = module_list[1]
    block = module_list[2]
    model_img.eval()
    model_text.eval()
    # fc = module_list[-1]
    block.train()
    # fc.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, data in enumerate(train_loader):
        batch_x, batch_img, batch_y, _, _ = data
        batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()

        with torch.no_grad():
            feat_s, logits_img = model_img(batch_img)
            feat_s = feat_s.detach()
            cls_output, logits_text = model_text(batch_x)
            cls_output = cls_output.detach()

        input1 = feat_s.view(batch_x.size(0), -1)
        input2 = cls_output
        input = [input1, input2]
        final = block(input)
        # print('final.size',final.size())
        # logits=fc(final)

        loss = criterion_cls(final, batch_y)
        acc = accuracy(final, batch_y)

        losses.update(loss.item(), batch_y.size(0))
        top1.update(float(acc[0]), batch_y.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % config.train_print_step == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses, top1=top1))

    print(' Train : Acc@1 {top1.avg:.3f}'.format(top1=top1))
    sys.stdout.flush()


def validate_2_stage(val_loader, module_list, criterion):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    count_0_lists = []
    correct_0_lists = []
    target_0_lists = []
    count_1_lists = []
    correct_1_lists = []
    target_1_lists = []

    # switch to evaluate mode
    for model in module_list:
        model.eval()
    model_img = module_list[0]
    model_text = module_list[1]
    block = module_list[2]
    # fc=module_list[-1]

    with torch.no_grad():
        cur_step = 0
        for idx, data in enumerate(val_loader):
            batch_x, batch_img, batch_y, _, _ = data
            batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
            batch_img = batch_img.float()

            feat_s, logits_img = model_img(batch_img)
            cls_output, logits_text = model_text(batch_x)
            input1 = feat_s.view(batch_x.size(0), -1)
            input2 = cls_output
            input = [input1, input2]
            final = block(input)
            # print('final.size',final.size())
            # logits=fc(final)
            # f = torch.cat((feat_s[-1], cls_output), 1)
            # logits = fc(f)

            loss = criterion(final, batch_y)

            acc = accuracy(final, batch_y)

            count_0, count_correct_0, count_target_0 = metric(final, batch_y, for_fake=True)
            count_1, count_correct_1, count_target_1 = metric(final, batch_y, for_fake=False)
            count_0_lists.append(count_0)
            correct_0_lists.append(count_correct_0)
            target_0_lists.append(count_target_0)

            count_1_lists.append(count_1)
            correct_1_lists.append(count_correct_1)
            target_1_lists.append(count_target_1)

            losses.update(loss.item(), batch_y.size(0))
            top1.update(float(acc[0]), batch_y.size(0))

            if idx % config.train_print_step == 0:
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

    # print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
    #
    # return top1.avg, losses.avg


if __name__ == '__main__':
    main()

#
# def predict():
#     comment_dict = {
#         0: '0c',
#         1: '2c',
#         2: 'all'
#     }
#     fake_prob_label = defaultdict(list)
#     real_prob_label = defaultdict(list)
#     ncw_prob_label = defaultdict(list)
#     test_df = pd.read_csv(config.test_path)
#     test_df.fillna({'content': ''}, inplace=True)
#
#     for i in range(3):
#         test_dataset = MyDataset(test_df, 'test', '{}'.format(i))
#         test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
#         model = resnet50(num_classes=3).to(device)
#         resume = os.path.join(config.model_path, 'img_{}_task{}_trial{}.bin'.format(model_name, i, args.trial))
#         model.load_state_dict(torch.load(resume))
#
#         model.eval()
#         with torch.no_grad():
#             for batch_x, batch_img,batch_y,_,_ in tqdm(test_loader):
#                 batch_x = batch_x.cuda
#                 logits, _ = model(batch_x)
#                 # _, preds = torch.max(probs, 1)
#                 probs = torch.softmax(logits, 1)
#                 # probs_data = probs.cpu().data.numpy()
#                 fake_prob_label[i] += [p[0].item() for p in probs]
#                 real_prob_label[i] += [p[1].item() for p in probs]
#                 ncw_prob_label[i] += [p[2].item() for p in probs]
#     submission = pd.read_csv(config.sample_submission_path)
#     for i in range(3):
#         submission['fake_prob_label_{}'.format(comment_dict[i])] = fake_prob_label[i]
#         submission['real_prob_label_{}'.format(comment_dict[i])] = real_prob_label[i]
#         submission['ncw_prob_label_{}'.format(comment_dict[i])] = ncw_prob_label[i]
#     submission.to_csv(config.submission_path + '/' + 'submission_text.csv', index=False)

