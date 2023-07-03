# ######################################################################################################3
# # import sys
# # import time
# # import torch
# # import os
# # import errno
# # import torch.nn as nn
# # from conf import config
# # from model.resnet import resnet50
# # import numpy as np
# #
# # def adjust_learning_rate(epoch, opt, optimizer):
# #     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
# #     steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
# #     if steps > 0:
# #         new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
# #         for param_group in optimizer.param_groups:
# #             param_group['lr'] = new_lr
# #
# # def mkdir_if_missing(directory):
# #     if not os.path.exists(directory):
# #         try:
# #             os.makedirs(directory)
# #         except OSError as e:
# #             if e.errno != errno.EEXIST:
# #                 raise
# #
# # class Logger(object):
# #     def __init__(self, fpath=None):
# #         self.console = sys.stdout
# #         self.file = None
# #         if fpath is not None:
# #             mkdir_if_missing(os.path.dirname(fpath))
# #             self.file = open(fpath, 'w')
# #
# #     def __del__(self):
# #         self.close()
# #
# #     def __enter__(self):
# #         pass
# #
# #     def __exit__(self, *args):
# #         self.close()
# #
# #     def write(self, msg):
# #         self.console.write(msg)
# #         if self.file is not None:
# #             self.file.write(msg)
# #
# #     def flush(self):
# #         self.console.flush()
# #         if self.file is not None:
# #             self.file.flush()
# #             os.fsync(self.file.fileno())
# #
# #     def close(self):
# #         self.console.close()
# #         if self.file is not None:
# #             self.file.close()
# #
# # # class LinearEmbed(nn.Module):
# # #     """Linear Embedding"""
# # #     def __init__(self, dim_in=1024, dim_out=128):
# # #         super(LinearEmbed, self).__init__()
# # #         self.linear = nn.Linear(dim_in, dim_out)
# # #
# # #     def forward(self, x):
# # #         x = x.view(x.shape[0], -1)
# # #         x = self.linear(x)
# # #         return x
# #
# #
# # class LinearEmbed(nn.Module):
# #     """Embedding module"""
# #     def __init__(self, dim_in=128, dim_out=3):
# #         super(LinearEmbed, self).__init__()
# #         self.linear = nn.Linear(dim_in, dim_out)
# #         # self.l2norm = Normalize(2)
# #
# #     def forward(self, x):
# #         x = x.view(x.shape[0], -1)
# #         x = self.linear(x)
# #         # x = self.l2norm(x)
# #         return x
# #
# #
# # class Normalize(nn.Module):
# #     """normalization layer"""
# #     def __init__(self, power=2):
# #         super(Normalize, self).__init__()
# #         self.power = power
# #
# #     def forward(self, x):
# #         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
# #         out = x.div(norm)
# #         return out
# #
# #
# #
# # class AverageMeter(object):
# #     """Computes and stores the average and current value"""
# #     def __init__(self):
# #         self.reset()
# #
# #     def reset(self):
# #         self.val = 0
# #         self.avg = 0
# #         self.sum = 0
# #         self.count = 0
# #
# #     def update(self, val, n=1):
# #         self.val = val
# #         self.sum += val * n
# #         self.count += n
# #         self.avg = self.sum / self.count
# #
# #
# def accuracy(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = topk
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#
#         correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
# def precision(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = topk
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#
#         correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
# def recall(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = topk
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#
#         correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
# def f1(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = topk
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#
#         correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res
# #
# #
# # def validate_text(val_loader, model, criterion):
# #     """validation"""
# #     losses = AverageMeter()
# #     top1 = AverageMeter()
# #
# #     # switch to evaluate mode
# #     model.eval()
# #
# #     with torch.no_grad():
# #         cur_step = 0
# #         for batch_x, batch_y in val_loader:
# #             batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
# #
# #             # input = input.float()
# #             # if torch.cuda.is_available():
# #             #     input = input.cuda()
# #             #     target = target.cuda()
# #
# #             # compute output
# #             cls_output, logits = model(batch_x)
# #             loss = criterion(logits, batch_y)
# #
# #             # measure accuracy and record loss
# #             acc1 = accuracy(logits, batch_y, topk=1)
# #             losses.update(loss.item(), batch_x.size(0))
# #             top1.update(float(acc1[0]), batch_x.size(0))
# #
# #
# #             # measure elapsed time
# #
# #             # end = time.time()
# #             cur_step += 1
# #             if cur_step % config.train_print_step == 0:
# #                 print('test: [{0}/{1}]\t'
# #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                        cur_step, len(val_loader),  loss=losses,
# #                        top1=top1))
# #
# #         print('val * Acc@1 {top1.avg:.3f} '
# #               .format(top1=top1))
# #
# #     return top1.avg,  losses.avg
# #
# #
# # def validate_img(val_loader, model, criterion):
# #     """validation"""
# #     losses = AverageMeter()
# #     top1 = AverageMeter()
# #
# #     # switch to evaluate mode
# #     model.eval()
# #
# #     with torch.no_grad():
# #         cur_step = 0
# #         for batch_img, batch_y in val_loader:
# #             batch_img, batch_y = batch_img.cuda(), batch_y.cuda()
# #
# #             batch_img = batch_img.float()
# #
# #             # compute output
# #             logits = model(batch_img)
# #             loss = criterion(logits, batch_y)
# #
# #             # measure accuracy and record loss
# #             acc1 = accuracy(logits, batch_y, topk=1)
# #             losses.update(loss.item(), batch_img.size(0))
# #             top1.update(float(acc1[0]), batch_img.size(0))
# #
# #             cur_step += 1
# #             if cur_step % config.train_print_step == 0:
# #                 print('test: [{0}/{1}]\t'
# #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                     cur_step, len(val_loader), loss=losses,
# #                     top1=top1))
# #
# #         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
# #
# #     return top1.avg,  losses.avg
# #
# # def validate_multi(val_loader, model, criterion, opt):
# #     """validation"""
# #     batch_time = AverageMeter()
# #     losses = AverageMeter()
# #     top1 = AverageMeter()
# #
# #     # switch to evaluate mode
# #     model.eval()
# #
# #     with torch.no_grad():
# #         cur_step = 0
# #         for batch_x, batch_img, batch_y in val_loader:
# #             # batch_len = len(batch_y.size(0))
# #             batch_x,batch_img, batch_y = batch_x.cuda(),batch_img.cuda(), batch_y.cuda()
# #             cls_output, logits = model(batch_x)
# #             net = resnet50(num_classes=3)
# #             net = net.eval()
# #             net.cuda()
# #             img_logits, img_feaures = net(batch_img)
# #             img_feaures = img_feaures.view(img_feaures.size(0), -1)
# #             final = torch.cat((cls_output, img_feaures), 1)
# #             fc = nn.Linear(2816, 3)
# #
# #             nn.init.normal_(fc.weight, std=0.001)
# #             nn.init.constant_(fc.bias, 0)
# #             trainable_list = nn.ModuleList([])
# #             trainable_list.append(model)
# #             trainable_list.append(fc)
# #             if torch.cuda.is_available():
# #                 trainable_list.cuda()
# #
# #             logits = fc(final)
# #
# #             # loss = criterion(probs, batch_y)
# #             loss = criterion(logits, batch_y)
# #
# #             # measure accuracy and record loss
# #             acc1 = accuracy(logits, batch_y, topk=1)
# #             losses.update(loss.item(), batch_x.size(0))
# #             top1.update(float(acc1[0]), batch_x.size(0))
# #
# #             cur_step += 1
# #             if cur_step % config.train_print_step == 0:
# #                 print('test: [{0}/{1}]\t'
# #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                     cur_step, len(val_loader), loss=losses,
# #                     top1=top1))
# #
# #         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
# #
# # def validate_crd_mutual(val_loader, model, criterion, opt):
# #
# #     """validation"""
# #     if opt==0:
# #         batch_time = AverageMeter()
# #         losses = AverageMeter()
# #         top1 = AverageMeter()
# #
# #         # switch to evaluate mode
# #         model.eval()
# #         model_img=model[0]
# #
# #
# #         with torch.no_grad():
# #             cur_step = 0
# #             for _, batch_img, batch_y, _, _ in val_loader:
# #                 batch_img, batch_y = batch_img.cuda(), batch_y.cuda()
# #
# #                 batch_img = batch_img.float()
# #
# #                 # compute output
# #                 logits = model_img(batch_img)
# #                 loss = criterion(logits, batch_y)
# #
# #                 # measure accuracy and record loss
# #                 acc1 = accuracy(logits, batch_y, topk=1)
# #                 losses.update(loss.item(), batch_img.size(0))
# #                 top1.update(float(acc1[0]), batch_img.size(0))
# #
# #                 cur_step += 1
# #                 if cur_step % config.train_print_step == 0:
# #                     print('test for image model: ')
# #                     print('test: [{0}/{1}]\t'
# #                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                         cur_step, len(val_loader), loss=losses,
# #                         top1=top1))
# #
# #             print('val img model * Acc@1 {top1.avg:.3f} '.format(top1=top1))
# #
# #         return top1.avg,  losses.avg
# #
# #     elif opt == 1:
# #         batch_time = AverageMeter()
# #         losses = AverageMeter()
# #         top1 = AverageMeter()
# #
# #         # switch to evaluate mode
# #         model.eval()
# #         model_text=model[-1]
# #
# #
# #         with torch.no_grad():
# #             cur_step = 0
# #             for batch_x, _, batch_y, _, _ in val_loader:
# #                 batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
# #
# #                 # compute output
# #                 logits = model_text(batch_x)
# #                 loss = criterion(logits, batch_y)
# #
# #                 # measure accuracy and record loss
# #                 acc1 = accuracy(logits, batch_y, topk=1)
# #                 losses.update(loss.item(), batch_x.size(0))
# #                 top1.update(float(acc1[0]), batch_x.size(0))
# #
# #                 cur_step += 1
# #                 if cur_step % config.train_print_step == 0:
# #                     print('test for text model: ')
# #                     print('test: [{0}/{1}]\t'
# #                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                         cur_step, len(val_loader), loss=losses,
# #                         top1=top1))
# #
# #             print('val text model * Acc@1 {top1.avg:.3f} '.format(top1=top1))
# #
# #         return top1.avg, losses.avg
# #
# #
# # def validate_2_stage(val_loader, module_list, criterion):
# #
# #     """validation"""
# #
# #     batch_time = AverageMeter()
# #     losses = AverageMeter()
# #     top1 = AverageMeter()
# #
# #     # switch to evaluate mode
# #     for model in module_list:
# #         model.eval()
# #     model_img=module_list[0]
# #     model_text=module_list[1]
# #     fc=module_list[-1]
# #
# #
# #
# #     with torch.no_grad():
# #         cur_step = 0
# #         for idx, data in enumerate(val_loader):
# #             batch_x, batch_img, batch_y, _, _ = data
# #             batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
# #             batch_img = batch_img.float()
# #
# #             feat_s, logits_img = model_img(batch_img, is_feat=True, preact=False)
# #             cls_output, logits_text = model_text(batch_x)
# #             f = torch.cat((feat_s[-1], cls_output), 1)
# #             logits = fc(f)
# #
# #             loss = criterion(logits, batch_y)
# #
# #             acc1 = accuracy(logits, batch_y, topk=1)
# #             losses.update(loss.item(), batch_y.size(0))
# #             top1.update(float(acc1[0]), batch_y.size(0))
# #
# #             if idx % config.train_print_step == 0:
# #                 print('test: [{0}/{1}]\t'
# #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# #                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
# #                     cur_step, len(val_loader), loss=losses,
# #                     top1=top1))
# #
# #         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
# #
# #     return top1.avg, losses.avg
# #
# # #########################################################################################################
#
#
# import sys
# import time
# import torch
# import os
# import errno
# import torch.nn as nn
# from conf import config
# from model.resnet import resnet50
# import numpy as np
#
# def mkdir_if_missing(directory):
#     if not os.path.exists(directory):
#         try:
#             os.makedirs(directory)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#
# class Logger(object):
#     def __init__(self, fpath=None):
#         self.console = sys.stdout
#         self.file = None
#         if fpath is not None:
#             mkdir_if_missing(os.path.dirname(fpath))
#             self.file = open(fpath, 'w')
#
#     def __del__(self):
#         self.close()
#
#     def __enter__(self):
#         pass
#
#     def __exit__(self, *args):
#         self.close()
#
#     def write(self, msg):
#         self.console.write(msg)
#         if self.file is not None:
#             self.file.write(msg)
#
#     def flush(self):
#         self.console.flush()
#         if self.file is not None:
#             self.file.flush()
#             os.fsync(self.file.fileno())
#
#     def close(self):
#         self.console.close()
#         if self.file is not None:
#             self.file.close()
#
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
# class LinearEmbed(nn.Module):
#     """Embedding module"""
#     def __init__(self, dim_in=128, dim_out=2):
#         super(LinearEmbed, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         # self.l2norm = Normalize(2)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         # x = self.l2norm(x)
#         return x
#
#
# def accuracy(output, target):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         batch_size = target.size(0)
#
#         probs = torch.softmax(output, 1)
#         pred=torch.argmax(probs, dim=1)
#
#         # pred
#         # tensor([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
#         #         0, 0, 1, 1, 0, 0, 1, 1], device='cuda:0')
#         # target
#         # tensor([0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1,
#         #         1, 0, 1, 1, 0, 0, 0, 1], device='cuda:0')
#
#         correct = pred.eq(target)
#         #tensor([ True,  True,  True,  True,  True,  True, False,  True,  True, False,
#         # False,  True, False,  True,  True,  True,  True,  True,  True,  True,
#         # False,  True,  True,  True,  True,  True,  True,  True,  True, False,
#         #  True,  True], device='cuda:0')
#
#         res = []
#         correct_k = correct.view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#
# def precision(output, target):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#
#         probs = torch.softmax(output, 1)
#         pred = torch.argmax(probs, dim=1)
#
#         count_1,count_correct=0,0
#
#         for i in range(len(pred)):
#             if pred[i]==1:
#                 count_1+=1
#                 if pred[i]==target[i]:
#                     count_correct+=1
#         prec=float(count_correct)*(100.0 / float(count_1))
#         return prec
#
#
# def recall(output, target):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#
#         probs = torch.softmax(output, 1)
#         pred = torch.argmax(probs, dim=1)
#
#         count_correct,count_target_1=0,0
#         for i in range(len(pred)):
#             if pred[i]==1 and pred[i]==target[i]:
#                 count_correct+=1
#             if target[i]==1:
#                 count_target_1+=1
#         rec=float(count_correct)*(100.0 / float(count_target_1))
#         return rec
#
# def f1(output, target):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         probs = torch.softmax(output, 1)
#         pred = torch.argmax(probs, dim=1)
#
#         count_predict_1,count_1_correct,count_target_1=0,0,0
#         for i in range(len(pred)):
#             if pred[i]==1:
#                 count_predict_1+=1
#             if pred[i]==1 and pred[i]==target[i]:
#                 count_1_correct+=1
#         count_target_1=target.float().sum(0,keepdim=True)
#         precision=float(count_1_correct)/float(count_predict_1)
#         recall=float(count_1_correct)/float(count_target_1)
#         f= 2*(precision*recall)/(precision+recall)
#
#         return f
#
#
#
# def validate_text(val_loader, model, criterion):
#     """validation"""
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         cur_step = 0
#         for batch_x, batch_y in val_loader:
#             batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
#
#             # input = input.float()
#             # if torch.cuda.is_available():
#             #     input = input.cuda()
#             #     target = target.cuda()
#
#             # compute output
#             cls_output, logits = model(batch_x)
#             loss = criterion(logits, batch_y)
#
#             # measure accuracy and record loss
#             acc1 = accuracy(logits, batch_y, topk=1)
#             losses.update(loss.item(), batch_x.size(0))
#             top1.update(float(acc1[0]), batch_x.size(0))
#
#
#             # measure elapsed time
#
#             # end = time.time()
#             cur_step += 1
#             if cur_step % config.train_print_step == 0:
#                 print('test: [{0}/{1}]\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                        cur_step, len(val_loader),  loss=losses,
#                        top1=top1))
#
#         print('val * Acc@1 {top1.avg:.3f} '
#               .format(top1=top1))
#
#     return top1.avg,  losses.avg
#
#
# def validate_img(val_loader, model, criterion):
#     """validation"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         cur_step = 0
#         for _, batch_img, batch_y, _, _ in val_loader:
#             batch_img, batch_y = batch_img.cuda(), batch_y.cuda()
#
#             batch_img = batch_img.float()
#
#             # compute output
#             logits = model(batch_img)
#             loss = criterion(logits, batch_y)
#
#             # measure accuracy and record loss
#             acc1 = accuracy(logits, batch_y, topk=1)
#             losses.update(loss.item(), batch_img.size(0))
#             top1.update(float(acc1[0]), batch_img.size(0))
#
#             cur_step += 1
#             if cur_step % config.train_print_step == 0:
#                 print('test: [{0}/{1}]\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                     cur_step, len(val_loader), loss=losses,
#                     top1=top1))
#
#         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#
#     return top1.avg,  losses.avg
#
# def validate_multi(val_loader, model, criterion):
#     """validation"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         cur_step = 0
#         for batch_x, batch_img, batch_y in val_loader:
#             # batch_len = len(batch_y.size(0))
#             batch_x,batch_img, batch_y = batch_x.cuda(),batch_img.cuda(), batch_y.cuda()
#             cls_output, logits = model(batch_x)
#             net = resnet50(num_classes=3)
#             net = net.eval()
#             net.cuda()
#             img_logits, img_feaures = net(batch_img)
#             img_feaures = img_feaures.view(img_feaures.size(0), -1)
#             final = torch.cat((cls_output, img_feaures), 1)
#             fc = nn.Linear(2816, 3)
#
#             nn.init.normal_(fc.weight, std=0.001)
#             nn.init.constant_(fc.bias, 0)
#             trainable_list = nn.ModuleList([])
#             trainable_list.append(model)
#             trainable_list.append(fc)
#             if torch.cuda.is_available():
#                 trainable_list.cuda()
#
#             logits = fc(final)
#
#             # loss = criterion(probs, batch_y)
#             loss = criterion(logits, batch_y)
#
#             # measure accuracy and record loss
#             acc1 = accuracy(logits, batch_y, topk=1)
#             losses.update(loss.item(), batch_x.size(0))
#             top1.update(float(acc1[0]), batch_x.size(0))
#
#             cur_step += 1
#             if cur_step % config.train_print_step == 0:
#                 print('test: [{0}/{1}]\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                     cur_step, len(val_loader), loss=losses,
#                     top1=top1))
#
#         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#     return top1.avg, losses.avg
#
#
# def validate_distillti(val_loader, model, criterion):
#     """validation"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         cur_step = 0
#         for _, batch_img, batch_y in val_loader:
#             batch_img, batch_y = batch_img.cuda(), batch_y.cuda()
#
#             batch_img = batch_img.float()
#
#             # compute output
#             logits = model(batch_img)
#             loss = criterion(logits, batch_y)
#
#             # measure accuracy and record loss
#             acc1 = accuracy(logits, batch_y, topk=1)
#             losses.update(loss.item(), batch_img.size(0))
#             top1.update(float(acc1[0]), batch_img.size(0))
#
#             cur_step += 1
#             if cur_step % config.train_print_step == 0:
#                 print('test: [{0}/{1}]\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                     cur_step, len(val_loader), loss=losses,
#                     top1=top1))
#
#         print('val * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#
#     return top1.avg,  losses.avg
#
# def adjust_learning_rate(epoch, opt, optimizer):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
#     if steps > 0:
#         new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lr
#
#
#
# def train_crd(epoch, train_loader, module_list, criterion_list, optimizer, opt):
#     for module in module_list:
#         module.train()
#         # set teacher as eval()
#     module_list[-1].eval()
#
#     criterion_cls = criterion_list[0]
#     criterion_div = criterion_list[1]
#     criterion_kd = criterion_list[2]
#
#     model_s = module_list[0]
#     model_t = module_list[-1]
#
#
#     losses_cls = AverageMeter()
#     losses_kd = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     for idx,data in enumerate(train_loader):
#         batch_x, batch_img, batch_y, index, contrast_idx=data
#         batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
#         index,contrast_idx=index.cuda(),contrast_idx.cuda()
#         # print('index',index)
#         # print('contrast_idx',contrast_idx)
#
#         feat_s, logits_s = model_s(batch_img, is_feat=True, preact=False)
#         with torch.no_grad():
#             cls_output, logits_t = model_t(batch_x)
#             cls_output = cls_output.detach()
#
#         loss_cls = criterion_cls(logits_s, batch_y)
#         loss_div = criterion_div(logits_s, logits_t)
#         loss_kd = criterion_kd(feat_s[-1], cls_output, index, contrast_idx)
#
#         train_loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
#
#         acc1 = accuracy(logits_s, batch_y, topk=1)
#         losses.update(train_loss.item(), batch_img.size(0))
#         losses_kd.update(loss_kd.item(), batch_img.size(0))
#         losses_cls.update(loss_cls.item(), batch_img.size(0))
#         top1.update(float(acc1[0]), batch_img.size(0))
#
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#
#         if idx % config.train_print_step == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'losses_cls {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
#                   'losses_kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                 epoch, idx, len(train_loader),
#                 loss=losses, losses_cls=losses_cls, losses_kd=losses_kd, top1=top1))
#             sys.stdout.flush()
#
#     print(' *train Acc@1 {top1.avg:.3f}'.format(top1=top1))
#
#
# def train_crd_mutual(epoch, train_loader, module_list, criterion_list, optimizers, opt):
#     for module in module_list:
#         module.train()
#
#     criterion_cls = criterion_list[0]
#     criterion_div = criterion_list[1]
#     criterion_kd = criterion_list[2]
#
#     model_img = module_list[0]
#     model_text = module_list[-1]
#
#     losses_img=AverageMeter()
#     losses_text = AverageMeter()
#     top1_img=AverageMeter()
#     top1_text = AverageMeter()
#
#
#     for idx,data in enumerate(train_loader):
#         _,batch_x, batch_img, batch_y, index, contrast_idx=data
#         batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
#         index,contrast_idx=index.cuda(),contrast_idx.cuda()
#
#
#         feat_s, logits_img = model_img(batch_img, is_feat=True, preact=False)
#         cls_output, logits_text = model_text(batch_x)
#
#         loss_cls_img=criterion_cls(logits_img,batch_y)
#         loss_cls_text=criterion_cls(logits_text,batch_y)
#
#
#         loss_div_img = criterion_div(logits_img, logits_text)
#         loss_div_text = criterion_div(logits_text, logits_img)
#
#
#         loss_kd = criterion_kd(feat_s[-1], cls_output, index, contrast_idx)
#
#         train_loss_img = opt.gamma * loss_cls_img + opt.alpha * loss_div_img + opt.beta * loss_kd
#         train_loss_text = opt.gamma * loss_cls_text + opt.alpha * loss_div_text + opt.beta * loss_kd
#
#         acc1_img = accuracy(logits_img, batch_y, topk=1)
#         losses_img.update(train_loss_img.item(), batch_img.size(0))
#         top1_img.update(float(acc1_img[0]), batch_img.size(0))
#
#         optimizers[0].zero_grad()
#         train_loss_img.backward(retain_graph=True)
#         optimizers[0].step()
#
#         acc1_text = accuracy(logits_text, batch_y, topk=1)
#         losses_text.update(train_loss_text.item(), batch_x.size(0))
#         top1_text.update(float(acc1_text[0]), batch_x.size(0))
#
#         optimizers[1].zero_grad()
#         train_loss_text.backward()
#         optimizers[1].step()
#
#
#
#         if idx % config.train_print_step == 0:
#             print('Image Model:')
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                 epoch, idx, len(train_loader),
#                 loss=losses_img, top1=top1_img))
#
#             print('Text Model:')
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                 epoch, idx, len(train_loader),
#                 loss=losses_text, top1=top1_text))
#     print(' Train image model : Acc@1 {top1.avg:.3f}'.format(top1=top1_img))
#     print(' Train text model : Acc@1 {top1.avg:.3f}'.format(top1=top1_text))
#     sys.stdout.flush()
#
#
# def validate_crd(val_loader, model, criterion, opt):
#
#     """validation"""
#     if opt==0:
#         batch_time = AverageMeter()
#         losses = AverageMeter()
#         top1 = AverageMeter()
#
#         # switch to evaluate mode
#         model.eval()
#         model_img=model[0]
#
#
#         with torch.no_grad():
#             cur_step = 0
#             for _, _, batch_img, batch_y, _, _ in val_loader:
#                 batch_img, batch_y = batch_img.cuda(), batch_y.cuda()
#
#                 batch_img = batch_img.float()
#
#                 # compute output
#                 logits = model_img(batch_img)
#                 loss = criterion(logits, batch_y)
#
#                 # measure accuracy and record loss
#                 acc1 = accuracy(logits, batch_y)
#                 losses.update(loss.item(), batch_img.size(0))
#                 top1.update(float(acc1[0]), batch_img.size(0))
#
#                 cur_step += 1
#                 if cur_step % config.train_print_step == 0:
#                     print('test for image model: ')
#                     print('test: [{0}/{1}]\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                         cur_step, len(val_loader), loss=losses,
#                         top1=top1))
#
#             print('val img model * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#
#         return top1.avg,  losses.avg
#
#     elif opt == 1:
#         batch_time = AverageMeter()
#         losses = AverageMeter()
#         top1 = AverageMeter()
#
#         # switch to evaluate mode
#         model.eval()
#         model_text=model[-1]
#
#
#         with torch.no_grad():
#             cur_step = 0
#             for _, batch_x, _, batch_y, _, _ in val_loader:
#                 batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
#
#                 # compute output
#                 cls_output, logits = model_text(batch_x)
#                 loss = criterion(logits, batch_y)
#
#                 # measure accuracy and record loss
#                 acc1 = accuracy(logits, batch_y, topk=1)
#                 losses.update(loss.item(), batch_x.size(0))
#                 top1.update(float(acc1[0]), batch_x.size(0))
#
#                 cur_step += 1
#                 if cur_step % config.train_print_step == 0:
#                     print('test for text model: ')
#                     print('test: [{0}/{1}]\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                         cur_step, len(val_loader), loss=losses,
#                         top1=top1))
#
#             print('val text model * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#
#         return top1.avg, losses.avg
#
#
#
#
#
#
import sys
import time
import torch
import os
import errno
import torch.nn as nn
from conf import config
from model.resnet import resnet50
import numpy as np

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0



    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class LinearEmbed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=128, dim_out=2):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        # x = self.l2norm(x)
        return x



def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        probs = torch.softmax(output, 1)
        pred=torch.argmax(probs, dim=1)

        # pred
        # tensor([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
        #         0, 0, 1, 1, 0, 0, 1, 1], device='cuda:0')
        # target
        # tensor([0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1,
        #         1, 0, 1, 1, 0, 0, 0, 1], device='cuda:0')

        correct = pred.eq(target)
        #tensor([ True,  True,  True,  True,  True,  True, False,  True,  True, False,
        # False,  True, False,  True,  True,  True,  True,  True,  True,  True,
        # False,  True,  True,  True,  True,  True,  True,  True,  True, False,
        #  True,  True], device='cuda:0')

        res = []
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def metric(output,target,for_fake=True):

    with torch.no_grad():
        batch_size = target.size(0)

        probs = torch.softmax(output, 1)
        pred=torch.argmax(probs, dim=1)
        if for_fake:
            count_0, count_correct_0,count_target_0 = 0, 0, 0
            for i in range(len(pred)):
                if pred[i] == 0:
                    count_0 += 1
                    if pred[i] == target[i]:
                        count_correct_0 += 1
                if target[i] == 0:
                    count_target_0 += 1
            return count_0, count_correct_0,count_target_0
        else:
            count_1, count_correct_1, count_target_1 = 0, 0, 0
            for i in range(len(pred)):
                if pred[i] == 1:
                    count_1 += 1
                    if pred[i] == target[i]:
                        count_correct_1 += 1
                if target[i] == 1:
                    count_target_1 += 1
            return count_1, count_correct_1, count_target_1








def precision_fake(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_0,count_correct=0,0

        for i in range(len(pred)):
            if pred[i]==0:
                count_0+=1
                if pred[i]==target[i]:
                    count_correct+=1
        prec = float(count_correct) * (100.0 / float(count_0))
        # try:
        #     prec=float(count_correct)*(100.0 / float(count_0))
        # except ZeroDivisionError:
        #     prec=None
        return prec


def recall_fake(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_correct,count_target_0=0,0
        for i in range(len(pred)):
            if pred[i]==0 and pred[i]==target[i]:
                count_correct+=1
            if target[i]==0:
                count_target_0+=1
        rec = float(count_correct) * (100.0 / float(count_target_0))
        # try:
        #     rec=float(count_correct)*(100.0 / float(count_target_0))
        # except ZeroDivisionError:
        #     rec=None
        return rec

def f1_fake(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_predict_0,count_0_correct,count_target_0=0,0,0
        for i in range(len(pred)):
            if pred[i]==0:
                count_predict_0+=1
            if pred[i]==0 and pred[i]==target[i]:
                count_0_correct+=1
        count_target_0=len(target.float())-target.float().sum(0,keepdim=True)
        precision = float(count_0_correct) / float(count_predict_0)
        recall = float(count_0_correct) / float(count_target_0)
        f = 2 * (precision * recall) / (precision + recall)
        # try:
        #     precision=float(count_0_correct)/float(count_predict_0)
        # except ZeroDivisionError:
        #     precision=None
        # try:
        #     recall=float(count_0_correct)/float(count_target_0)
        # except ZeroDivisionError:
        #     recall=None
        # if not precision or not recall:
        #     f=None
        # else:
        #     try:
        #         f= 2*(precision*recall)/(precision+recall)
        #     except ZeroDivisionError:
        #         f=None

        return f



def precision_real(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_1,count_correct=0,0

        for i in range(len(pred)):
            if pred[i]==1:
                count_1+=1
                if pred[i]==target[i]:
                    count_correct+=1
        try:
            prec=float(count_correct)*(100.0 / float(count_1))
        except ZeroDivisionError:
            prec=None
        return prec


def recall_real(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_correct,count_target_1=0,0
        for i in range(len(pred)):
            if pred[i]==1 and pred[i]==target[i]:
                count_correct+=1
            if target[i]==1:
                count_target_1+=1
        try:
            rec=float(count_correct)*(100.0 / float(count_target_1))
        except ZeroDivisionError:
            rec=None
        return rec

def f1_real(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        probs = torch.softmax(output, 1)
        pred = torch.argmax(probs, dim=1)

        count_predict_1,count_1_correct,count_target_1=0,0,0
        for i in range(len(pred)):
            if pred[i]==1:
                count_predict_1+=1
            if pred[i]==1 and pred[i]==target[i]:
                count_1_correct+=1
        count_target_1=target.float().sum(0,keepdim=True)
        try:
            precision=float(count_1_correct)/float(count_predict_1)
        except ZeroDivisionError:
            precision=None
        try:
            recall=float(count_1_correct)/float(count_target_1)
        except ZeroDivisionError:
            recall=None
        if not precision or not recall:
            f=None
        else:
            try:
                f= 2*(precision*recall)/(precision+recall)
            except ZeroDivisionError:
                f = None

        return f











# def accuracy(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = topk
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []

#         correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


