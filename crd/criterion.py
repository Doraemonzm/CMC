import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7

#
# class CRDLoss(nn.Module):
#     """CRD Loss function
#     includes two symmetric parts:
#     (a) using teacher as anchor, choose positive and negatives over the student side
#     (b) using student as anchor, choose positive and negatives over the teacher side
#     Args:
#         opt.s_dim: the dimension of student's feature
#         opt.t_dim: the dimension of teacher's feature
#         opt.feat_dim: the dimension of the projection space
#         opt.nce_k: number of negatives paired with each positive
#         opt.nce_t: the temperature
#         opt.nce_m: the momentum for updating the memory buffer
#         opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
#     """
#     def __init__(self, opt):
#         super(CRDLoss, self).__init__()
#         self.embed_s = Embed(2048, opt.feat_dim)
#         self.embed_t = Embed(3840, opt.feat_dim)
#         self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m,opt.crd_op)
#         self.criterion_t = ContrastLoss(opt.n_data)
#         self.criterion_s = ContrastLoss(opt.n_data)
#         self.criterion_cls=nn.CrossEntropyLoss()
#
#     def forward(self, f_s, f_t, idx, contrast_idx=None):
#         """
#         Args:
#             f_s: the feature of student network, size [batch_size, s_dim]
#             f_t: the feature of teacher network, size [batch_size, t_dim]
#             idx: the indices of these positive samples in the dataset, size [batch_size]
#             contrast_idx: the indices of negative samples, size [batch_size, nce_k]
#         Returns:
#             The contrastive loss
#         """
#         f_s = self.embed_s(f_s)
#         f_t = self.embed_t(f_t)
#         out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
#         s_loss = self.criterion_s(out_s)
#         t_loss = self.criterion_t(out_t)
#         # if opinon==0:
#         #     loss=s_loss
#         # else :
#         #     loss=t_loss
#         loss = s_loss + t_loss
#         return loss

#
class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, feat_dim, n_data, nce_k, nce_t, nce_m):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(2048, feat_dim)
        self.embed_t = Embed(15360, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion_t = ContrastLoss(n_data)
        self.criterion_s = ContrastLoss(n_data)
        self.criterion_cls=nn.CrossEntropyLoss()

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        # if opinon==0:
        #     loss=s_loss
        # else :
        #     loss=t_loss
        loss = s_loss + t_loss
        return loss




class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class CRDsoftmax(nn.Module):

    def __init__(self, opt):
        super(CRDsoftmax, self).__init__()
        self.embed_s = Embed(25088, opt.feat_dim)
        self.embed_t = Embed(768, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = nn.CrossEntropyLoss()
        self.criterion_s = nn.CrossEntropyLoss()

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        bsz = f_s.shape[0]
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        out_s=out_s.squeeze()
        out_t = out_t.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        s_loss = self.criterion_s(out_s,label)
        t_loss = self.criterion_t(out_t,label)
        loss = s_loss + t_loss
        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss