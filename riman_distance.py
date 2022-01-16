from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from at import AT
import numpy as np
import matplotlib.pyplot as plt


# the bn and relu module for the feature map
class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout).cuda()
        self.bn = nn.BatchNorm1d(nout).cuda()
        self.relu = nn.ReLU(False).cuda()

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x))).cuda()
        return self.bn(self.linear(x)).cuda()


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim)).to('cuda')
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim)).to('cuda')
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        # p_logit = torch.matmul(self.p_t, self.p_s.t())

        # logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        logit = torch.einsum('bstq,btq->bts', bilinear_key, query)

        atts = F.softmax(logit, dim=2)
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        '''
        transfrom the teacher attention feature set to the embedding vector
        '''
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[0], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        '''
        transfrom the student attention feature set to the embedding vector
        '''
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])  # channel_key+sample_key

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes]).cuda()
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2).cuda() for f_s in g_s]  # calc_the channle.mean
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s.cuda()) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                          dim=1).view(bs * self.s, -1).cuda()  # Bs x h
        bilinear_key = self.bilinear(key.cuda(), relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    # using the adaptive average pool to get the feature
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)  #

        return g_s


class IMG(nn.Module):
    def __init__(self, opt):
        super(IMG, self).__init__()

        self.w_irg_vert = opt.w_irg_vert
        self.w_irg_edge = opt.w_irg_edge
        self.w_irg_tran = opt.w_irg_tran
        self.margin = 1
        self.alpha = 0.1
        self.attention = Attention(opt)

    def forward(self, irg_s, irg_t):
        fm_s0, fm_s1, fm_s2, feat_s, out_s = irg_s
        fm_t0, fm_t1, fm_t2, feat_t, out_t = irg_t

        criterionAT = AT(2.0)
        loss_at = criterionAT(fm_s0, fm_t0) * 2000 + criterionAT(fm_s1, fm_t1) * 5000 + criterionAT(fm_s2, fm_t2) * 2000  # Node dis
        loss_irg_vert = loss_at
        fm_s0_attention = criterionAT.attention_map(fm_s0)
        fm_t0_attention = criterionAT.attention_map(fm_t0)

        fm_s1_attention = criterionAT.attention_map(fm_s1)
        fm_t1_attention = criterionAT.attention_map(fm_t1)

        fm_s2_attention = criterionAT.attention_map(fm_s2)
        fm_t2_attention = criterionAT.attention_map(fm_t2)

        # the newly ways to calculate the global attention vector
        AT_s = [fm_s0_attention, fm_s1_attention, fm_s2_attention]
        AT_t = [fm_t0_attention, fm_t1_attention, fm_t2_attention]
        loss_global_list = self.attention(AT_s, AT_t)
        loss_global = 0
        for i in range(len(AT_s)):
            loss_global += loss_global_list[i]
        # fm_s3_attention = criterionAT.attention_map(feat_s)
        # fm_t3_attention = criterionAT.attention_map(feat_t)

        # attentio_vision = fm_s0_attention.detach()
        # # self.viz(fm_s0.detach())
        # self.viz(attentio_vision)
        # #
        # # attentio_vision = fm_t0_attention.detach()
        # # # self.viz(fm_t0)
        # # # self.viz(attentio_vision)
        # attentio_vision = fm_s1_attention.detach()
        # # self.viz(fm_s1.detach())
        # self.viz(attentio_vision)
        # # attentio_vision = fm_t1_attention.detach()
        # # # self.viz(fm_t1.detach())
        # # # self.viz(attentio_vision)
        # attentio_vision = fm_s2_attention.detach()
        # # # self.viz(fm_s2.detach())
        # self.viz(attentio_vision)
        # attentio_vision = fm_t2_attention.detach()
        # self.viz(fm_t2.detach())
        # self.viz(attentio_vision)

        # attentio_vision = fm_s3_attention.detach()
        # self.viz(attentio_vision)
        # attentio_vision = fm_t3_attention.detach()
        # self.viz(attentio_vision)

        # feat_s_attention = criterionAT.attention_map(feat_s)
        # feat_t_attention = criterionAT.attention_map(feat_t)
        #
        irg_tran_s2 = self.euclidean_dist_fms(fm_s0_attention, fm_s2_attention, squared=True)

        irg_tran_s1 = self.euclidean_dist_fms(fm_s1_attention, fm_s2_attention, squared=True)

        irg_tran_s0 = self.euclidean_dist_fms(fm_s0_attention, fm_s1_attention, squared=True)

        irg_tran_t2 = self.euclidean_dist_fms(fm_t0_attention, fm_t2_attention, squared=True)
        irg_tran_t1 = self.euclidean_dist_fms(fm_t1_attention, fm_t2_attention, squared=True)
        irg_tran_t0 = self.euclidean_dist_fms(fm_t0_attention, fm_t1_attention, squared=True)

        loss_irg_edge = (F.mse_loss(irg_tran_s0, irg_tran_t0) + F.mse_loss(irg_tran_s1, irg_tran_t1) + F.mse_loss(
            irg_tran_s2, irg_tran_t2)) / 3

        loss = (
                self.w_irg_vert * loss_irg_vert
                + self.w_irg_edge *  loss_irg_edge
                + loss_global
        )

        return loss

    def euclidean_dist_fms(self, fm1, fm2, squared=False, eps=1e-12):

        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:, 0::2, :, :] + fm2[:, 1::2, :, :]) / 2.0

        fm1 = fm1.view(fm1.size(0), -1)
        fm2 = fm2.view(fm2.size(0), -1)
        fms_dist = torch.sum(torch.pow(fm1 - fm2, 2), dim=-1).clamp(min=eps)

        if not squared:
            fms_dist = fms_dist.sqrt()

        fms_dist = fms_dist / fms_dist.max()

        return fms_dist

    def euclidean_dist_fm(self, fm, squared=False, eps=1e-12):
        fm = fm.view(fm.size(0), -1)
        fm_square = fm.pow(2).sum(dim=1)
        fm_prod = torch.mm(fm, fm.t())
        fm_dist = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)

        if not squared:
            fm_dist = fm_dist.sqrt()

        fm_dist = fm_dist.clone()
        fm_dist[range(len(fm)), range(len(fm))] = 0
        fm_dist = fm_dist / fm_dist.max()

        return fm_dist

    def euclidean_dist_feat(self, feat, squared=False, eps=1e-12):
        '''
		Calculating the IRG edge of feat.
		'''
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0
        feat_dist = feat_dist / feat_dist.max()

        return feat_dist

    def resize(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:, 0::2, :, :] + fm2[:, 1::2, :, :]) / 2.0

            return fm1, fm2

    # def riman_distance(self, anchor, positive, negative):
    #
    #     distance_positive = (anchor - positive).pow(2).sum(1)
    #     distance_positive = distance_positive
    #     distance_negative = (anchor - negative).pow(2).sum(1)
    #     distance_negative = distance_negative
    #     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #     cos_reg = cos(positive, anchor).sum(0)
    #     losses = F.relu((distance_positive - distance_negative + self.alpha * cos_reg))
    #     return losses.mean()
    #     #

    # def riemannian_distance(self, x, y):
    #     if x.size(2) > y.size(2):
    #         x = F.adaptive_avg_pool2d(x, (y.size(2), y.size(3)))
    #     if x.size(1) < y.size(1):
    #         y = (y[:, 0::2, :, :] + y[:, 1::2, :, :]) / 2.0
    #     x_norm = torch.sum(x ** 2, dim=-1)
    #     y_norm = torch.sum(y ** 2, dim=-1)
    #     d_norm = torch.sum((x - y) ** 2, dim=-1)
    #     cc = 1 + 2 * d_norm / ((1 - x_norm) * (1 - y_norm))
    #     cc = cc / cc.max()
    #     return cc

    def viz(self, input):
        x = input[0][0].cpu()
        # 最多显示4张图
        # min_num = np.minimum(4, x.size()[0])
        plt.imshow(x, interpolation='bicubic')
        plt.show()
