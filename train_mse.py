#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import argparse
from models.selector import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader, get_train_loader
from config import get_arguments
from typing import Any, Union


class MSE():
    """
    Fine Pruning Defense is described in the paper 'Fine-Pruning'_ by KangLiu. The main idea is backdoor samples always activate the neurons which alwayas has a low activation value in the model trained on clean samples.
    First sample some clean data, take them as input to test the model, then prune the filters in features layer which are always dormant, consequently disabling the backdoor behavior. Finally, finetune the model to eliminate the threat of backdoor attack.
    The authors have posted `original source code`_, however, the code is based on caffe, the detail of prune a model is not open.
    Args:
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epoch of finetuning. Default: 10.
    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185
    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense
    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune-
        https://github.com/ain-soph/trojanzoo/blob/3cdd5519c0a946e66fbec5bf34a956d1982d9e1f/trojanvision/defenses/backdoor/fine_pruning.py
    """

    name = 'fine_pruning'

    def __init__(self, opt, optimizer, input_model, criterions, test_clean_loader, test_bad_loader, train_loader,
                 prune_ratio=0.98):
        self.prune_ratio = prune_ratio
        self.input_model = input_model
        self.criterions = criterions
        self.train_loader = train_loader
        self.test_clean_loader = test_clean_loader
        self.test_bad_loader = test_bad_loader
        self.opt = opt
        self.optimizer = optimizer

    def detect(self):
        defence_module = self.input_model
        for module in defence_module.modules():
            if isinstance(module, nn.Conv2d):
                self.conv_module: nn.Module = prune.identity(module, 'weight')
        self.length = self.conv_module.out_channels
        self.prune_num: int = int(self.length * self.prune_ratio)
        self.prune()

    def prune(self):
        mask = self.conv_module.weight_mask
        # self.prune_step(mask, prune_num=max(self.prune_num - 10, 0))
        # calc the clean_acc and backdoor_acc
        clean_acc, bd_acc = test(self.test_clean_loader, self.test_bad_loader, self.input_model, self.criterions, 0)

        for i in range(2):
            print('Iter: ', i)
            self.prune_step(mask, self.prune_num)
            clean_acc_1, bd_acc = test(self.test_clean_loader, self.test_bad_loader, self.input_model, self.criterions,
                                       0)
            if clean_acc - clean_acc_1 > 20:
                break

    def prune_step(self, mask: torch.Tensor, prune_num: int = 1):
        with torch.no_grad():
            feats_list = []
            for idx, (img, target) in enumerate(self.train_loader):
                img = img.cuda()
                # target = target.cuda()
                with torch.no_grad():
                    _, _, _, _, output_s = self.input_model(img)
                    feat_s = self.input_model.get_final_fm(img)
                feats_list.append(feat_s)
            feats_list = torch.cat(feats_list).mean(dim=0)
            idx_rank = self.to_list(feats_list.argsort())
        counter = 0
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                counter += 1
                # print(f'    {output_iter(counter, prune_num)} Prune {idx:4d} / {len(idx_rank):4d}')
                print('counter:', counter)
                print('num', prune_num)
                print('all_num', len(idx_rank))
                if counter >= min(prune_num, len(idx_rank)):
                    break

    def to_list(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x.tolist()
        return list(x)

    def fineprune(self):
        # retrain the prune model
        for epoch in range(0, 1000):
            adjust_learning_rate(self.optimizer, epoch, self.opt.lr)
            train_step_ce(self.opt, self.train_loader, self.input_model, self.optimizer, self.criterions, epoch)
            clean_acc, bd_acc = test(self.test_clean_loader, self.test_bad_loader, self.input_model, self.criterions,
                                     0)
            print("clean_acc", clean_acc)
            print("bd_acc", bd_acc)


def train_step(opt, optimizer, nets, criterions, test_clean_loader, test_bad_loader, train_loader):
    snet = nets['snet']
    snet.train()
    prune_model = FinePruning(opt, optimizer, snet, criterions, test_clean_loader, test_bad_loader, train_loader, )
    prune_model.detect()
    prune_model.fineprune()


def train_step_ce(opt, train_loader, nets, optimizer, criterions, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    snet = nets

    ###criterion XX for KD algoriithm

    criterionCls = criterions['criterionCls']
    snet.train()
    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        activation1_s, activation2_s, activation3_s, feat_s, output_s = snet(img)
        ####################################################
        '''
        baseline cls: 82.47777777777777,1.7444444444444445
        '''
        cls_loss = criterionCls(output_s, target)

        loss_sum = cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        losses.update(loss_sum.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                 top1=top1, top5=top5))


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=False,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')

    nets = {'snet': student}

    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(1):
        # train every epoch
        criterions = {'criterionCls': criterionCls}
        snet = nets['snet']
        acc_clean, acc_bad = test(test_clean_loader, test_bad_loader, snet, criterions, epoch)

        train_step(opt, optimizer, nets, criterions, test_clean_loader, test_bad_loader, train_loader)

        # evaluate on testing set
        print('testing the models......')

        acc_clean, acc_bad = test(test_clean_loader, test_bad_loader, snet, criterions, epoch)

        # remember best precision and save checkpointk
        if opt.save:
            is_best = acc_bad > opt.threshold_bad
            opt.threshold_bad = min(acc_bad, opt.threshold_bad)

            best_clean_acc = acc_clean
            best_bad_acc = acc_bad

            s_name = opt.s_name + '-S-model_best_prune.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def test(test_clean_loader, test_bad_loader, net, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = net
    criterionCls = criterions['criterionCls']
    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, _, output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, _, output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = 'backdoor_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=(
        "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean[0], acc_bd[0]


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    train(opt)


if (__name__ == '__main__'):
    main()
