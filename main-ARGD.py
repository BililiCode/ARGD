from torch import nn
from models.selector import *
from utils.util import *
from data_loader import get_train_loader, get_test_loader
from at import AT
from riman_distance import ARGD
from config import get_arguments
import datetime



def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    ###criterion XX for KD algoriithm
    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']
    criterionARGD = criterions['criterionARGD']
    feat_s_last = None
    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
        # x = img[0][0].cpu()
        # plt.imshow(x, interpolation='bicubic')
        # plt.show()

        activation1_s, activation2_s, activation3_s, feat_s, output_s = snet(img)
        with torch.no_grad():
            activation1_t, activation2_t, activation3_t, feat_t, output_t = tnet(img)
        if feat_s_last is None or (feat_s_last.size(0) != (feat_s.size(0))):
            feat_s_last = feat_s
        else:
            pass
        ####################################################
        cls_loss = criterionCls(output_s, target)
        ####################################################
        '''
        NAD loss function
        '''
        # at3_loss = criterionAT(activation3_s, activation3_t.detach()) * 5000
        #
        # at2_loss = criterionAT(activation2_s, activation2_t.detach()) * 5000
        # at1_loss = criterionAT(activation1_s, activation1_t.detach()) * 5000
        # at_loss = at1_loss + at2_loss + at3_loss + cls_loss
        ####################################################
        # adaptive factor to irg_loss to change this

        ###############################################
        '''
        ARGD:
        '''
        ARG_loss = criterionARGD([activation1_s, activation2_s, activation3_s, feat_s, output_s],
                                [activation1_t.detach(), activation2_t.detach(),
                                 activation3_t.detach(),
                                 feat_t.detach(),
                                 output_t.detach()])
        ARG_loss = cls_loss + ARG_loss

        loss_sum = ARG_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        losses.update(loss_sum.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        feat_s_last = feat_s.detach()
        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                 top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.eval()
    tnet.eval()
    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, _, output_s = snet(img)
            _, _, _, _, output_t = tnet(img)

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
            activation1_s, activation2_s, activation3_s, feat_s, output_s = snet(img)
            activation1_t, activation2_t, activation3_t, feat_t, output_t = tnet(img)

            at3_loss = criterionAT(activation3_s, activation3_t).detach()
            at2_loss = criterionAT(activation2_s, activation2_t).detach()
            at1_loss = criterionAT(activation1_s, activation1_t).detach()
            at_loss = at3_loss + at2_loss + at1_loss
            cls_loss = criterionCls(output_s, target).detach()
        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
    df = pd.DataFrame(test_process, columns=(
        "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "test_bad_at_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd



def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    teacher = select_model(dataset=opt.data_name,
                           model_name='WRN-40-2',
                           pretrained=True,
                           pretrained_models_path=opt.t_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished teacher model init...')

    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')
    criterionAT = AT(2.0)
    teacher.eval()
    opt.image_size = 32
    # get the dim in feat for teacher and student model
    data = torch.randn(1, 3, opt.image_size, opt.image_size).to(opt.device)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        activation1_t, activation2_t, activation3_t, feat_t, _ = teacher(data)
        activation1_s, activation2_s, activation3_s, feat_s, _ = student(data)
    AT_s = [criterionAT.attention_map(activation1_s), criterionAT.attention_map(activation2_s), criterionAT.attention_map(activation3_s)]
    AT_t = [criterionAT.attention_map(activation1_t), criterionAT.attention_map(activation2_t), criterionAT.attention_map(activation3_t)]
    opt.s_shapes = [AT_s[i].size() for i in range(3)]  # get the layer feat from the s feat shape
    opt.t_shapes = [AT_t[i][0].size() for i in range(3)]  # get the layer feat from the t
    opt.n_t, opt.unique_t_shapes = unique_shape(
        opt.t_shapes)  # n_t is the vector for s model, unique_t_shapes is the s_shapes

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False

    # initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions

    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)
        criterionARGD = ARGD(opt)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)
        criterionARGD = ARGD(opt)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    print('testing the initialize models......')
    criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT, 'criterionARGD': criterionARGD}
    acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, 0)
    print("clean acc:", acc_clean)
    print("backdoor acc:", acc_bad)
    for epoch in range(0, opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt.lr)
        # train every epoch
        train_step(opt, train_loader, nets, optimizer, criterions, epoch + 1)
        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1)

        # remember best precision and save checkpoint
        # save_root = opt.checkpoint_root + '/' + opt.s_name
        if opt.save:
            is_best = acc_clean[0] > opt.threshold_clean
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, opt.s_name)


def main():
    # Prepare arguments

    opt = get_arguments().parse_args()
    starttime = datetime.datetime.now()
    train(opt)
    endtime = datetime.datetime.now()
    print(endtime - starttime)

if (__name__ == '__main__'):
    main()
