from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
import scipy.stats as st


def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if opt.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader


def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if opt.dataset == 'CIFAR10':
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=opt.batch_size,
                                   shuffle=False,
                                   )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()
                                   ])
    if opt.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train,
                               mode='train')
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    )

    return train_clean_loader


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=0.05)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"),
                 distance=3):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w,
                                       opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()
        data_random = dataset[2]
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':  # have the all class to one class

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    img_r = np.array(data_random[0])
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    img_r = np.array(data_random[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack have error is the  success
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    img_r = np.array(data_random[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    img_r = np.array(data_random[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    img_r = np.array(data_random[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    img_r = np.array(data_random[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            elif target_type == 'Refool':

                if mode == 'train':
                    img = np.array(data[0])
                    img_r = np.array(data_random[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:
                            img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    img_r = np.array(data_random[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, img_r, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, img_r, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'refoolTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'refoolTrigger':
            img, _, _ = self.blend_images(img, img_r)

        else:
            raise NotImplementedError

        return img

    # def _reflectionTrigger(self, img, width, height, distance):
    #     for j in range(width - distance - trig_w, width - distance):
    #         for k in range(height - distance - trig_h, height - distance):
    #             img[j, k] = 255.0
    #
    #     return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        # img[width - 1][height - 1] = 255
        # img[width - 1][height - 2] = 0
        # img[width - 1][height - 3] = 255
        #
        # img[width - 2][height - 1] = 0
        # img[width - 2][height - 2] = 255
        # img[width - 2][height - 3] = 0
        #
        # img[width - 3][height - 1] = 255
        # img[width - 3][height - 2] = 0
        # img[width - 3][height - 3] = 0

        # adptive center trigger
        alpha = 1
        img[width - 14][height - 14] = 255 * alpha
        img[width - 14][height - 13] = 128 * alpha
        img[width - 14][height - 12] = 255 * alpha

        img[width - 13][height - 14] = 128 * alpha
        img[width - 13][height - 13] = 255 * alpha
        img[width - 13][height - 12] = 128 * alpha

        img[width - 12][height - 14] = 255 * alpha
        img[width - 12][height - 13] = 128 * alpha
        img[width - 12][height - 12] = 128 * alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

    def blend_image(self, img_t, img_r):
        """
        Read image T from pwd_t and R from pwd_r
        return the blended image and precessed reflection image
        :param s_pwd_t:
        :param s_pwd_r:
        :return: T + R
        # """
        # img_t = cv2.imread(s_pwd_t)
        # img_r = cv2.imread(s_pwd_r)

        # h, w = img_t.shape[:2]
        # img_r = cv2.resize(img_r, (w, h))
        weight_t = np.mean(img_t)
        weight_r = np.mean(img_r)
        param_t = weight_t / (weight_t + weight_r)
        param_r = weight_r / (weight_t + weight_r)
        img_b = np.uint8(np.clip(param_t * img_t / 255. + param_r * img_r / 255., 0, 1) * 255)
        plt.imshow(img_b, interpolation='bilinear')
        plt.show()
        # cv2.waitKey()
        return img_b

    def blend_images(self, img_t, img_r, max_image_size=32, ghost_rate=0.39, alpha_t=-1., offset=(0, 0), sigma=-1,
                     ghost_alpha=-1.):
        """
        Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
        return the blended image and precessed reflection image
        """
        t = np.float32(img_t) / 255.
        r = np.float32(img_r) / 255.
        h, w, _ = t.shape
        # convert t.shape to max_image_size's limitation
        scale_ratio = float(max(h, w)) / float(max_image_size)
        w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
            else (int(round(w / scale_ratio)), max_image_size)
        t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

        if alpha_t < 0:
            alpha_t = 1. - random.uniform(0.05, 0.45)

        if random.randint(0, 100) < ghost_rate * 100:
            t = np.power(t, 2.2)
            r = np.power(r, 2.2)

            # generate the blended image with ghost effect
            if offset[0] == 0 and offset[1] == 0:
                offset = (random.randint(3, 8), random.randint(3, 8))
            r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                             'constant', constant_values=0)
            r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                             'constant', constant_values=(0, 0))
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if random.random() > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
            reflection_mask = ghost_r * (1 - alpha_t)

            blended = reflection_mask + t * alpha_t

            transmission_layer = np.power(t * alpha_t, 1 / 2.2)

            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.] = 1.
            ghost_r[ghost_r < 0.] = 0.

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.

            ghost_r = np.power(ghost_r, 1 / 2.2)
            ghost_r[blended > 1.] = 1.
            ghost_r[blended < 0.] = 0.

            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            # generate the blended image with focal blur
            if sigma < 0:
                sigma = random.uniform(1, 5)

            t = np.power(t, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            blend = r_blur + t

            # get the reflection layers' proper range
            att = 1.08 + np.random.random() / 10.0
            for i in range(3):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                # get normal distribution
                kern1d = np.diff(st.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel

            h, w = r_blur.shape[0: 2]
            new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
            new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

            g_mask = gen_kernel(max_image_size, 3)
            g_mask = np.dstack((g_mask, g_mask, g_mask))
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
            blend = r_blur_mask + t * alpha_t

            transmission_layer = np.power(t * alpha_t, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0

            blended = np.uint8(blend * 255)
            reflection_layer = np.uint8(r_blur_mask * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
            # plt.imshow(img_t, interpolation='bicubic')
            # plt.show()
            # plt.imshow(img_r, interpolation='bicubic')
            # plt.show()
            # plt.imshow(blended, interpolation='bicubic')
            # plt.show()
            # plt.imshow(transmission_layer, interpolation='bicubic')
            # plt.show()
            # plt.imshow(reflection_layer, interpolation='bicubic')
            # plt.show()

        return blended, transmission_layer, reflection_layer
