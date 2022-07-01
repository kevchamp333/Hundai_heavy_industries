import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math
import numpy as np
import cv2 as cv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        h = img.size(1)
        w = img.size(2)
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)
        mask = np.ones((hh, hh), np.float32)

        st_h = np.random.randint(d)   # delta x
        st_w = np.random.randint(d)  # delta y

        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0

        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        mask = mask.cuda()

        # mask_new = torch.zeros(3, mask.shape[0], mask.shape[1])
        # for i in range(3):
        #     mask_new[i] = mask
        #
        # mask_new = mask_new.view(mask_new.shape[1], mask_new.shape[2], 3)
        # mask_new = mask_new.expand_as(img)
        #mask = mask.expand_as(img)
        #mask_new = mask_new.cuda()

        img = img * mask

        return img

class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x

        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y

# import glob
# import cv2
#                                   #배치                      가로    세로  채널
# array =np.zeros((len(glob.glob("*.png")), 32,  100,  3 ), dtype=np.uint8)
#
# for idx,value in enumerate(glob.glob("*.png")) :
#     img = cv2.imread(value)
#     resize_img = cv.resize(img, (100, 32))
#     array[idx] = resize_img
#
# # cv.imshow("Title_color", array[1])
# # cv.waitKey()
# # cv.destroyAllWindows()
#
# grid = GridMask(args.d1, args.d2, args.rotate, args.ratio, args.mode, args.prob)
# array=torch.Tensor(array).to(device)
# input = grid(array)
#
# # grid mask 적용 이미지 보기
# input = input.cpu().numpy().astype(np.uint8)
# cv.imshow("Title_color", input[0])
# cv.waitKey()
# cv.destroyAllWindows()
# image_pixel = cv.imread("000419_1_deg_r90_bot_dst.png")
# resize_img = cv.resize(image_pixel, (300, 200))

