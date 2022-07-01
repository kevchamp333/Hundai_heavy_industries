import os
import sys
import re
import six
import math
import lmdb
import torch
from natsort import natsorted
import numpy as np
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image, ImageEnhance

class Batch_Balanced_Dataset(object):

    def __init__(self, opt, AlignCollate_):

        log = open(f'./saved_models/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')

        # print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        # log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')

        assert len(opt.select_data) == len(opt.batch_ratio)

        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0

        _batch_size = max(round(opt.batch_size * float(opt.batch_ratio[0])), 1)
        print(dashed_line)
        log.write(dashed_line + '\n')

        #_dataset= Make_custom_dataset(root=opt.train_data, label=opt.train_label, opt=opt)
        _dataset = Make_custom_dataset(data_1=opt.train_data_1, label_1 =opt.train_label_1,
                                       data_2=opt.train_data_2, label_2 =opt.train_label_2, data_3=opt.train_data_3, label_3 =opt.train_label_3, opt=opt) # train 1, train 2, train 3 합치기

        total_number_dataset = len(_dataset)
        #log.write(_dataset_log)

        """
        The total number of data can be modified with opt.total_data_usage_ratio.
        ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
        See 4.2 section in our paper.
        """
        number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
        dataset_split = [number_dataset, total_number_dataset - number_dataset]
        indices = range(total_number_dataset)
        _dataset, _ = [Subset(_dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(dataset_split), dataset_split)]
        selected_d_log = f'num total samples of {opt.select_data[0]}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
        selected_d_log += f'num samples of {opt.select_data[0]} per batch: {opt.batch_size} x {float(opt.batch_ratio[0])} (batch_ratio) = {_batch_size}'
        print(selected_d_log)
        log.write(selected_d_log + '\n')

        batch_size_list.append(str(_batch_size))
        Total_batch_size += _batch_size

        _data_loader = torch.utils.data.DataLoader(
            _dataset, batch_size=_batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_, pin_memory=True)

        len(_data_loader)
        self.data_loader_list.append(_data_loader)
        self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):

        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

# class Make_custom_dataset(Dataset):
#
#     def __init__(self, root, label, opt):
#
#         self.opt = opt
#         self.main_dir = root
#
#         self.total_imgs_list = []
#         self.total_label_list = []
#
#         # root 폴더 안에 있는 데이터 이름과 gt파일 이름이 같은 레이블 가져와야 함
#         image_list = open(label, "r")
#         image_list = image_list.readlines()
#
#         for i in image_list :
#             if '.png' in i  :
#                 if 'train' in i:
#                     i = i.split('/')[1]  # train 문자 제거
#                 self.total_imgs_list.append(i.split('\t')[0])
#                 self.total_label_list.append(i.split('\t')[1].split('\n')[0])
#
#     def __len__(self):
#         return len(self.total_imgs_list)
#
#     def __getitem__(self, idx):
#
#         img_loc = os.path.join(self.main_dir, self.total_imgs_list[idx])
#         label = self.total_label_list[idx]
#
#         # image = cv.imread(img_loc, 0)
#         image = Image.open(img_loc)
#
#         # '''contrast'''
#         contrast_enhancer = ImageEnhance.Contrast(image)
#         pil_enhanced_image = contrast_enhancer.enhance(2)
#         enhanced_image = np.asarray(pil_enhanced_image)
#         r, g, b = cv.split(enhanced_image)
#         #enhanced_image = cv.merge([b, g, r])
#         enhanced_image = cv.merge([r, g, b])
#         image_b = Image.fromarray(np.uint8(enhanced_image))
#
#         # plt.imshow(enhanced_image)
#         # plt.imshow(image)
#
#         # '''sharpness'''
#         # sharpness_enhancer = ImageEnhance.Sharpness(image_b)
#         # pil_enhanced_image = sharpness_enhancer.enhance(5.0)
#         # image_b = pil_enhanced_image
#         # enhanced_image = np.asarray(image_b)
#         #
#         # '''erode'''
#         # kernel = np.ones((2, 2), np.uint8)
#         # erode = cv.erode(enhanced_image, kernel, iterations=1)
#         # image_b = Image.fromarray(np.uint8(erode))
#
#         if self.opt.rgb:
#             image = image_b.convert('RGB')  # for color image
#         else:
#             image = image_b.convert('L')
#
#         if not self.opt.sensitive:
#             label = label.lower()
#
#         # plt.imshow(image)
#         # sys.exit()
#
#         return (image, label)
#
#     def get_file_name_list (self) :
#         return  self.total_imgs_list

class Make_custom_dataset(Dataset):

    def __init__(self, data_1, label_1, data_2, label_2, data_3, label_3, opt):

        self.opt = opt
        self.main_dir_1 = data_1
        self.main_dir_2 = data_2
        self.main_dir_3 = data_3

        self.total_imgs_list = []
        self.total_label_list = []

        self.image_1_count = 0
        self.image_2_count = 0
        self.image_3_count = 0

        # root 폴더 안에 있는 데이터 이름과 gt파일 이름이 같은 레이블 가져와야 함
        if (self.main_dir_1 != 'None') :
            image_list_1 = open(label_1, "r")
            image_list_1 = image_list_1.readlines()

            for i in image_list_1 :
                if '.png' in i  :
                    if 'train' in i:
                        i = i.split('/')[1]  # train 문자 제거
                    image = i.split('\t')[0]
                    label = i.split('\t')[1].split('\n')[0]
                    if (label != 'X') :
                        self.image_1_count += 1
                        self.total_imgs_list.append(image)
                        self.total_label_list.append(label)

        if (self.main_dir_2 != 'None'):
            image_list_2 = open(label_2, "r")
            image_list_2 = image_list_2.readlines()

            for i in image_list_2 :
                if '.png' in i  :
                    if 'train' in i:
                        i = i.split('/')[1]  # train 문자 제거
                    image = i.split('\t')[0]
                    label = i.split('\t')[1].split('\n')[0]
                    if (label != 'X') :
                        self.image_2_count += 1
                        self.total_imgs_list.append(image)
                        self.total_label_list.append(label)

        if (self.main_dir_3 != 'None'):
            image_list_3 = open(label_3, "r")
            image_list_3 = image_list_3.readlines()

            for i in image_list_3 :
                if '.png' in i  :
                    if 'train' in i:
                        i = i.split('/')[1]  # train 문자 제거
                    image = i.split('\t')[0]
                    label = i.split('\t')[1].split('\n')[0]
                    if (label != 'X') :
                        self.image_3_count += 1
                        self.total_imgs_list.append(image)
                        self.total_label_list.append(label)

    def __len__(self):
        return len(self.total_imgs_list)

    def __getitem__(self, idx):

        if(idx+1 <= self.image_1_count) :
            main_dir = self.main_dir_1
        elif(idx+1 <= self.image_1_count + self.image_2_count) :
            main_dir = self.main_dir_2
        elif (idx + 1 <= self.image_1_count + self.image_2_count + self.image_3_count):
            main_dir = self.main_dir_3

        img_loc = os.path.join(main_dir, self.total_imgs_list[idx])
        label = self.total_label_list[idx]

        # image = cv.imread(img_loc, 0)
        image = Image.open(img_loc)

        # '''contrast'''
        contrast_enhancer = ImageEnhance.Contrast(image)
        pil_enhanced_image = contrast_enhancer.enhance(2)
        enhanced_image = np.asarray(pil_enhanced_image)
        r, g, b = cv.split(enhanced_image)
        #enhanced_image = cv.merge([b, g, r])
        enhanced_image = cv.merge([r, g, b])
        image_b = Image.fromarray(np.uint8(enhanced_image))
        # plt.imshow(enhanced_image)
        # plt.imshow(image)

        # '''sharpness'''
        # sharpness_enhancer = ImageEnhance.Sharpness(image_b)
        # pil_enhanced_image = sharpness_enhancer.enhance(5.0)
        # image_b = pil_enhanced_image
        # enhanced_image = np.asarray(image_b)
        #
        # '''erode'''
        # kernel = np.ones((2, 2), np.uint8)
        # erode = cv.erode(enhanced_image, kernel, iterations=1)
        # image_b = Image.fromarray(np.uint8(erode))

        if self.opt.rgb:
            image = image_b.convert('RGB')  # for color image
        else:
            image = image_b.convert('L')

        # plt.imshow(image)
        # sys.exit()

        return (image, label)

    def get_file_name_list (self) :
        return  self.total_imgs_list


class Make_custom_dataset_test(Dataset):

    def __init__(self, data, label, opt):

        self.opt = opt
        self.main_dir = data

        self.total_imgs_list = []
        self.total_label_list = []

        image_list_1 = open(label, "r")
        image_list_1 = image_list_1.readlines()

        for i in image_list_1 :
            if '.png' in i  :
                if 'train' in i:
                    i = i.split('/')[1]  # train 문자 제거
                image = i.split('\t')[0]
                label = i.split('\t')[1].split('\n')[0]

                self.total_imgs_list.append(image)
                self.total_label_list.append(label)

    def __len__(self):
        return len(self.total_imgs_list)

    def __getitem__(self, idx):

        img_loc = os.path.join(self.main_dir, self.total_imgs_list[idx])
        label = self.total_label_list[idx]

        image = Image.open(img_loc)

        # # '''contrast'''
        # contrast_enhancer = ImageEnhance.Contrast(image)
        # pil_enhanced_image = contrast_enhancer.enhance(2)
        # enhanced_image = np.asarray(pil_enhanced_image)
        # r, g, b = cv.split(enhanced_image)
        # #enhanced_image = cv.merge([b, g, r])
        # enhanced_image = cv.merge([r, g, b])
        # image_b = Image.fromarray(np.uint8(enhanced_image))

        # plt.imshow(enhanced_image)
        # plt.imshow(image)

        # '''sharpness'''
        # sharpness_enhancer = ImageEnhance.Sharpness(image_b)
        # pil_enhanced_image = sharpness_enhancer.enhance(5.0)
        # image_b = pil_enhanced_image
        # enhanced_image = np.asarray(image_b)
        #
        # '''erode'''
        # kernel = np.ones((2, 2), np.uint8)
        # erode = cv.erode(enhanced_image, kernel, iterations=1)
        # image_b = Image.fromarray(np.uint8(erode))

        if self.opt.rgb:
            image = image.convert('RGB')  # for color image
        else:
            image = image.convert('L')

        if not self.opt.sensitive:
            label = label.lower()

        # plt.imshow(image)
        # sys.exit()

        return (image, label)

    def get_file_name_list (self) :
        return  self.total_imgs_list

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)