#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2

"""
2016年的原图是4通道的，而且蓝绿通道顺序反了

处理图像：将蓝绿通道交换顺序，并去掉红外波段

"""


# def RGBA_RGB(path, save_path):
#     fileList = os.listdir(path)
#     for file in fileList:
#         image = tiff.imread(os.path.join(path, file))
#         # 交换通道
#         image = image[:, :, (2, 1, 0)]
#         # image = image.resize((1000, 1000))
#         print(image.shape)
#         tiff.imsave(save_path + file, image)


# 只保留后缀是.TIF的图像
def remove_file(path):
    fileList = os.listdir(path)
    for file in fileList:
        pathname = os.path.splitext(os.path.join(path, file))
        if pathname[1] != ".TIF":
            os.remove(os.path.join(path, file))


img_size = 256


# sys.path.append("/search/odin/yangyuran/program/Anaconda3/envs/tensorflow/lib/python3.6/site-packages/")
# TODO:每次分割样本都是一样的，需要不一样嘛
def split_train_val_test(root_total, root_train, root_val, root_test, val_ratio, test_ratio):
    """
    将样本分为训练集、验证集、测试集
    :param root_total 原始数据地址
    :param root_train 训练集保存地址
    :param root_val  验证集保存地址
    :param root_test  测试集保存地址
    :param val_ratio  验证集所占比例
    :param test_ratio  测试集所占比例
    """
    np.random.seed(2018)
    if not os.path.exists(root_train):
        os.mkdir(root_train)

    if not os.path.exists(root_val):
        os.mkdir(root_val)

    if not os.path.exists(root_test):
        os.mkdir(root_test)

    print(os.listdir(root_total))
    Names = os.listdir(root_total)  # 每个文件夹代表一类

    nbr_train_samples = 0
    nbr_val_samples = 0
    nbr_test_samples = 0

    for name in Names:
        # 如果该文件夹不存在，则创建
        if name not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, name))

        if name not in os.listdir(root_val):
            os.mkdir(os.path.join(root_val, name))

        if name not in os.listdir(root_test):
            os.mkdir(os.path.join(root_test, name))

        total_images = os.listdir(os.path.join(root_total, name))

        nbr_val = int(len(total_images) * val_ratio)
        nbr_test = int(len(total_images) * test_ratio)
        nbr_train = int(len(total_images) - nbr_val - nbr_test)

        # 数据打乱顺序
        np.random.shuffle(total_images)

        train_images = total_images[:nbr_train]
        val_images = total_images[nbr_train: nbr_train + nbr_val]
        test_images = total_images[nbr_train + nbr_val:]

        tmp_train = 0
        for img in train_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_train, name, img)
            # shutil.copy(source, target)
            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            # image_resize = cv2.resize(image, (img_size, img_size))
            image_resize = image
            image_resize = (image_resize[:, :, 3] - image_resize[:, :, 0]) / (
                        image_resize[:, :, 3] + image_resize[:, :, 0] + 0.1)
            cv2.imwrite(target, image_resize)
            # np.save(target, image_resize)
            tmp_train += 1
            nbr_train_samples += 1

        tmp_val = 0
        for img in val_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_val, name, img)
            # shutil.copy(source, target)
            # image_resize = resize(source, img_size)

            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            # image_resize = cv2.resize(image, (img_size, img_size))
            image_resize = image
            image_resize = (image_resize[:, :, 3] - image_resize[:, :, 0]) / (
                        image_resize[:, :, 3] + image_resize[:, :, 0] + 0.1)
            cv2.imwrite(target, image_resize)
            # np.save(target, image_resize)
            tmp_val += 1
            nbr_val_samples += 1

        tmp_test = 0
        for img in test_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_test, name, img)
            # shutil.copy(source, target)
            # image_resize = resize(source, img_size)
            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            # image_resize = cv2.resize(image, (img_size, img_size))
            image_resize = image
            image_resize[:, :, 3] = (image_resize[:, :, 3] - image_resize[:, :, 0]) / (
                        image_resize[:, :, 3] + image_resize[:, :, 0] + 0.1)
            cv2.imwrite(target, image_resize)
            # np.save(target, image_resize)
            tmp_test += 1
            nbr_test_samples += 1

        print(name + " : " + ' # training samples: {}, # validation samples: {}, # test samples: {}'
              .format(tmp_train, tmp_val, tmp_test))

    print('Finish splitting train and test images!')
    print('# training samples: {}, # validation samples: {}, # test samples: {}'
          .format(nbr_train_samples, nbr_val_samples, nbr_test_samples))


def resize(source, img_size):
    # image = Image.open(source)
    # image_resize = image.resize((img_size, img_size), PIL.Image.BILINEAR)
    # print(image_resize.size)
    #
    # image_resize.save("1.tif")
    image = cv2.imread(source)
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    image_resize = cv2.resize(image, (img_size, img_size))
    return image_resize


# total = r'F:\remote_sensing\2017_10\2cls'
# train = r'F:\remote_sensing\2017_10\NDVI\train'
# val = r'F:\remote_sensing\2017_10\NDVI\val'
# test = r'F:\remote_sensing\2017_10\NDVI\test'

# total = "/media/files/xdm/classification/data/process_1"
# train = "/media/files/xdm/classification/data/process_imgsize400/train"
# val = "/media/files/xdm/classification/data/process_imgsize400/val"
# test = "/media/files/xdm/classification/data/process_imgsize400/test"

# total = "/search/odin/xudongmei/data/process_channel_change"
# train = "/search/odin/xudongmei/data/process_imgsize400/train"
# val = "/search/odin/xudongmei/data/process_imgsize400/val"
# test = "/search/odin/xudongmei/data/process_imgsize400/test"

total = "/search/odin/xudongmei/data/2cls"
train = "/search/odin/xudongmei/data/2cls_huafen/train"
val = "/search/odin/xudongmei/data/2cls_huafen/val"
test = "/search/odin/xudongmei/data/2cls_huafen/test"

split_train_val_test(total, train, val, test, 0.1, 0.1)
# image = tiff.imread(r"F:\remote_sensing\201702.TIF")
# print(image.shape)
