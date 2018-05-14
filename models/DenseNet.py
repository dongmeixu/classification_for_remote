# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import keras.backend as K
import warnings

import datetime
import os

from keras import Input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import get_source_inputs
from keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from override_image import ImageDataGenerator
import numpy as np
import matplotlib
from keras.utils import layer_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 定义超参数
learning_rate = 0.0001
img_width = 256
img_height = 256
# nbr_train_samples = 1672
# nbr_validation_samples = 419
nbr_train_samples = 191
nbr_validation_samples = 51
nbr_epochs = 800
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 4

base_dir = '/media/files/xdm/classification/'
# model_dir = base_dir + 'weights/UCMerced_LandUse/'
model_dir = base_dir + 'weights/2015_4_classes/'


# 定义训练集以及验证集的路径
# train_data_dir = base_dir + 'data/UCMerced_LandUse/train_split'
# val_data_dir = base_dir + 'data/UCMerced_LandUse/val_split'
train_data_dir = base_dir + 'data/2015_4_classes/aug_256/train_split'
val_data_dir = base_dir + 'data/2015_4_classes/aug_256/val_split'

# # 共21类(影像中所有地物的名称)
# ObjectNames = ['agricultural', 'airplane', 'baseballdiamond', 'beach',
#                'buildings', 'chaparral', 'denseresidential', 'forest',
#                'freeway', 'golfcourse', 'harbor', 'intersection',
#                'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot',
#                'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
#                ]

# 共21类(影像中所有地物的名称)
ObjectNames = ['building', 'other', 'water', 'zhibei']


WEIGHTS_PATH = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


"""
This DenseNet implementation comes from:
    https://github.com/titu1994/DenseNet/blob/master/densenet_fast.py
I've made some modifications so as to make it consistent with Keras2 interface
"""

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
    Args:
        input: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added
    """

    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
    Args:
        input: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """ Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                   weight_decay=1E-4, verbose=True):
    """ Build the create_dense_net model
    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay
    Returns: keras tensor with nb_layers of conv_block appended
    """

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D",
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet


if __name__ == '__main__':
    densenet = createDenseNet(n_classes, (img_width, img_height, img_channel))
    densenet.summary()

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # autosave best Model
    # best_model_file = model_dir + "VGG16_UCM_weights.h5"
    # best_model_file = model_dir + "VGG16_2015_4_classes_weights.h5"
    best_model_file = model_dir + "densenet_2015_4_classes_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        samplewise_center=True,  # 输入数据集去中心化，按feature执行
        rescale=1. / 255,  # 重缩放因子
        shear_range=0.1,  # 剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.1,  # 随机缩放的幅度
        rotation_range=10.,  # 图片随机转动的角度
        width_shift_range=0.1,  # 图片水平偏移的幅度
        height_shift_range=0.1,  # 图片竖直偏移的幅度
        horizontal_flip=True,  # 进行随机水平翻转
        vertical_flip=True,  # 进行随机竖直翻转
    )

    # this is the augmentation configuration we will use for validation:
    # only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes=ObjectNames,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        # save_prefix = 'aug',
        classes=ObjectNames,
        class_mode='categorical')

    begin = datetime.datetime.now()
    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    # Model visualization
    from keras.utils.vis_utils import plot_model

    # plot_model(VGG16_model, to_file=model_dir + 'VGG16_UCM_{}_{}.png'.format(batch_size, nbr_epochs), show_shapes=True)
    # plot_model(VGG16_model, to_file=model_dir + 'VGG16_2015_4_classes_model.png', show_shapes=True)
    plot_model(densenet, to_file=model_dir + 'densenet_2015_4_classes_model.png', show_shapes=True)

    H = densenet.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        callbacks=[best_model])

    # plot the training loss and accuracy
    plt.figure()
    N = nbr_epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy on Satellite")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # 存储图像，注意，必须在show之前savefig，否则存储的图片一片空白
    # plt.savefig(model_dir + "VGG16_UCM_{}_{}.png".format(batch_size, nbr_epochs))
    # plt.savefig(model_dir + "VGG16_2015_4_classes_{}_{}.png".format(batch_size, nbr_epochs))
    plt.savefig(model_dir + "densenet_2015_4_classes_{}_{}.png".format(batch_size, nbr_epochs))
    # plt.show()

    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("总的训练时间为：", end - begin)



