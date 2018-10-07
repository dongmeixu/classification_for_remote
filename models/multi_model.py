"""

将vgg16与xception进行模型融合

"""
from __future__ import print_function
from __future__ import absolute_import

import datetime
import os

from keras import Input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import get_source_inputs
from keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from override_image import ImageDataGenerator
import numpy as np
import matplotlib
from keras.utils import layer_utils, plot_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from VGG16 import VGG16
from Xception import Xception

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义超参数
learning_rate = 0.0001
img_width = 200
img_height = 200
# nbr_train_samples = 1672
# nbr_validation_samples = 419
nbr_train_samples = 2829
nbr_validation_samples = 712
nbr_epochs = 800
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 10

base_dir = '/media/files/xdm/classification/'
# model_dir = base_dir + 'weights/UCMerced_LandUse/'
model_dir = base_dir + 'weights/new_10_classes/'

# 定义训练集以及验证集的路径
# train_data_dir = base_dir + 'data/UCMerced_LandUse/train_split'
# val_data_dir = base_dir + 'data/UCMerced_LandUse/val_split'
train_data_dir = base_dir + 'data/train_split'
val_data_dir = base_dir + 'data/test_split'

# # 共21类(影像中所有地物的名称)
# ObjectNames = ['agricultural', 'airplane', 'baseballdiamond', 'beach',
#                'buildings', 'chaparral', 'denseresidential', 'forest',
#                'freeway', 'golfcourse', 'harbor', 'intersection',
#                'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot',
#                'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
#                ]

# 共4类(影像中所有地物的名称)
# ObjectNames = ['building', 'other', 'water', 'zhibei']

ObjectNames = ['01_gengdi', '02_yuandi', '03_lindi', '04_caodi', '05_fangwujianzhu',
               '06_road', '07_gouzhuwu', '08_rengong', '09_huangmo', "10_water"]

WEIGHTS_PATH = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Multimodel(cnn_weights_path=None, all_weights_path=None, class_num=5, cnn_no_vary=False):
    """
    获取densent121,xinception并联的网络
    此处的cnn_weights_path是个列表是densenet和xception的卷积部分的权值
    """
    input_layer = Input(shape=(img_width, img_height, img_channel))

    vgg_notop = VGG16(include_top=False, weights='imagenet',
                      input_tensor=None, input_shape=(img_width, img_height, img_channel))

    xception_notop = Xception(include_top=False, weights='imagenet',
                              input_tensor=None, input_shape=(img_width, img_height, img_channel))

    # res=ResNet50(include_top=False,weights=None,input_shape=(224,224,3))

    if cnn_no_vary:
        for i, layer in enumerate(vgg_notop.layers):
            vgg_notop.layers[i].trainable = False
        for i, layer in enumerate(xception_notop.layers):
            xception_notop.layers[i].trainable = False
            # for i,layer in enumerate(res.layers):
        #   res.layers[i].trainable=False

    if cnn_weights_path is not None:
        vgg_notop.load_weights(cnn_weights_path[0])
        xception_notop.load_weights(cnn_weights_path[1])
        # res.load_weights(cnn_weights_path[2])
    vgg = vgg_notop(input_layer)
    xception = xception_notop(input_layer)

    # 对dense_121和xception进行全局最大池化
    top1_model = GlobalMaxPooling2D(data_format='channels_last')(vgg)
    top2_model = GlobalMaxPooling2D(data_format='channels_last')(xception)
    # top3_model=GlobalMaxPool2D(input_shape=res.output_shape)(res.outputs[0])

    print(top1_model.shape, top2_model.shape)
    # 把top1_model和top2_model连接起来
    t = Concatenate(axis=1)([top1_model, top2_model])
    # 第一个全连接层
    top_model = Dense(units=512, activation="relu")(t)
    top_model = Dropout(rate=0.5)(top_model)
    top_model = Dense(units=class_num, activation="softmax")(top_model)

    model = Model(inputs=input_layer, outputs=top_model)

    # 加载全部的参数
    if all_weights_path:
        model.load_weights(all_weights_path)
    return model


if __name__ == "__main__":
    weights_path = ['/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    '/media/files/xdm/classification/pre_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5']
    model = Multimodel(cnn_weights_path=weights_path, class_num=4)
    # plot_model(model, to_file="model.png")

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # autosave best Model
    # best_model_file = model_dir + "VGG16_UCM_weights.h5"
    # best_model_file = model_dir + "VGG16_2015_4_classes_weights.h5"
    best_model_file = model_dir + "multiModel_2015_4_classes_weights.h5"
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
    plot_model(model, to_file=model_dir + 'multiModel_2015_4_classes_model.png', show_shapes=True)

    H = model.fit_generator(
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
    plt.savefig(model_dir + "multiModel_2015_4_classes_{}_{}.png".format(batch_size, nbr_epochs))
    # plt.show()

    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("总的训练时间为：", end - begin)
