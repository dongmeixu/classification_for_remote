"""
模型主要用tf+keras实现。首先导入各种模块

"""
import datetime

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Embedding, Flatten
from keras.layers.merge import multiply
from Xception import Xception

from keras import Input
import os

from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import matplotlib
import keras.backend as K

from override_image import ImageDataGenerator

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 指定随机种子
np.random.seed(2018)
tf.set_random_seed(2018)

# 定义超参数
learning_rate = 0.0001
img_width = 256
img_height = 256
# nbr_train_samples = 1672
# nbr_validation_samples = 419
nbr_train_samples = 191
nbr_validation_samples = 51
nbr_epochs = 5000
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 4
feature_size = 3

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


if __name__ == '__main__':
    print('Loading Xception Weights ...')

    # 基础模型Xception,然后使用了GLU激活函数来压缩特征，最后接softmax分类，
    # 此外添加了center loss 和auxiliary loss(直连边)作为辅助，这两项可以看成是正则项
    input_image = Input(shape=(img_width, img_height, img_channel))
    base_model = Xception(include_top=False, weights=None,
                          input_tensor=input_image, classes=n_classes)

    for layer in base_model.layers:  # 冻结Xception的所有层
        layer.trainable = False

    dense = Dense(feature_size)(base_model.output)
    gate = Dense(feature_size, activation='sigmoid')(base_model.output)
    feature = multiply([dense, gate])  # 以上三步构成了所谓的GLU激活函数
    feature = Flatten()(feature)
    predict = Dense(n_classes, activation='softmax', name='softmax')(feature)  # 分类
    auxiliary = Dense(n_classes, activation='softmax', name='auxiliary')(base_model.output)  # 直连边分类

    input_target = Input(shape=(1,))
    centers = Embedding(n_classes, feature_size)(input_target)
    print(feature.shape)
    print(centers.shape)
    l2_loss = Lambda(lambda x: K.sum(
        K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2')([feature, centers])  # 定义center_loss

    model_1 = Model(inputs=[input_image, input_target], outputs=[predict, l2_loss, auxiliary])
    model_1.compile(optimizer='adam',
                    loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred,
                          'sparse_categorical_crossentropy'],
                    loss_weights=[1., 0.25, 0.25],
                    metrics={'softmax': 'accuracy', 'auxiliary': 'accuracy'})
    model_1.summary()  # 第一阶段的模型，用adam优化

    for i, layer in enumerate(model_1.layers):
        if 'block13' in layer.name:
            break

    for layer in model_1.layers[i:len(base_model.layers)]:  # 这两个循环结合，实现了放开两个block的参数
        layer.trainable = True

    sgd = SGD(lr=1e-4, momentum=0.9)  # 定义低学习率的SGD优化器
    model_2 = Model(inputs=[input_image, input_target], outputs=[predict, l2_loss, auxiliary])
    model_2.compile(optimizer=sgd,
                    loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred,
                          'sparse_categorical_crossentropy'],
                    loss_weights=[1., 0.25, 0.25],
                    metrics={'softmax': 'accuracy', 'auxiliary': 'accuracy'})
    model_2.summary()  # 第二阶段的模型，用sgd优化

    model = Model(inputs=input_image, outputs=[predict, auxiliary])  # 用来预测的模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # autosave best Model
    # best_model_file = model_dir + "Xception_UCM_weights.h5"
    best_model_file = model_dir + "Xception_2015_4_classes_weights_with_multiLoss.h5"
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

    plot_model(model_1, to_file=model_dir + 'Xception_2015_4classes_model_{}_{}_with_multiLoss.png'.format(batch_size, nbr_epochs),
               show_shapes=True)

    H = model_1.fit_generator(
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
    plt.savefig(model_dir + "Xception_2014_classes_{}_{}_with_multiLoss.png".format(batch_size, nbr_epochs))
    # plt.show()

    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("总的训练时间为：", end - begin)
