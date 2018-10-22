import datetime

import keras
from keras import layers, Input
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import os

from keras.engine import get_source_inputs
from keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, warnings, Dropout
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from override_image import ImageDataGenerator
import numpy as np
import matplotlib
from keras.utils import layer_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义超参数
learning_rate = 0.0001
img_width = 256
img_height = 256

nbr_train_samples = 2835
nbr_validation_samples = 353

nbr_epochs = 800
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 2

# base_dir = '/media/files/xdm/classification/'
# # model_dir = base_dir + 'weights/UCMerced_LandUse/'
# model_dir = base_dir + 'weights/new_10_classes/'

base_dir = '/search/odin/xudongmei/'
model_dir = base_dir + 'weights/new_10_classes/'

# 定义训练集以及验证集的路径
train_data_dir = base_dir + 'data/NVDI_2cls_256/train'
val_data_dir = base_dir + 'data/NVDI_2cls_256/val'
test_data_dir = base_dir + 'data/NVDI_2cls_256/test'

# # 共21类(影像中所有地物的名称)
ObjectNames = ["zhibei", "no_zhibei"]


def get_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu")(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(2, activation="sigmoid")(x)

    model = Model(input, x)

    return model


class LossHistory(keras.callbacks.Callback):
    def __init__(self):

        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        # plt.show()
        plt.title("Training Loss and Accuracy on Satellite")
        plt.savefig(model_dir + "2cls_256_{}_{}.png".format(batch_size, nbr_epochs))


if __name__ == '__main__':
    lenet_model = get_net(input_shape=(img_width, img_height, img_channel))
    lenet_model.summary()
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    lenet_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    # 创建一个实例LossHistory
    history = LossHistory()

    # autosave best Model
    best_model_file = model_dir + "2cls_256_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

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

    H = lenet_model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        callbacks=[history, early_stop]
    )
    lenet_model.save_weights(best_model_file)

    history.loss_plot('epoch')
    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("Total train time: ", end - begin)

    lenet_model.load_weights(best_model_file)

    # test data generator for prediction
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,  # Important !!!
        classes=ObjectNames,
        class_mode='categorical')

    test_image_list = test_generator.filenames
    print('Loading model and weights from training process ...')

    print('Begin to predict for testing data ...')
    preds = lenet_model.predict_generator(test_generator, 353)
    print(preds)
    predictions = lenet_model.predict_generator(test_generator, steps=batch_size)
    print(lenet_model.metrics_names)  # ['loss', 'acc']

    test_image_classes = test_generator.classes
    # test_image_list.reshape(-1, 1)
    # np.expand_dims(test_image_list, -1)
    # print(test_image_list.shape)
    labels = []
    for i in test_image_classes:
        labels.append(i)

    train_labels = []
    train_image_classes = train_generator.classes
    for i in train_image_classes:
        train_labels.append(i)
    train_preds = lenet_model.predict_generator(train_generator, 2835)
    print(lenet_model.evaluate_generator(test_generator, batch_size),
          lenet_model.evaluate_generator(train_generator, batch_size))
    train_ypre = []
    for i, pre in enumerate(train_preds):
        train_ypre.append(pre.argmax())

    y_pre = []
    for i, pre in enumerate(predictions):
        y_pre.append(pre.argmax())

    print("The Confusion Matrix:")

    # -*-coding:utf-8-*-
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    # y_true代表真实的label值 y_pred代表预测得到的lavel值
    y_true = train_labels
    y_pred = train_ypre

    tick_marks = np.array(range(len(labels))) + 0.5


    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(ObjectNames)))
        plt.xticks(xlocations, ObjectNames, rotation=90)
        plt.yticks(xlocations, ObjectNames)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(ObjectNames))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('confusion_matrix_2cls_256.png', format='png')
    # plt.show()
