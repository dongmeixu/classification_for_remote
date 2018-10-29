import datetime
import random
import threading

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
from collections import Iterator

from sklearn.preprocessing import OneHotEncoder

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
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
img_channel = 4   # RGB+NDVI
# n_classes = 21
n_classes = 2

# base_dir = '/media/files/xdm/classification/'
# # model_dir = base_dir + 'weights/UCMerced_LandUse/'
# model_dir = base_dir + 'weights/new_10_classes/'

base_dir = '/search/odin/xudongmei/'
model_dir = base_dir + 'weights/new_10_classes/'

# 定义训练集以及验证集的路径
train_data_dir = base_dir + 'data/2cls_256_4channels/train'
val_data_dir = base_dir + 'data/2cls_256_4channels/val'
test_data_dir = base_dir + 'data/2cls_256_4channels/test'

# # 共21类(影像中所有地物的名称)
ObjectNames = ["zhibei", "no_zhibei"]


def get_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu")(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="sigmoid")(x)

    model = Model(input, x)


    return model


class LossHistory(keras.callbacks.Callback):
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
        plt.savefig(model_dir + "2cls_256_NDVI_4channel{}_{}.png".format(batch_size, nbr_epochs))


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, img_width, img_height, img_channel))
    y_batch = np.zeros((batch_size, img_width, img_height, n_classes))
    X_height = X.shape[1]
    X_width = X.shape[2]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_width - 1)
        random_height = random.randint(0, X_height - img_height - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, random_height: random_height + img_width, random_width: random_width + img_height, :]
        X_batch[i] = np.array(
            X[random_image, random_height: random_height + img_width, random_width: random_width + img_height, :])
    return X_batch, y_batch


class threadsafe_iter(Iterator):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        random_batch = random.randint(0, X.shape[0] - batch_size)
        X_batch, y_batch = X[random_batch: batch_size + random_batch, :, :, :], y[random_batch: batch_size + random_batch, :]
        # print("X_batch.shape",X_batch.shape)#X_batch.shape (128, 16, 112, 112)  tensorflow  X_batch.shape (16, 112, 112, 3)
        # print("y_batch.shape", y_batch.shape)#y_batch.shape (128, 1, 112, 112)  tensorflow  y_batch.shape (16, 112, 112, 1)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]
            # print("xb.shape", xb.shape)#xb.shape (16, 112, 112)  tensorflow xb.shape (112, 112, 3)
            # print("yb.shape", yb.shape)#yb.shape (1, 112, 112)    tensorflow   yb.shape (112, 112, 1)
            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = np.transpose(xb, (2, 0, 1))
                    xb = flip_axis(xb, 1)
                    xb = np.transpose(xb, (1, 2, 0))

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = np.transpose(xb, (2, 0, 1))
                    xb = flip_axis(xb, 2)
                    xb = np.transpose(xb, (1, 2, 0))

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(0, 1)

            X_batch[i] = xb
            y_batch[i] = yb
        # print("yield")
        yield X_batch, y_batch # y_batch((128, 1, 80, 80))


if __name__ == '__main__':
    begin = datetime.datetime.now()
    lenet_model = get_net(input_shape=(img_width, img_height, img_channel))
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    lenet_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    # 创建一个实例LossHistory
    history = LossHistory()

    # autosave best Model
    best_model_file = model_dir + "2cls_256_NDVI4ch_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

    # 获取数据
    x_train = np.zeros(shape=(nbr_train_samples, img_height, img_width, img_channel))
    y_train = np.zeros(shape=(nbr_train_samples, 1))

    train_lists = os.listdir(train_data_dir)
    i = 0

    import sklearn.preprocessing.label
    for label, dir in enumerate(train_lists):
        img_list = os.path.join(train_data_dir, dir)
        for img in img_list:
            x_train[i, :, :, :] = cv2.imread(os.path.join(train_data_dir, dir, img))
            y_train[i, :] = label
            i += 1
    # y_train = keras.utils.to_categorical(y_train, n_classes)
    enc = OneHotEncoder()
    enc.fit(y_train)

    # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    y_train = enc.transform(y_train).toarray()
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    x_val = np.zeros(shape=(nbr_validation_samples, img_height, img_width, img_channel))
    y_val = np.zeros(shape=(nbr_validation_samples, 1))

    val_lists = os.listdir(val_data_dir)
    i = 0

    for lable, dir in enumerate(val_lists):
        img_list = os.path.join(val_data_dir, dir)
        print(lable)
        for img in img_list:
            x_val[i, :, :, :] = cv2.imread(os.path.join(val_data_dir, dir, img))
            y_val[i, :] = lable
            i += 1
    # y_val = keras.utils.to_categorical(y_val, n_classes)
    enc = OneHotEncoder()
    enc.fit(y_val)

    # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    y_val = enc.transform(y_val).toarray()
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    lenet_model.fit_generator(
        batch_generator(x_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
        nb_epoch=nbr_epochs,
        samples_per_epoch=nbr_train_samples // batch_size,
        validation_data=batch_generator(x_val, y_val, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
        validation_steps=nbr_validation_samples // batch_size,
        callbacks=[history, early_stop],
        nb_worker=8
    )

    lenet_model.save_weights(best_model_file)

    history.loss_plot('epoch')
    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("Total train time: ", end - begin)

    lenet_model.load_weights(best_model_file)
    x_test = np.zeros(shape=(nbr_validation_samples, img_height, img_width, img_channel))
    y_test = np.zeros(shape=(nbr_validation_samples, 1))

    test_lists = os.listdir(test_data_dir)
    i = 0

    for lable, dir in enumerate(test_lists):
        img_list = os.path.join(test_data_dir, dir)
        for img in img_list:
            x_test[i, :, :, :] = cv2.imread(os.path.join(test_data_dir, dir, img))
            y_test[i, :] = lable
            i += 1
    # y_val = keras.utils.to_categorical(y_val, n_classes)
    enc = OneHotEncoder()
    enc.fit(y_test)

    # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    y = enc.transform(y_test).toarray()
    print("x_test: ", x_test.shape)
    print("y_test: ", y.shape)
    print(lenet_model.evaluate(x_test, y))

    #
    # # test data generator for prediction
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    #
    # test_generator = test_datagen.flow_from_directory(
    #     test_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     shuffle=False,  # Important !!!
    #     classes=ObjectNames,
    #     class_mode='categorical')
    #
    # test_image_list = test_generator.filenames
    # print('Loading model and weights from training process ...')
    #
    # print('Begin to predict for testing data ...')
    # preds = lenet_model.predict_generator(test_generator, 353)
    # print(preds)
    # predictions = lenet_model.predict_generator(test_generator, steps=batch_size)
    # print(lenet_model.metrics_names)  # ['loss', 'acc']
    #
    # test_image_classes = test_generator.classes
    # # test_image_list.reshape(-1, 1)
    # # np.expand_dims(test_image_list, -1)
    # # print(test_image_list.shape)
    # labels = []
    # for i in test_image_classes:
    #     labels.append(i)
    #
    # train_labels = []
    # train_image_classes = train_generator.classes
    # for i in train_image_classes:
    #     train_labels.append(i)
    # train_preds = lenet_model.predict_generator(train_generator, 2835)
    # print(lenet_model.evaluate_generator(test_generator, batch_size),
    #       lenet_model.evaluate_generator(train_generator, batch_size))
    # train_ypre = []
    # for i, pre in enumerate(train_preds):
    #     train_ypre.append(pre.argmax())
    #
    # y_pre = []
    # for i, pre in enumerate(predictions):
    #     y_pre.append(pre.argmax())
    #
    # print("The Confusion Matrix:")
    #
    # # -*-coding:utf-8-*-
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    train_preds = lenet_model.predict(x_test)
    train_ypre = []
    for i, pre in enumerate(train_preds):
        train_ypre.append(pre.argmax())

    # y_true代表真实的label值 y_pred代表预测得到的lavel值
    y_true = y_test
    y_pred = train_ypre
    print(y_true)
    print("---")
    print(y_pred)
    tick_marks = np.array(range(len([0, 1]))) + 0.5


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
    plt.savefig('confusion_matrix_2cls_256_NDVI4cj.png', format='png')
    # plt.show()
