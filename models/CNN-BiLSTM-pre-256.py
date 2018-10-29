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
from four_dir_rnn import four_dir_rnn
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

nbr_train_samples = 2843
nbr_validation_samples = 349

nbr_epochs = 800
batch_size = 32
img_channel = 4   # RGB+NDVI
# n_classes = 21
n_classes = 10

# base_dir = '/media/files/xdm/classification/'
# # model_dir = base_dir + 'weights/UCMerced_LandUse/'
# model_dir = base_dir + 'weights/new_10_classes/'

base_dir = '/search/odin/xudongmei/'
model_dir = base_dir + 'weights/new_10_classes/'

# 定义训练集以及验证集的路径
train_data_dir = base_dir + 'data/10cls_256_4channels/train'
val_data_dir = base_dir + 'data/10cls_256_4channels/val'
test_data_dir = base_dir + 'data/10cls_256_4channels/test'

# ObjectNames = ['building', 'other', 'water', 'zhibei']
ObjectNames = ['01_gengdi', '02_yuandi', '03_lindi', '04_caodi', '05_fangwujianzhu', '06_road', '07_gouzhuwu',
               '08_rengong', '09_huangmo', '10_water']

# WEIGHTS_PATH = r'C:\Users\ASUS\Desktop\高分影像\pre_weights\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# WEIGHTS_PATH_NO_TOP = r'C:\Users\ASUS\Desktop\高分影像\pre_weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# WEIGHTS_PATH = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# WEIGHTS_PATH_NO_TOP = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

WEIGHTS_PATH = '/search/odin/xudongmei/working/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/search/odin/xudongmei/working/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""
预训练vgg16 + 2个四向rnn

"""


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP

        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

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
        plt.savefig(model_dir + "10cls_256_NDVI_4channel{}_{}.png".format(batch_size, nbr_epochs))


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
    VGG16_notop = VGG16(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape=(img_width, img_height, img_channel))
    # VGG16_notop.summary()

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = VGG16_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
    output = Conv2D(128, (1, 1), activation='relu', padding='same')(output)  # 作用是降维

    rnn_shape = K.int_shape(output)
    print(rnn_shape)  # (None, 8, 8, 512)
    output = four_dir_rnn(output, rnn_shape)

    # 2层rnn
    rnn_shape = K.int_shape(output)
    # 通道数为512时OOM
    print(rnn_shape)  # (None, 8, 8, 512)
    output = four_dir_rnn(output, rnn_shape)
    # output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(n_classes, activation='softmax', name='predictions')(output)

    VGG16_model = Model(VGG16_notop.input, output)
    VGG16_model.summary()

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    VGG16_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                        metrics=['accuracy', "top_k_categorical_accuracy"])
    # 创建一个实例LossHistory
    history = LossHistory()

    # autosave best Model
    # best_model_file = model_dir + "VGG16_UCM_weights.h5"
    # best_model_file = model_dir + "RVGG16_2015_4_classes_weights.h5"
    best_model_file = model_dir + "RVGG16_10_cls_256_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

    # 获取数据
    x_train = np.zeros(shape=(nbr_train_samples, img_height, img_width, img_channel))
    y_train = np.zeros(shape=(nbr_train_samples, 1))

    train_lists = os.listdir(train_data_dir)
    i = 0

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

    VGG16_model.fit_generator(
        batch_generator(x_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
        nb_epoch=nbr_epochs,
        samples_per_epoch=nbr_train_samples // batch_size,
        validation_data=batch_generator(x_val, y_val, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
        validation_steps=nbr_validation_samples // batch_size,
        callbacks=[history, early_stop],
        nb_worker=8
    )

    VGG16_model.save_weights(best_model_file)

    history.loss_plot('epoch')
    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("Total train time: ", end - begin)

    VGG16_model.load_weights(best_model_file)
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
    print(VGG16_model.evaluate(x_test, y))

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

    train_preds = VGG16_model.predict(x_test)
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
    plt.savefig('confusion_matrix_10cls_256_NDVI4cj.png', format='png')
    # plt.show()
