from __future__ import print_function
from __future__ import absolute_import

import keras
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

nbr_train_samples = 2843
nbr_validation_samples = 349

nbr_epochs = 800
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 10

# base_dir = '/media/files/xdm/classification/'
# # model_dir = base_dir + 'weights/UCMerced_LandUse/'
# model_dir = base_dir + 'weights/new_10_classes/'

base_dir = '/search/odin/xudongmei/'
model_dir = base_dir + 'weights/new_10_classes/'

# 定义训练集以及验证集的路径
# train_data_dir = base_dir + 'data/UCMerced_LandUse/train_split'
# val_data_dir = base_dir + 'data/UCMerced_LandUse/val_split'
# train_data_dir = base_dir + 'data/train_split'
# val_data_dir = base_dir + 'data/test_split'
train_data_dir = base_dir + 'data/process_imgsize256/train'
val_data_dir = base_dir + 'data/process_imgsize256/val'
test_data_dir = base_dir + 'data/process_imgsize256/test'

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
#
# WEIGHTS_PATH = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# WEIGHTS_PATH_NO_TOP = '/media/files/xdm/classification/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH = '/search/odin/xudongmei/working/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/search/odin/xudongmei/working/pre_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.1), axis=-1)


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
        plt.savefig(model_dir + "vgg16_10_cls_256_pre_{}_{}.png".format(batch_size, nbr_epochs))


if __name__ == '__main__':
    print('Loading VGG16 Weights ...')
    VGG16_notop = VGG16(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape=(img_width, img_height, img_channel))
    VGG16_notop.summary()

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = VGG16_notop.get_layer(index=-1).output  # Shape: (6, 6, 2048)
    output = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(output)
    # output = Flatten(name='flatten')(output)
    # output = Dense(n_classes, activation='softmax', name='predictions')(output)
    output = output = Conv2D(kernel_size=(1, 1), filters=n_classes, activation='softmax', name='predictions')(output)
    output = Flatten(name='flatten')(output)
    VGG16_model = Model(VGG16_notop.input, output)
    VGG16_model.summary()

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    VGG16_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    # 创建一个实例LossHistory
    history = LossHistory()

    # autosave best Model
    # best_model_file = model_dir + "VGG16_UCM_weights.h5"
    # best_model_file = model_dir + "RVGG16_2015_4_classes_weights.h5"
    best_model_file = model_dir + "VGG16_10_cls_256_weights.h5"
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

    # Model visualization
    # from keras.utils.vis_utils import plot_model

    # plot_model(VGG16_model, to_file=model_dir + 'RVGG16_UCM_{}_{}.png'.format(batch_size, nbr_epochs), show_shapes=True)
    # plot_model(VGG16_model, to_file=model_dir + 'RVGG16_10_cls_400_model.png', show_shapes=True)

    # H = VGG16_model.fit_generator(
    #     train_generator,
    #     samples_per_epoch=nbr_train_samples,
    #     nb_epoch=nbr_epochs,
    #     validation_data=validation_generator,
    #     nb_val_samples=nbr_validation_samples,
    #     callbacks=[history, early_stop]
    # )
    # VGG16_model.save_weights(best_model_file)

    # VGG16_model.save_weights(model_dir + 'my_10_cls_128_weights_pre.h5')

    # plot the training loss and accuracy
    # plt.figure()
    # N = nbr_epochs
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    #
    # plt.title("Training Loss and Accuracy on Satellite")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # 存储图像，注意，必须在show之前savefig，否则存储的图片一片空白
    # plt.savefig(model_dir + "VGG16_UCM_{}_{}.png".format(batch_size, nbr_epochs))
    # plt.savefig(model_dir + "RVGG16_10_cls_128_pre_{}_{}.png".format(batch_size, nbr_epochs))
    # # plt.show()
    # history.loss_plot('epoch')
    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("Total train time: ", end - begin)

    VGG16_model.load_weights(best_model_file)

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
    preds = VGG16_model.predict_generator(test_generator, 349)
    # print(preds)
    predictions = VGG16_model.predict_generator(test_generator, steps=batch_size)
    print(VGG16_model.metrics_names)  # ['loss', 'acc']
    # print(predictions)  # [1.0047961547970772, 0.6640625]

    # np.savetxt(os.path.join('predictions_pre200.txt'), predictions)

    print('Begin to write submission file ..')
    f_submit = open(os.path.join('submitVGG16.csv'), 'w')
    # f_submit.write('image,01_gengdi,02_yuandi,03_lindi,04_caodi,05_fangwujianzhu,06_road,07_gouzhuwu,08_rengong,09_huangmo,10_water\n')

    f_submit.write('image,pre\n')
    for i, image_name in enumerate(test_image_list):
        print(np.array(predictions).shape)
        pred = ['%.6f' % p for p in predictions[i, :]]
        pred = np.argmax(pred)
        print(pred)
        if i % 100 == 0:
            print('{} / {}'.format(i, 349))
        f_submit.write('%s,%d\n' % (os.path.dirname(image_name), pred))

    f_submit.close()
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import OneHotEncoder

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
    train_preds = VGG16_model.predict_generator(train_generator, 2843)
    print(VGG16_model.evaluate_generator(test_generator, batch_size))
    train_ypre = []
    for i, pre in enumerate(train_preds):
        train_ypre.append(pre.argmax())

    # one_hot = OneHotEncoder()
    # one_hot.fit(labels)
    # labels = one_hot.transform(labels)
    print(predictions.shape)
    y_pre = []
    for i, pre in enumerate(predictions):
        y_pre.append(pre.argmax())

    print("The Confusion Matrix:")
    print(confusion_matrix(labels, y_pre))

    # -*-coding:utf-8-*-
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    # y_true代表真实的label值 y_pred代表预测得到的lavel值
    # y_true = train_labels
    # y_pred = train_ypre
    y_true = labels
    y_pred = y_pre

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
    # cm = confusion_matrix(y_true, y_pred)
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

    plot_confusion_matrix(cm, title='VGG16-confusion matrix')
    # show confusion matrix
    plt.savefig('confusion_matrix_VGG16.png', format='png')
    # plt.show()
