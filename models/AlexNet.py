import datetime

import keras
from keras import layers, Input
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import os

from keras.engine import get_source_inputs
from keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, warnings, Dropout
from keras.models import Model, Sequential
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

# # 共21类(影像中所有地物的名称)
# ObjectNames = ['building', 'other', 'water', 'zhibei']
ObjectNames = ['01_gengdi', '02_yuandi', '03_lindi', '04_caodi', '05_fangwujianzhu',
               '06_road', '07_gouzhuwu', '08_rengong', '09_huangmo', "10_water"]

def get_net(input_shape):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
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
        plt.savefig(model_dir + "alexnet_10_cls_256_pre_{}_{}.png".format(batch_size, nbr_epochs))


if __name__ == '__main__':
    print('Loading VGG16 Weights ...')
    lenet_model = get_net(input_shape=(img_width, img_height, img_channel))
    lenet_model.summary()

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    lenet_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    # 创建一个实例LossHistory
    history = LossHistory()

    # autosave best Model
    best_model_file = model_dir + "alexnet_10_cls_256_weights.h5"
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
    preds = lenet_model.predict_generator(test_generator, 349)
    print(preds)
    predictions = lenet_model.predict_generator(test_generator, steps=batch_size)
    print(lenet_model.metrics_names)  # ['loss', 'acc']
    # print(predictions)  # [1.0047961547970772, 0.6640625]

    # np.savetxt(os.path.join('predictions_pre200.txt'), predictions)

    # print('Begin to write submission file ..')
    # f_submit = open(os.path.join('submit_pre200.csv'), 'w')
    # f_submit.write('image,01_gengdi,02_yuandi,03_lindi,04_caodi,05_fangwujianzhu,06_road,07_gouzhuwu,08_rengong,09_huangmo,10_water\n')
    # for i, image_name in enumerate(test_image_list):
    #     print(np.array(predictions).shape)
    #     pred = ['%.6f' % p for p in predictions[i, :]]
    #     if i % 100 == 0:
    #         print('{} / {}'.format(i, 349))
    #     f_submit.write('%s,%s\n' % (os.path.dirname(image_name), ','.join(pred)))
    #
    # f_submit.close()
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
    train_preds = lenet_model.predict_generator(train_generator, 2843)
    print(lenet_model.evaluate_generator(test_generator, batch_size),
          lenet_model.evaluate_generator(train_generator, batch_size))
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
    plt.savefig('confusion_matrix_alexnet_256.png', format='png')
    # plt.show()
