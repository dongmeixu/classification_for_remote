import keras
from keras.models import load_model
import os
from override_image import ImageDataGenerator
import numpy as np

img_width = 200
img_height = 200
batch_size = 16
nbr_test_samples = 349


# 共21类(影像中所有地物的名称)
# ObjectNames = ['building', 'other', 'water', 'zhibei']
ObjectNames = ['01_gengdi', '02_yuandi', '03_lindi', '04_caodi', '05_fangwujianzhu', '06_road', '07_gouzhuwu', '08_rengong', '09_huangmo', '10_water']

root_path = '/search/odin/xudongmei'

# weights_path = os.path.join(root_path, 'weights/2015_4_classes/VGG16_2015_4_classes_weights_without_fc_hinge.h5')
# test_data_dir = os.path.join(root_path, 'data/2015_4_classes/aug_256/train')
weights_path = os.path.join(root_path, 'weights/new_10_classes/RVGG16_10_cls_200_weights.h5')
test_data_dir = os.path.join(root_path, 'data/process_imgsize200/test')

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
InceptionV3_model = keras.(weights_path)

print('Begin to predict for testing data ...')
preds = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
print(preds)
predictions = InceptionV3_model.evaluate_generator(test_generator, steps=batch_size)
print(InceptionV3_model.metrics_names)  # ['loss', 'acc']
print(predictions) # [1.0047961547970772, 0.6640625]
np.savetxt(os.path.join(root_path, 'predictions_new_data.txt'), predictions)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit_new_data.csv'), 'w')
f_submit.write('image,01_gengdi,02_yuandi,03_lindi,04_caodi,05_fangwujianzhu,06_road,07_gouzhuwu,08_rengong,09_huangmo,10_water\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.dirname(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
