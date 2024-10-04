import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Инструменты для работы с изображениями
from tensorflow.keras.preprocessing import image
import numpy as np

from libs import (load_imageset, rgb_to_labels,
                  TRAIN_DIRECTORY, VAL_DIRECTORY, TEST_DIRECTORY,
                  CLASS_COUNT,
                  IMG_WIDTH, IMG_HEIGHT)

from model import unet

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



train_images = load_imageset(TRAIN_DIRECTORY, 'original', 'Обучающая')
val_images = load_imageset(VAL_DIRECTORY, 'original', 'Проверочная')

train_segments = load_imageset(TRAIN_DIRECTORY, 'segment', 'Обучающая')
val_segments = load_imageset(VAL_DIRECTORY, 'segment', 'Проверочная')

test_images = load_imageset(TEST_DIRECTORY, 'original', 'Проверочная')
test_segments = load_imageset(TEST_DIRECTORY, 'segment', 'Проверочная')


# Формирование обучающей выборки

x_train = []                          # Cписок под обучающую выборку
for img in train_images:              # Для всех изображений выборки:
    x = image.img_to_array(img)       # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_train.append(x)                 # Добавление элемента в x_train
x_train = np.array(x_train)           # Перевод всей выборки в numpy
print(x_train.shape)                  # Форма x_train

# Формирование проверочной выборки

x_val = []                            # Cписок под проверочную выборку
for img in val_images:                # Для всех изображений выборки:
    x = image.img_to_array(img)       # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_val.append(x)                   # Добавление элемента в x_train
x_val = np.array(x_val)               # Перевод всей выборки в numpy
print(x_val.shape)                    # Форма x_train

# Формирование проверочной выборки

x_test = []                            # Cписок под проверочную выборку
for img in test_images:                # Для всех изображений выборки:
    x = image.img_to_array(img)       # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_test.append(x)                   # Добавление элемента в x_train
x_test = np.array(x_test)               # Перевод всей выборки в numpy
print(x_test.shape)                    # Форма x_train

# Преобразование сегментов в метки классов

y_train = rgb_to_labels(train_segments)
y_val = rgb_to_labels(val_segments)
y_test = rgb_to_labels(test_segments)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)



# Создание модели и вывод сводки по архитектуре

model_unet = unet(CLASS_COUNT,
                  (IMG_WIDTH, IMG_HEIGHT, 3))

model_unet.summary()

# Определение путей для сохранения модели
model_save_path = os.path.join("/content/drive/MyDrive/ColabNotebooks/MUSAC-UNET/data_training_hessingeim",
                               'model_exp_3_lr_0.0001.{epoch:02d}.keras')

# Настройка колбеков
early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min')
mcp_save = ModelCheckpoint(model_save_path,
                           save_best_only=True,  # сохранять только лучшие модели
                          #  monitor='val_loss',
                          #  mode='min',
                           monitor='val_loss',
                           mode='min',
                           save_freq='epoch')  # сохранять каждую эпоху

# Обучение модели
history = model_unet.fit(x_train, y_train,
                         epochs=50,
                         batch_size=2,
                         validation_data=(x_val, y_val),
                         callbacks=[early_stopping, mcp_save])
