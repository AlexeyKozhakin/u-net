from PIL import Image
from tensorflow.keras.preprocessing import image
# Инструменты для работы с массивами
import numpy as np
# Системные инструменты
import time
import os



class_colors_hessinheim = {
    0: (34, 139, 34),   # Low Vegetation - Низкая растительность
    1: (128, 128, 128), # Impervious Surface - Непроницаемая поверхность
    2: (255, 0, 0),     # Vehicle - Транспортное средство
    3: (255, 165, 0),   # Urban Furniture - Городская мебель
    4: (0, 0, 255),     # Roof - Крыша
    5: (128, 0, 128),   # Façade - Фасад
    6: (0, 255, 0),     # Shrub - Кустарник
    7: (0, 100, 0),     # Tree - Дерево
    8: (139, 69, 19),   # Soil/Gravel - Почва/Гравий
    9: (64, 224, 208),  # Vertical Surface - Вертикальная поверхность
    10: (255, 255, 0),  # Chimney - Дымоход
}

CLASS_LABELS = list(class_colors_hessinheim.values())

CLASS_COUNT = len(CLASS_LABELS)


IMG_WIDTH = 128               # Ширина картинки
IMG_HEIGHT = 128              # Высота картинки
TRAIN_DIRECTORY = 'data/data_training_hessingeim/train'     # Название папки с файлами обучающей выборки
VAL_DIRECTORY = 'data/data_training_hessingeim/val'         # Название папки с файлами проверочной выборки
TEST_DIRECTORY = 'data/data_training_hessingeim/test'




def load_imageset(folder,   # имя папки
                  subset,   # подмножество изображений - оригинальные или сегментированные
                  title     # имя выборки
                  ):

    # Cписок для хранения изображений выборки
    image_list = []

    # Отметка текущего времени
    cur_time = time.time()

    # Для всех файлов в каталоге по указанному пути:
    for filename in sorted(os.listdir(f'{folder}/{subset}')):

        # Чтение очередной картинки и добавление ее в список изображений требуемого размера
        image_list.append(image.load_img(os.path.join(f'{folder}/{subset}', filename),
                                         target_size=(IMG_WIDTH, IMG_HEIGHT)))

    # Вывод времени загрузки картинок выборки
    print('{} выборка загружена. Время загрузки: {:.2f} с'.format(title,
                                                                  time.time() - cur_time))

    # Вывод количества элементов в выборке
    print('Количество изображений:', len(image_list))

    return image_list


# Функция преобразования цветного сегментированного изображения в метки классов

def rgb_to_labels(image_list  # список цветных изображений
                 ):

    result = []

    # Для всех картинок в списке:
    for d in image_list:
        sample = np.array(d)
        # Создание пустой 1-канальной картики
        y = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype='uint8')

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение 3-х канальных пикселей классов и занесение метки класса
            y[np.where(np.all(sample == CLASS_LABELS[i], axis=-1))] = i

        result.append(y)

    return np.array(result)

# Функция преобразования тензора меток класса в цветное сегметрированное изображение

def labels_to_rgb(image_list  # список одноканальных изображений
                 ):

    result = []

    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype='uint8')

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y==i, axis=-1))] = CLASS_LABELS[i]

        result.append(temp)

    return np.array(result)


# Функция визуализации процесса сегментации изображений
def process_images_predict_save(model,        # обученная модель
                        count = 1,    # количество случайных картинок для сегментации
                        save_dir='predictions'  # директория для сохранения изображений
                       ):

    # Создание директории, если её не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Генерация случайного списка индексов в количестве count между (0, len(x_val)
    indexes = np.arange(count)

    # Вычисление предсказания сети для картинок с отобранными индексами
    predict = np.argmax(model.predict(x_test[indexes]), axis=-1)

    # Подготовка цветов классов для отрисовки предсказания
    orig = labels_to_rgb(predict[..., None])
    fig, axs = plt.subplots(3, count, figsize=(25, 15))

    # Отрисовка результата работы модели
    for i in range(count):
        # Сохранение предсказанного изображения в формате PNG
        pred_image = Image.fromarray(orig[i])
        pred_image.save(os.path.join(save_dir, f'{indexes[i]}.png'))