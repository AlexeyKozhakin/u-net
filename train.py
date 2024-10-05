import torch
import os
import albumentations as album
from torch.utils.data import DataLoader
from datasets.dataset import DefectsDataset, get_preprocessing
from models.unet import preprocessing_fn
import segmentation_models_pytorch as smp
from settings import class_rgb_values
from models.unet import model

# Data directory
DATA_DIR = "data/data_training_hessingeim"

# Directories of train, test, etc.
x_train_dir = os.path.join(DATA_DIR, 'train/original')
y_train_dir = os.path.join(DATA_DIR, 'train/segment')

x_valid_dir = os.path.join(DATA_DIR, 'val/original')
y_valid_dir = os.path.join(DATA_DIR, 'val/segment')

x_test_dir = os.path.join(DATA_DIR, 'test/original')
y_test_dir = os.path.join(DATA_DIR, 'test/segment')

# Resize images and optionally apply other augmentations
def get_no_augmentation(target_size=(256, 256)):
    return album.Compose([
        album.Resize(height=target_size[0], width=target_size[1])  # Resize images
    ])

# Prepare your datasets without augmentation but with resizing
train_dataset = DefectsDataset(
    x_train_dir, y_train_dir,
    augmentation=get_no_augmentation(target_size=(256, 256)),  # Указание целевого размера
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
    mask_format='png'
)

valid_dataset = DefectsDataset(
    x_valid_dir, y_valid_dir,
    augmentation=get_no_augmentation(target_size=(256, 256)),  # Указание целевого размера
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
    mask_format='png'
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)


from segmentation_models_pytorch import utils
TRAINING = True

# Set num of epochs
EPOCHS = 30

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.utils.losses.DiceLoss()#DiceLoss_()#

# define metrics
metrics = [
    #IoU_(),
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    #Precision(),
    #Recall(),
    #F1(),
    #Accuracy()
    #smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


if TRAINING:
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):
        # Выполнение обучения и валидации
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        val_loss = valid_logs['dice_loss']
        train_loss = train_logs['dice_loss']
        val_iou = valid_logs['iou_score']
        train_iou = train_logs['iou_score']

        # Вывод значений на экран
        print(f"Epoch: {i}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"Train IoU: {train_iou}, Val IoU: {val_iou}")

        # Сохранение модели каждые 10 эпох
        # if i % 10 == 0:
        #     torch.save(model, f'/content/drive/MyDrive/Проект_дефекты/Models/best_model_14_{i + 1}_epoch.pth')

        # Сохранение модели, если получен лучший val IoU score
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, 'model_save/best_model_test.pth')
            print('Model saved!')
