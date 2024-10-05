import torch
import os
import albumentations as album
from torch.utils.data import DataLoader
from datasets.dataset import DefectsDataset, get_preprocessing
from models.unet import preprocessing_fn, model
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from settings import class_rgb_values

# Data directory
DATA_DIR = "data/data_training_hessingeim"

# Directories of train, test, etc.
x_train_dir = os.path.join(DATA_DIR, 'train/original')
y_train_dir = os.path.join(DATA_DIR, 'train/segment')

x_valid_dir = os.path.join(DATA_DIR, 'val/original')
y_valid_dir = os.path.join(DATA_DIR, 'val/segment')
image_size = 512
# Resize images and optionally apply other augmentations
def get_no_augmentation(target_size=(image_size, image_size)):
    return album.Compose([
        album.Resize(height=target_size[0], width=target_size[1])
    ])

# Prepare datasets without augmentation but with resizing
train_dataset = DefectsDataset(
    x_train_dir, y_train_dir,
    augmentation=get_no_augmentation(target_size=(image_size, image_size)),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
    mask_format='png'
)

valid_dataset = DefectsDataset(
    x_valid_dir, y_valid_dir,
    augmentation=get_no_augmentation(target_size=(image_size, image_size)),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
    mask_format='png'
)

# Get train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

TRAINING = True

# Set number of epochs
EPOCHS = 30

# Set device: cuda or cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function
loss = smp.utils.losses.DiceLoss()


# Define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
]

# Define optimizer
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

    for epoch in range(EPOCHS):
        print('\nEpoch: {}'.format(epoch))
        
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Output metrics
        print(f"Epoch: {epoch}, Train Loss: {train_logs['dice_loss']}, Val Loss: {valid_logs['dice_loss']}")
        print(f"Train IoU: {train_logs['iou_score']}, Val IoU: {valid_logs['iou_score']}")
        print(f"Train F-score: {train_logs['fscore']}, Val F-score: {valid_logs['fscore']}")
        print(f"Train Accuracy: {train_logs['accuracy']}, Val Accuracy: {valid_logs['accuracy']}")
        print(f"Train Recall: {train_logs['recall']}, Val Recall: {valid_logs['recall']}")
        print(f"Train Precision: {train_logs['precision']}, Val Precision: {valid_logs['precision']}")

        # Save model weights with epoch number in filename
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model.state_dict(), f'model_save/best_model_epoch_{epoch + 1}.pth')
            print('Model saved!')

