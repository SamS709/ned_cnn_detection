import torchvision
import torch.nn as nn
import torch
pretrained = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

encoder_layers = list(pretrained.children())
encoder_early = nn.Sequential(*encoder_layers[:-1]) # layer4 only (remove avgpool and fc)
print(pretrained)

X = torch.zeros((32,  3 * 9))
X = X.view(-1, 3, 3, 3)
print(X.shape)

import torch
from torch.utils.data import DataLoader, random_split
from model import Model
from data_loader import TicTacToeDataset
from transformer import transform_train
dataset = TicTacToeDataset(
    labels_path="tictactoe/data/labels.json",
    images_dir="tictactoe/data/images",
    transform=transform_train
)
model = Model()
# Split into train/validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

print(f"Train set: {len(train_set)} images")
print(f"Validation set: {len(val_set)} images")


# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
Xs, ys = next(iter(train_loader))
ypred = model(Xs)
print(ys.shape)
print(ypred.shape)


