import torch
import os
from model import Model
import torchvision.transforms.v2 as T
from data_loader import Connect4Dataset
from torch.utils.data import random_split
from transformer import transform_test

model = torch.load(os.path.join("models", "first_model.pt"), weights_only= False)



# Create dataset
dataset = Connect4Dataset(
    labels_path="data/labels.json",
    images_dir="data/images",
    transform=transform_test
)

# Split into train/validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Get device from model
device = next(model.parameters()).device

model.eval()
with torch.no_grad():
    # Move input to same device as model and add batch dimension
    input_image = val_set[0][0].unsqueeze(0).to(device)
    y_true = val_set[0][1].unsqueeze(0).to(device)
    output = model(input_image)
y_pred = torch.argmax(output, dim=1)
print("y_pred = \n", y_pred)
print("y_true = \n", y_true.reshape([6,7]).cpu().numpy())
print(torch.sum((y_pred - y_true), dim=[0,1,2]))
    