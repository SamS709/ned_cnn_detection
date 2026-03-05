import torch
import os
import torch.nn as nn
import torchmetrics
import torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np

import matplotlib.pyplot as plt
from model import Model
from data_loader import Connect4Dataset
from transformer import transform_train


def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train_with_scheduler(model, optimizer, loss_fn, metric, train_loader,
                         valid_loader, n_epochs, scheduler, device):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        losses = []
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        history["train_losses"].append(np.mean(losses))
        history["train_metrics"].append(metric.compute().item())
        val_metric = evaluate_tm(model, valid_loader, metric, device).item()
        history["valid_metrics"].append(val_metric)
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric)
        else:
            scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.5f}")
    return history



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = Connect4Dataset(
        labels_path="data/labels.json",
        images_dir="data/images",
        transform=transform_train
    )
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    print(f"Train set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")

    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    model = Model().to(device)
    
    # Training parameters
    model_dir = "models"
    plot_dir = "plots"
    n_epochs = 50
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Train the model
    history = train_with_scheduler(model, optimizer, loss_fn, accuracy, train_loader, val_loader, n_epochs, scheduler, device)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir) 
    plt.plot(history["train_losses"])
    plt.plot(history["train_metrics"])
    plt.plot(history["valid_metrics"])
    
    plt.savefig(os.path.join(plot_dir, "training_plot2.png"))
    torch.save(model, os.path.join(model_dir,"model2.pt"))
    

