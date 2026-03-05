from torch.utils.data import Dataset, random_split
from pathlib import Path
from PIL import Image
import json
import numpy as np
import torch
class TicTacToeDataset(Dataset):
    def __init__(self, labels_path="data/labels.json", images_dir="data/images", transform=None):
        """
        Custom Dataset for Tic Tac Toe images and labels.
        
        Args:
            labels_path: Path to labels.json file
            images_dir: Directory containing images
            transform: Optional transforms to apply to images
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels_data = json.load(f)
        
        # Get list of labeled images and filter out missing ones
        all_image_names = list(self.labels_data.keys())
        self.image_names = [name for name in all_image_names 
                           if (self.images_dir / name).exists()]
        
        missing_count = len(all_image_names) - len(self.image_names)
        if missing_count > 0:
            print(f"Warning: {missing_count} images not found and will be skipped")
        print(f"Loaded {len(self.image_names)} labeled images")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Get image name and load image
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        image = Image.open(image_path).convert('RGB')
        
        # Get label grid (3x3)
        grid = np.array(self.labels_data[image_name]["grid"], dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        # Convert grid to tensor
        label = torch.from_numpy(grid)  # Shape: (3, 3)
        
        
        return image, label
    
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, random_split
    import numpy as np

    import matplotlib.pyplot as plt
    from data_loader import TicTacToeDataset
    from transformer import transform_train
    dataset = TicTacToeDataset(
        labels_path="tictactoe/data/labels.json",
        images_dir="tictactoe/data/images",
        transform=transform_train
    )
    
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
    
    # Display 20 images in a grid
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(20):
        img = Xs[i].permute(1, 2, 0)  # Convert from CHW to HWC
        label = ys[i].numpy()
        
        axes[i].imshow(img)
        axes[i].set_title(f"Grid:\n{label}", fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
