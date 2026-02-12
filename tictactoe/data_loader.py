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
        image = Image.open(image_path).convert('L')
        
        # Get label grid (3x3)
        grid = np.array(self.labels_data[image_name]["grid"], dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        # Convert grid to tensor
        label = torch.from_numpy(grid)  # Shape: (3, 3)
        
        
        return image, label

