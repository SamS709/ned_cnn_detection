

import torchvision.transforms.v2 as T

# Define transforms with color/lighting augmentation
transform_train = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Slight rotation and offset    
    T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5),  # Removed hue
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
