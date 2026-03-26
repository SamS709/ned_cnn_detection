

import torchvision.transforms.v2 as T

# Define transforms with color/lighting augmentation
transform_train = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=8),
    T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.7, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
