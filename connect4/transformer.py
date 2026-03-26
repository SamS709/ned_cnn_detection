

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

# Define transforms with color/lighting augmentation
transform_train = T.Compose([
    T.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=8),
    T.Resize((192, 224)),  # height, width
    T.Lambda(lambda img: F.crop(img, top=5, left=25, height=160, width=180)),
    T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.7, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.Lambda(lambda img: F.crop(img, top=5, left=25, height=160, width=180)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
