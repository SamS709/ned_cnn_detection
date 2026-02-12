

import torchvision.transforms.v2 as T

# Define transforms for grayscale images
transform_train = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

transform_test = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
