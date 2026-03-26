import torch
import os
from PIL import Image
from transformer import transform_test

model = torch.load(os.path.join("models", "model4.pt"), weights_only= False, map_location=torch.device('cpu') )

image_name = "current.png"
image_path = os.path.join("data", "images_sample", image_name)

image = Image.open(image_path).convert("RGB")
input_image = transform_test(image).unsqueeze(0)

# Get device from model
device = next(model.parameters()).device

model.eval()
with torch.no_grad():
    input_image = input_image.to(device)
    output = model(input_image)

y_pred = torch.argmax(output, dim=1)
print(f"Prediction for {image_name}:")
print(y_pred.reshape(6, 7).cpu().numpy())
    