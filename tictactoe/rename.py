import os
from pathlib import Path

# Rename the pictures by "image_ijkl"

def rename_images():
    images_dir = Path("tictactoe/data/images")
    
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix == '.png'])
    
    print(f"Found {len(image_files)} images to rename")
    
    # Rename each image with the pattern image_0000, image_0001, etc.
    for idx, old_path in enumerate(image_files):
        new_name = f"image_{idx:04d}.png"
        new_path = images_dir / new_name
        
        old_path.rename(new_path)
        print(f"Renamed: {old_path.name} -> {new_name}")
    
    print(f"\nSuccessfully renamed {len(image_files)} images!")

if __name__ == "__main__":
    rename_images()
