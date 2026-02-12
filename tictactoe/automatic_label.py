import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# bad data:
# 80, 133, 149, 207, 308, 325, 337, 367, 388, 395, 444, 454

class Connect4Labeler:
    def __init__(self, images_dir="data/images", labels_path="data/labels.json"):
        self.images_dir = Path(images_dir)
        self.labels_path = Path(labels_path)
        self.grid = np.zeros((3, 3), dtype=int)  # 3 * 3 grid
        self.current_image = None
        self.current_image_name = None
        self.image_list = []
        self.current_index = 0
        self.labels_data = {}
        self.width_resize = 224
        self.height_resize = 192
        
        # Colors for visualization
        self.colors = {
            0: (200, 200, 200),  # Empty - light gray
            1: (0, 0, 255),      # Player 1 - red
            2: (255, 255, 0)     # Player 2 - cyan
        }
        
        # Load existing labels if they exist
        self.load_labels()
        
        # Get list of images
        self.load_image_list()
        self.load_first_image_index()
        

        
    def load_labels(self):
        """Load existing labels from JSON file"""
        if self.labels_path.exists():
            try:
                with open(self.labels_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.labels_data = json.loads(content)
                        print(f"Loaded {len(self.labels_data)} existing labels")
                    else:
                        self.labels_data = {}
                        print("Labels file is empty, starting fresh")
            except json.JSONDecodeError:
                self.labels_data = {}
                print("Labels file contains invalid JSON, starting fresh")
        else:
            self.labels_data = {}
            
    def save_labels(self):
        """Save labels to JSON file"""
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_path, 'w') as f:
            json.dump(self.labels_data, f, indent=2)
        print(f"Saved labels to {self.labels_path}")
        
    def load_image_list(self):
        """Get list of all images in the directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            self.image_list.extend(list(self.images_dir.glob(ext)))
        
        self.image_list.sort()
        print(f"Found {len(self.image_list)} images")
    
    def load_first_image_index(self):
        if not self.image_list:
            print("No images found!")
            return False
        if len(self.labels_data) > 0:
            self.current_index = len(self.labels_data) - 2
        print("[INFO]: First image not labelled found at index: ", self.current_index)
        
    def load_current_image(self):
        """Load the current image and its label if exists"""
        if not self.image_list:
            print("No images found!")
            return False
            
        image_path = self.image_list[self.current_index]
        self.current_image_name = image_path.name
        self.current_image = cv2.imread(str(image_path))
        
        if self.current_image is None:
            print(f"Error loading image: {image_path}")
            return False
        
        # Resize to 224×192 (width × height)
        self.current_image = cv2.resize(self.current_image, (4 * self.width_resize, 4 * self.height_resize ))
        
        # Load existing label if available
        if self.current_image_name in self.labels_data:
            self.grid = np.array(self.labels_data[self.current_image_name]["grid"], dtype=int)
            print(f"Loaded existing label for {self.current_image_name}")
        # else:
        #     self.grid = np.zeros((3, 3), dtype=int)
            
        return True
        
    def draw_grid_overlay(self, display_image):
        """Draw the Connect 4 grid overlay on the image"""
        h, w = display_image.shape[:2]
        
        # Calculate cell dimensions
        cell_width = w // 3
        cell_height = h // 3
        
        # Draw grid lines
        for i in range(1, 3):
            cv2.line(display_image, (i * cell_width, 0), (i * cell_width, h), (0, 255, 0), 2)
        for i in range(1, 3):
            cv2.line(display_image, (0, i * cell_height), (w, i * cell_height), (0, 255, 0), 2)
            
        # Draw circles for labeled positions
        for row in range(3):
            for col in range(3):
                if self.grid[row, col] != 0:
                    center_x = col * cell_width + cell_width // 2
                    center_y = row * cell_height + cell_height // 2
                    white = (255, 255, 255)
                    size = min(cell_width, cell_height) // 3
                    
                    if self.grid[row, col] == 2:
                        # Draw white square for value 1
                        top_left = (center_x - size, center_y - size)
                        bottom_right = (center_x + size, center_y + size)
                        cv2.rectangle(display_image, top_left, bottom_right, white, -1)
                        # Add black border
                        cv2.rectangle(display_image, top_left, bottom_right, (0, 0, 0), 2)
                    else:
                        # Draw white circle for value 2
                        cv2.circle(display_image, (center_x, center_y), size, white, -1)
                        # Add black border
                        cv2.circle(display_image, (center_x, center_y), size, (0, 0, 0), 2)
                             
        return display_image
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the image"""
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.current_image.shape[:2]
            cell_width = w // 3
            cell_height = h // 3
            
            # Calculate which cell was clicked
            col = x // cell_width
            row = y // cell_height
            
            # Ensure valid indices
            if 0 <= row < 3 and 0 <= col < 3:
                # Cycle through states: 0 -> 1 -> 2 -> 0
                self.grid[row, col] = (self.grid[row, col] + 1) % 3
                print(f"Cell ({row}, {col}) = {self.grid[row, col]}")
                
    def save_current_label(self):
        """Save the current grid label"""
        self.labels_data[self.current_image_name] = {
            "grid": self.grid.tolist(),
            "labeled_at": datetime.now().isoformat()
        }
        self.save_labels()
        
    def next_image(self):
        """Move to the next image"""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            return True
        return False
        
    def prev_image(self):
        """Move to the previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
        
    def run(self):
        """Main labeling loop"""
        if not self.image_list:
            print("No images found in directory!")
            return
            
        cv2.namedWindow('Connect 4 Labeler')
        cv2.setMouseCallback('Connect 4 Labeler', self.mouse_callback)
        
        self.load_current_image()
        
        print("\n=== Connect 4 Labeling Tool ===")
        print("Controls:")
        print("  Left Click: Cycle cell state (Empty -> Player 1 -> Player 2)")
        print("  's': Save current label")
        print("  'n': Next image (auto-saves)")
        print("  'p': Previous image (auto-saves)")
        print("  'r': Reset current grid")
        print("  'q': Quit (auto-saves)")
        print("================================\n")
        
        while True:
            # Create display image
            display_image = self.current_image.copy()
            display_image = self.draw_grid_overlay(display_image)
            
            # Add info text
            info_text = f"Image {self.current_index + 1}/{len(self.image_list)}: {self.current_image_name}"
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Display
            cv2.imshow('Connect 4 Labeler', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                self.save_current_label()
                break
            elif key == ord('s'):
                # Save
                self.save_current_label()
            elif key == ord('n'):
                # Next image
                self.save_current_label()
                if self.next_image():
                    self.load_current_image()
                else:
                    print("Already at last image")
            elif key == ord('p'):
                # Previous image
                self.save_current_label()
                if self.prev_image():
                    self.load_current_image()
                else:
                    print("Already at first image")
            elif key == ord('r'):
                # Reset grid
                self.grid = np.zeros((3, 3), dtype=int)
                print("Grid reset")
                
        cv2.destroyAllWindows()
        print(f"\nLabeling complete! Total labels: {len(self.labels_data)}")


if __name__ == "__main__":
    labeler = Connect4Labeler()
    labeler.run()