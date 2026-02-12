# Connect 4 Detection

This project implements computer vision detection for Connect 4 game boards using deep learning. It's designed as the vision component for a robotic arm system that can play Connect 4 against humans.

## Project Overview

This detection system is part of a larger robotic Connect 4 player project. The full robot arm implementation can be found here: [NED Robot Arm Project](https://github.com/SamS709/ned_project)

## Training Process

### 1. Data Collection
Approximately **700 pictures** of Connect 4 game boards were captured in various game states and lighting conditions to create a diverse training dataset.

### 2. Image Preprocessing
The `rename.py` script normalizes all image filenames to ensure consistent naming conventions across the dataset.

### 3. Labeling
The `automatic_label.py` tool provides a user-friendly GUI for annotating the Connect 4 board states in each image:

<img src="connect4/data/labeling.png" width="400" alt="Labeling Interface">

This interface allows quick and accurate labeling of each cell in the 6x7 Connect 4 grid, marking empty spaces, red pieces, and yellow pieces.

### 4. Model Training
A **Fully Convolutional Neural Network (FCNN)** is trained to detect and classify each position on the Connect 4 board. The model learns to:
- Identify the board grid structure
- Classify each cell as empty, red piece, or yellow piece
- Handle various lighting conditions

### Training Results

<img src="connect4/plots/training_plot2.png" alt="Training Progress">

The plot shows the model's learning progress across training epochs, demonstrating convergence of both training and validation metrics.

## Purpose

This detection system enables a robotic arm to:
1. Visually perceive the current game state
2. Identify valid moves
3. Plan and execute strategic gameplay against human opponents

The complete robotic system integrating this vision module with motion control is available at: [https://github.com/SamS709/ned_project](https://github.com/SamS709/ned_project)

## Usage

### Training
```bash
python main.py
```

### Testing
```bash
python test_model.py
```

### Labeling New Images
```bash
python automatic_label.py
```
