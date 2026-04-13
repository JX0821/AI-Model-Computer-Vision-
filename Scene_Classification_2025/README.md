# Scene Classification

A deep learning project for scene image classification using the MiniPlaces dataset. This implementation includes a custom ResNet-like architecture with SE blocks for improved feature representation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Configuration](#configuration)

## 🎯 Overview

This project trains a deep convolutional neural network to classify scene images into different categories using the MiniPlaces dataset. The model combines residual blocks with squeeze-and-excitation (SE) attention mechanisms to achieve better performance.

**Key Characteristics:**
- Scene classification task with 100 classes
- Custom CNN architecture with SE blocks
- Advanced training techniques (label smoothing, cosine annealing)
- Data augmentation optimized for scene understanding
- Both training and inference modes

## ✨ Features

- **Custom Architecture**: ResNet-like model with SE attention blocks for channel-wise feature recalibration
- **Advanced Training**: 
  - AdamW optimizer with weight decay
  - Label smoothing for better generalization
  - Cosine annealing with warm restarts for learning rate scheduling
- **Data Augmentation**: Optimized augmentation pipeline for scene classification
- **Checkpoint Management**: Automatic saving of best model based on validation accuracy
- **Flexible Inference**: Support for test mode with batch predictions
- **Progress Tracking**: Real-time training progress with tqdm

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
pillow>=8.0.0
tqdm>=4.50.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scene-classification
```

2. Install dependencies:
```bash
pip install torch torchvision pillow tqdm
```

3. Prepare your dataset (see [Dataset](#dataset) section)

## 📂 Dataset

### MiniPlaces Dataset Structure

The dataset should be organized as follows:

```
data/
├── train.txt          # Training file list
├── val.txt            # Validation file list
├── test.txt           # Test file list
└── images/
    ├── train/
    │   └── a/b/image.jpg
    ├── val/
    │   └── c/d/image.jpg
    └── test/
        └── e/f/image.jpg
```

### File Format

Each `.txt` file should contain:
- **Train/Val**: `<image_path> <label_id>`
- **Test**: `<image_path>`

Example:
```
train/a/abbey/00000000.jpg 0
train/a/abbey/00000001.jpg 0
train/a/airport_indoor/00000000.jpg 1
```

## 💻 Usage

### Training

To train the model:

```bash
python scene_classification.py
```

The training will:
- Run for 70 epochs by default
- Save the best model to `model.ckpt` based on validation accuracy
- Display training progress with loss and accuracy metrics
- Print epoch summaries showing train loss, validation loss, and accuracy

**Training Parameters:**
- Batch size: 32
- Learning rate: 0.001 (initial)
- Optimizer: AdamW with weight decay (1e-4)
- Loss function: CrossEntropyLoss with label smoothing (0.1)

### Testing

To generate predictions on test set:

```bash
python scene_classification.py --test --checkpoint model.ckpt
```

This will:
- Load the saved checkpoint
- Generate predictions for all test images
- Save results to `predictions.csv`

**Output Format** (`predictions.csv`):
```
test/path/to/image.jpg,42
test/path/to/image.jpg,15
```

### Command Line Arguments

```
--test              Run in test mode (inference on test set)
--checkpoint PATH   Path to checkpoint file (default: model.ckpt)
```

## 🏗️ Model Architecture

### Overall Structure

```
Input (3, 224, 224)
    ↓
Stem: Conv + BN + ReLU + MaxPool (64 channels)
    ↓
Residual Block L1 → 128 channels (stride 2)
    ↓
Residual Block L2 → 256 channels (stride 2)
    ↓
Residual Block L3 → 512 channels (stride 2)
    ↓
Adaptive Average Pooling (1, 1)
    ↓
FC Layers: 512 → 256 → num_classes
    ↓
Output (num_classes)
```

### Key Components

**SEBlock (Squeeze-and-Excitation Block)**
- Channel-wise attention mechanism
- Adaptively recalibrates feature maps
- Improves feature representation with minimal computational overhead

**ResidualBlock**
- Two 3×3 convolutions with batch normalization
- Skip connection (identity or 1×1 convolution)
- SE block for attention
- Optional stride-2 downsampling

**Data Augmentation Pipeline**

Training augmentations:
- Random resized crop (scale: 0.6-1.0)
- Random horizontal flip
- Normalization with ImageNet statistics

Validation/Test augmentations:
- Resize to (224, 224)
- Normalization with ImageNet statistics

## 📊 Results

Training progress is saved with the best model checkpoint containing:
- Model state dictionary
- Optimizer state dictionary
- Epoch number
- Best validation accuracy

Monitor the console output for:
```
Epoch: 1: Train Loss = 4.5234; Val Loss = 4.2145, Val Acc = 0.0856, Best Acc = 0.0856
Epoch: 2: Train Loss = 4.1234; Val Loss = 3.9845, Val Acc = 0.1234, Best Acc = 0.1234
...
```

## ⚙️ Configuration

### Hyperparameters (in `main()` function)

```python
batch_size = 32           # Mini-batch size
num_workers = 2           # Data loading workers
num_epochs = 70           # Total training epochs
image_size = 224          # Input image resolution
```

### Optimizer Configuration

```python
lr = 0.001               # Initial learning rate
weight_decay = 1e-4      # L2 regularization
betas = (0.9, 0.999)     # Adam betas
```

### Learning Rate Scheduler

```python
T_0 = 10                 # Initial restart cycle
T_mult = 2               # Cycle length multiplier
eta_min = 1e-6           # Minimum learning rate
```

### Loss Function

```python
label_smoothing = 0.1    # Label smoothing factor
```

## 💡 Tips for Best Results

1. **Data**: Ensure clean, properly labeled dataset with balanced classes
2. **GPU**: Use CUDA for faster training (automatically detected)
3. **Augmentation**: Adjust augmentation intensity if overfitting occurs
4. **Epochs**: Increase `num_epochs` if validation accuracy is still improving
5. **Batch Size**: Larger batch sizes (64-128) may improve performance on powerful GPUs
6. **Learning Rate**: Fine-tune initial LR based on your specific dataset

## 📝 License

[Add your license information here]

## 👥 Author

[Your name/organization]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

## 📧 Contact

[Your contact information]

---

**Last Updated**: April 2026
