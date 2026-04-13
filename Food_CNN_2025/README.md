# Food-91 Classification with Custom CNN

A deep learning project that classifies food images into **91 categories** using a custom CNN trained from scratch, then generates a user taste profile with an LLM.

## Demo

```
Predicted Taste Profile (10 images):
  grilled_salmon: 1
  onion_rings: 1
  poutine: 1
  french_onion_soup: 1
  hamburger: 1
  escargots: 1
  clam_chowder: 1
  chicken_curry: 1
  grilled_cheese_sandwich: 1
  filet_mignon: 1
```

> *This person enjoys a diverse range of foods, spanning from seafood like shrimp, fish and chips, and chowder to savory dishes like wings, curry, and pork chops. They also have a liking for classic comfort food such as pie and soup, but with an adventurous side as shown by their preference for escargots.*

## Model Architecture

Custom CNN built from scratch (no pretrained weights):

```
Stem (Conv→BN→ReLU→MaxPool)
  ↓
ResidualBlock  64 → 128  (with SE attention)
  ↓
ResidualBlock 128 → 256  (with SE attention)
  ↓
ResidualBlock 256 → 512  (with SE attention)
  ↓
Global Average Pool → FC 512→256 → Dropout → FC 256→91
```

Key components:
- **Squeeze-and-Excitation (SE) blocks** — channel attention that learns which feature maps matter most
- **Residual connections** — skip connections for stable deep training
- **Weighted sampling** — handles class imbalance in the dataset

**Result: 51.32% test accuracy** on 91 food categories.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 96 × 96 |
| Batch size | 32 |
| Optimizer | AdamW (weight decay 3e-4) |
| Learning rate | 5e-4 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 4) |
| Loss | CrossEntropy (label smoothing 0.2) |
| Dropout | 0.5 |
| Max epochs | 100 (early stop patience 15) |
| Augmentation | RandomResizedCrop, HorizontalFlip |

## Quick Start

### 1. Requirements

```bash
pip install torch torchvision numpy
pip install google-genai  # optional, for LLM taste profile
```

Tested with PyTorch 2.6 and Python 3.10+.

### 2. Prepare Dataset

Download the [Food-91 dataset](https://github.com/MaritPaul/Neural-Computing-datasets) and organize as:

```
datasets/
├── train/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   └── ...  (91 classes, ~45k images)
└── test/
    ├── apple_pie/
    ├── baby_back_ribs/
    └── ...  (91 classes, ~22k images)
```

### 3. Run

Open `food_classification_cnn.ipynb` and update the dataset paths:

```python
train_path = "./datasets/train"
test_path  = "./datasets/test"
```

Run all cells. Training takes roughly 1–2 hours on a single GPU.

### 4. LLM Taste Profile (Optional)

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Then run the bonus section in the notebook.

## Project Structure

```
.
├── README.md
├── food_classification_cnn.ipynb   # full pipeline: model → training → evaluation → LLM
└── best_model.pth                  # saved weights (generated after training)
```

## License

MIT
