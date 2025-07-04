# Butterfly andn Moth Image Classification using CNN from scratch
This project aims to classify images of butterflies and moths into 10 different species using a custom-built Convolutional Neural Network (CNN) trained from scratch using PyTorch. The project also features an interactive UI using Gradio where users can upload an image and get real-time predictions.

## 🧠 Pretrained Model Included

This repository includes the saved model weights:  
**`butterfly_model.pth`**
You can reuse the trained model without retraining by loading it as follows:
```python
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load("butterfly_model.pth", map_location="cpu"))
model.eval()
```python

## Dataset
The dataset was manually curated from a 100-class butterfly/moth dataset. We selected 10 unique species and structured the dataset as follows:
my_10_class_dataset/
├── TRAIN_BALANCED/ # Balanced training set
├── VALID/ # Validation set (5 images per class)
└── TEST/ # Test set (5 images per class)
- Image size: `224 x 224 x 3` (RGB)
- Total classes: **10**
- Train samples: **1210**
- Test/Validation samples: **50 each**


## Sample Classes

1. BANDED TIGER MOTH  
2. CHALK HILL BLUE  
3. EMPEROR GUM MOTH  
4. GLITTERING SAPPHIRE  
5. GREEN HAIRSTREAK  
6. MOURNING CLOAK  
7. POPINJAY  
8. PURPLISH COPPER  
9. RED CRACKER  
10. ROSY MAPLE MOTH

## Future Improvements

- Use pretrained models like ResNet for higher accuracy
- Handle class imbalance with SMOTE or weighted loss
- Deploy on HuggingFace Spaces or a Flask web server
