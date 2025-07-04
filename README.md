# Butterfly and Moth Image Classification using CNN from scratch
This project aims to classify images of butterflies and moths into 10 different species using a custom-built Convolutional Neural Network (CNN) trained from scratch using PyTorch. The project also features an interactive UI using Gradio where users can upload an image and get real-time predictions.
![Uploading Screenshot 2025-07-03 141919.png…]()
![Uploading Screenshot 2025-07-03 142016.png…]()

## 🧠 Pretrained Model Included
This repository includes the saved model weights:  
**`butterfly_model.pth`**
You can reuse the trained model without retraining by loading it as follows:

model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load("butterfly_model.pth", map_location="cpu"))
model.eval()

## Tools & Libraries Used

- Python, PyTorch
- Torchvision
- Gradio (UI)
- Matplotlib & Sklearn (metrics)
- Google Colab (CPU environment)

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

## Class Imbalance Handling

The original training dataset had imbalanced classes. To ensure fairness, we:

- Calculated the minimum class count
- Performed **random undersampling** of all classes to match the smallest one
- Stored the balanced data in a new folder: `TRAIN_BALANCED/`

## Data Preprocessing & Augmentation

Applied using `torchvision.transforms`:

- `Resize(224, 224)` to normalize image size
- `RandomHorizontalFlip()` & `RandomRotation(10)` for augmentation
- `ToTensor()` to convert image to PyTorch tensor
- `Normalize([0.5], [0.5])` for input scaling

## Model Architecture

Built from scratch using PyTorch:

- 3 Convolutional blocks with ReLU, BatchNorm, and MaxPooling
- `Flatten → Dropout → Linear (128) → ReLU → Linear (10 classes)`
- Regularization using **Dropout (0.5)** and **BatchNorm**

### Loss & Optimization:
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Batch size: `16`, Epochs: `10`, Device: `CPU`

## Model Evaluation

- Evaluation metrics on the test set:
  - Accuracy:  **92%**
  - Precision, Recall, F1-score per class using `classification_report()`
  - Confusion matrix (visualized using matplotlib)

## Model Saving & Inference

After training:
- Saved model using `torch.save(model.state_dict(), 'butterfly_model.pth')`
- Reloaded model for inference in Colab

## Interactive UI with Gradio

Built a lightweight inference UI using `gr.Interface()`:
- Users can **upload any butterfly/moth image**
- The trained model returns the predicted class
- Runs directly inside Google Colab

### How to Use:
1. Upload your image using the Gradio UI
2. The model will return the butterfly/moth class name
3. Behind the scenes:
   - Image is preprocessed
   - Forward pass through CNN
   - Predicted label returned

## Future Improvements

- Use pre-trained models like ResNet for higher accuracy
- Handle class imbalance with SMOTE or weighted loss
- Deploy on HuggingFace Spaces or a Flask web server
