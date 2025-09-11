# White Blood Cell Classifier

A deep learning project that classifies four types of white blood cells using Convolutional Neural Networks (CNNs) and transfer learning with ResNet-50.

## 🔬 Project Overview

This project implements an automated white blood cell classification system that can distinguish between four main types of white blood cells:

- **Eosinophil** - involved in allergic reactions and parasitic infections
- **Lymphocyte** - key players in adaptive immunity
- **Monocyte** - largest white blood cells that differentiate into macrophages
- **Neutrophil** - most abundant white blood cells, first responders to infections

## 📊 Dataset

- **Training Dataset**: 28,000 images (22,400 train / 2,800 validation / 2,800 test)
- **Unseen Test Dataset**: 9,042 images for final evaluation
- **Image Resolution**: Original 480×640 pixels, resized to 224×224 for model input
- **Data Split**: 80% training, 10% validation, 10% testing

## 🏗️ Model Architecture

### Primary Model: ResNet-50 + Custom Classifier

1. **Feature Extraction**: Pre-trained ResNet-50 (frozen weights)

   - Extracts 2048-dimensional feature vectors
   - Uses transfer learning from ImageNet

2. **Custom Classifier Head**:
   ```
   Linear(2048 → 512) + BatchNorm + ReLU + Dropout
   Linear(512 → 128) + BatchNorm + ReLU + Dropout
   Linear(128 → 4) [Output layer]
   ```

### Baseline Model: GPU-Accelerated SVM

- Color histogram features + Edge detection features
- Support Vector Machine with linear kernel
- Used for performance comparison

## 🎯 Model Performance

### Final Results

- **Test Accuracy**: 98.06%
- **Validation Accuracy**: 97.81%
- **Unseen Dataset Accuracy**: 75.35%

### Hyperparameter Optimization

Used Optuna for automated hyperparameter tuning:

- **Dropout Rate**: 0.25
- **Learning Rate**: 1.678 × 10⁻⁵
- **Batch Size**: 32
- **Weight Decay**: 0.00562
- **Training Epochs**: 30

## 📈 Model Results

### Side-by-Side Comparison

![Side by Side Comparison](img/side_by_side_image.png)
_Comparison of predicted vs actual classifications_

### Prediction Results

![Reshaped Predictions](img/reshaped_predictions.png)
_Model predictions visualized on test samples_

## 🚀 Key Features

- **Transfer Learning**: Leverages pre-trained ResNet-50 for robust feature extraction
- **Automated Hyperparameter Tuning**: Uses Optuna for optimal parameter selection
- **Data Augmentation**: Normalization and tensor transformations
- **GPU Acceleration**: CUDA support for faster training and inference
- **Comprehensive Evaluation**: Multiple datasets for thorough performance assessment

## 📋 Technical Implementation

### Data Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Training Configuration

- **Optimizer**: Adam with weight decay
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Learning Rate**: 1.678 × 10⁻⁵ (optimized)
- **Regularization**: Dropout (0.25) + Weight Decay (0.00562)

## 📁 Project Structure

```
white-blood-cell-classifier/
├── white_blood_cell_classifier.ipynb  # Main project notebook
├── best_model.pth                     # Trained model weights
├── img/                               # Result visualizations
│   ├── side_by_side_image.png
│   └── reshaped_predictions.png
└── README.md                          # Project documentation
```

## 🔧 Requirements

- Python 3.8+
- PyTorch
- torchvision
- CUDA (for GPU acceleration)
- scikit-learn
- matplotlib
- numpy
- Optuna (for hyperparameter tuning)
- cuML (for GPU-accelerated SVM baseline)

## 🏥 Medical Significance

Accurate white blood cell classification is crucial for:

- **Disease Diagnosis**: Identifying infections, immune disorders, and blood cancers
- **Treatment Monitoring**: Tracking patient response to therapy
- **Laboratory Efficiency**: Automating manual microscopic analysis
- **Standardization**: Reducing inter-observer variability in cell counting

## 🎓 Academic Context

This project was developed as part of **APS360: Applied Fundamentals of Deep Learning** at the University of Toronto. It demonstrates practical application of deep learning techniques in medical image analysis and showcases the effectiveness of transfer learning for specialized classification tasks.

## 📊 Performance Analysis

The model shows excellent performance on the test set (98.06%) but experiences a notable drop on the unseen dataset (75.35%), indicating potential domain shift or overfitting. This highlights the importance of:

- Cross-validation across multiple datasets
- Domain adaptation techniques
- Robust evaluation protocols in medical AI applications

## 🚀 Future Improvements

- Implement data augmentation strategies
- Explore ensemble methods
- Add attention mechanisms for interpretability
- Develop real-time inference pipeline
- Extend to additional cell types and abnormal cell detection
