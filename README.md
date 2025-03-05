# Deep Learning Classifier with Interpretability

A comprehensive Python project that implements deep learning classifiers with integrated model interpretability for both image and text data.

## Features

### Image Classification
- CNN architecture built with TensorFlow/Keras
- Training on standard datasets (CIFAR-10, MNIST)
- Grad-CAM visualization for model interpretability

### Text Classification
- Transformer-based model using Hugging Face
- Sentiment analysis on IMDb reviews
- SHAP (SHapley Additive exPlanations) for explaining model predictions

## Project Structure

```
Deep Learning Image-Text Classifier/
├── config/                 # Configuration files
├── data/                   # Data handling modules
│   ├── image_data.py       # Image dataset loaders
│   └── text_data.py        # Text dataset loaders
├── models/                 # Model definitions
│   ├── image_model.py      # CNN architecture for images
│   └── text_model.py       # Transformer model for text
├── interpretability/       # Interpretability implementations
│   ├── gradcam.py          # Grad-CAM for image models
│   └── shap_explainer.py   # SHAP for text models
├── utils/                  # Utility functions
│   ├── visualization.py    # Visualization tools
│   └── metrics.py          # Evaluation metrics
├── main.py                 # Main entry point
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Dependencies
```

## Setup and Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Image Classification Mode
```
python main.py --mode image --dataset cifar10 --epochs 10
```

### Text Classification Mode
```
python main.py --mode text --dataset imdb --epochs 5
```

### Model Interpretation
After training a model, you can generate interpretability visualizations:

```
python main.py --mode image --interpret --model_path path/to/saved/model
```

## Requirements
- Python 3.8+
- TensorFlow/Keras or PyTorch
- Hugging Face Transformers
- SHAP library
- Grad-CAM library
- Other dependencies in requirements.txt
