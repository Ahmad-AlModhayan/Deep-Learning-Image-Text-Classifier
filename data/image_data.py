"""
Image data handling module for loading and preprocessing image datasets.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from config.config import IMAGE_CONFIG, IMAGE_AUGMENTATION, BATCH_SIZE, RANDOM_SEED

def load_dataset(dataset_name):
    """
    Load and preprocess the specified image dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('cifar10' or 'mnist')
        
    Returns:
    --------
    tuple
        (x_train, y_train, x_test, y_test, class_names)
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Add channel dimension for grayscale images
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from 'cifar10' or 'mnist'")
    
    # Get dataset configuration
    config = IMAGE_CONFIG[dataset_name]
    class_names = config['class_names']
    num_classes = config['num_classes']
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Dataset loaded: train shape: {x_train.shape}, test shape: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test, class_names

def create_data_generators(x_train, y_train, x_test, y_test, use_augmentation=True):
    """
    Create data generators for training and testing.
    
    Parameters:
    -----------
    x_train, y_train, x_test, y_test : numpy array
        Training and testing data
    use_augmentation : bool
        Whether to use data augmentation for training
        
    Returns:
    --------
    tuple
        (train_generator, test_generator)
    """
    # Data augmentation for training
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=IMAGE_AUGMENTATION['rotation_range'],
            width_shift_range=IMAGE_AUGMENTATION['width_shift_range'],
            height_shift_range=IMAGE_AUGMENTATION['height_shift_range'],
            horizontal_flip=IMAGE_AUGMENTATION['horizontal_flip'],
            zoom_range=IMAGE_AUGMENTATION['zoom_range']
        )
    else:
        train_datagen = ImageDataGenerator()
    
    test_datagen = ImageDataGenerator()  # No augmentation for test data
    
    # Create generators
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    test_generator = test_datagen.flow(
        x_test, y_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, test_generator

def visualize_dataset_samples(x_data, y_data, class_names, num_samples=10):
    """
    Visualize random samples from the dataset.
    
    Parameters:
    -----------
    x_data : numpy array
        Image data
    y_data : numpy array
        Labels (one-hot encoded)
    class_names : list
        Names of the classes
    num_samples : int
        Number of samples to visualize
    """
    # Convert one-hot encoded labels back to class indices
    if len(y_data.shape) > 1:  # Check if one-hot encoded
        y_indices = np.argmax(y_data, axis=1)
    else:
        y_indices = y_data
    
    # Select random indices
    indices = np.random.choice(range(len(x_data)), num_samples, replace=False)
    
    # Plot the samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, idx in enumerate(indices):
        img = x_data[idx]
        label = y_indices[idx]
        
        # If image is grayscale, remove the channel dimension
        if img.shape[-1] == 1:
            img = img.squeeze(axis=-1)
        
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(f"{class_names[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_augmented_samples(x_data, y_data, class_names, num_samples=5):
    """
    Visualize augmented samples from the dataset.
    
    Parameters:
    -----------
    x_data : numpy array
        Image data
    y_data : numpy array
        Labels (one-hot encoded)
    class_names : list
        Names of the classes
    num_samples : int
        Number of samples to visualize
    """
    # Create data generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=IMAGE_AUGMENTATION['rotation_range'],
        width_shift_range=IMAGE_AUGMENTATION['width_shift_range'],
        height_shift_range=IMAGE_AUGMENTATION['height_shift_range'],
        horizontal_flip=IMAGE_AUGMENTATION['horizontal_flip'],
        zoom_range=IMAGE_AUGMENTATION['zoom_range']
    )
    
    # Select a few samples
    indices = np.random.choice(range(len(x_data)), num_samples, replace=False)
    samples = x_data[indices]
    labels = np.argmax(y_data[indices], axis=1) if len(y_data.shape) > 1 else y_data[indices]
    
    # Generate augmented versions
    augmented_samples = []
    for sample in samples:
        sample = np.expand_dims(sample, axis=0)  # Add batch dimension
        # Get 4 augmented versions of the same image
        for i, augmented in enumerate(datagen.flow(sample, batch_size=1)):
            augmented_samples.append(augmented[0])
            if i >= 3:  # Get 4 versions of each image
                break
    
    # Plot original and augmented samples
    num_rows = num_samples
    num_cols = 5  # Original + 4 augmented versions
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # Original image
        img = samples[i]
        if img.shape[-1] == 1:  # If grayscale
            img = img.squeeze(axis=-1)
        
        axes[i, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i, 0].set_title(f"Original: {class_names[labels[i]]}")
        axes[i, 0].axis('off')
        
        # Augmented versions
        for j in range(4):
            aug_img = augmented_samples[i * 4 + j]
            if aug_img.shape[-1] == 1:  # If grayscale
                aug_img = aug_img.squeeze(axis=-1)
            
            axes[i, j + 1].imshow(aug_img, cmap='gray' if len(aug_img.shape) == 2 else None)
            axes[i, j + 1].set_title(f"Augmented {j+1}")
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# To help ensure the code follows principles from Andrew Ng's Deep Learning Specialization:
# - Data normalization is a key preprocessing step (scaling to [0,1])
# - Data augmentation helps prevent overfitting by creating variations of the training data
# - One-hot encoding is used for the categorical target values
