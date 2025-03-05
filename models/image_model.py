"""
CNN model architecture for image classification tasks.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from config.config import CNN_CONFIG, IMAGE_CONFIG, LEARNING_RATE

def create_cnn_model(dataset_name, input_shape=None):
    """
    Create a CNN model for image classification.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('cifar10' or 'mnist')
    input_shape : tuple, optional
        Shape of input images, defaults to dataset's standard shape
        
    Returns:
    --------
    keras.Model
        Compiled CNN model
    """
    # Get dataset configuration
    config = IMAGE_CONFIG[dataset_name]
    num_classes = config['num_classes']
    
    if input_shape is None:
        # Use dataset's standard shape
        img_size = config['img_size']
        # Add channel dimension (3 for RGB, 1 for grayscale)
        channels = 1 if dataset_name == 'mnist' else 3
        input_shape = img_size + (channels,)
    
    print(f"Creating CNN model for {dataset_name} with input shape {input_shape}")
    
    # Define model architecture
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(CNN_CONFIG['filters'][0], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu', 
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(CNN_CONFIG['filters'][0], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=CNN_CONFIG['pool_size']))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(CNN_CONFIG['filters'][1], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(CNN_CONFIG['filters'][1], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=CNN_CONFIG['pool_size']))
    model.add(Dropout(0.25))
    
    # Third convolutional block
    model.add(Conv2D(CNN_CONFIG['filters'][2], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(CNN_CONFIG['filters'][2], CNN_CONFIG['kernel_size'], 
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=CNN_CONFIG['pool_size']))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(CNN_CONFIG['dense_units'][0], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(CNN_CONFIG['dropout_rate']))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def create_callbacks(model_name, patience=10):
    """
    Create callbacks for model training.
    
    Parameters:
    -----------
    model_name : str
        Name to use for model checkpoint
    patience : int
        Number of epochs with no improvement after which training will be stopped
        
    Returns:
    --------
    list
        List of callbacks
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Save the best model
    checkpoint = ModelCheckpoint(
        f"saved_models/{model_name}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]

def train_model(model, train_generator, validation_data, epochs=10, model_name='image_model'):
    """
    Train the model.
    
    Parameters:
    -----------
    model : keras.Model
        The model to train
    train_generator : keras.preprocessing.image.DataGenerator
        Generator for training data
    validation_data : tuple or keras.preprocessing.image.DataGenerator
        Validation data as (x_val, y_val) or a generator
    epochs : int
        Number of epochs to train
    model_name : str
        Name to use for model checkpoint
        
    Returns:
    --------
    keras.callbacks.History
        Training history
    """
    # Create callbacks
    callbacks = create_callbacks(model_name)
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    test_generator : keras.preprocessing.image.DataGenerator
        Generator for test data
        
    Returns:
    --------
    tuple
        (loss, accuracy)
    """
    # Evaluate the model
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return loss, accuracy

def visualize_training_history(history):
    """
    Visualize training history.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def predict_batch(model, images, class_names):
    """
    Make predictions on a batch of images.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    images : numpy.ndarray
        Batch of images
    class_names : list
        Names of the classes
        
    Returns:
    --------
    tuple
        (predictions, predicted_classes)
    """
    # Make predictions
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes

def visualize_predictions(images, true_labels, predictions, class_names, num_samples=5):
    """
    Visualize model predictions on random samples.
    
    Parameters:
    -----------
    images : numpy.ndarray
        Batch of images
    true_labels : numpy.ndarray
        True labels (one-hot encoded)
    predictions : numpy.ndarray
        Model predictions
    class_names : list
        Names of the classes
    num_samples : int
        Number of samples to visualize
    """
    # Convert one-hot encoded labels to class indices
    if len(true_labels.shape) > 1:
        true_classes = np.argmax(true_labels, axis=1)
    else:
        true_classes = true_labels
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Select random indices
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    # Plot the samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, idx in enumerate(indices):
        img = images[idx]
        
        # If image is grayscale, remove channel dimension
        if img.shape[-1] == 1:
            img = img.squeeze(axis=-1)
        
        true_class = true_classes[idx]
        pred_class = predicted_classes[idx]
        
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        title = f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}"
        color = 'green' if true_class == pred_class else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print(f"Showing {num_samples} random samples with predictions")
    for i, idx in enumerate(indices):
        true_class = true_classes[idx]
        pred_class = predicted_classes[idx]
        confidence = predictions[idx][pred_class] * 100
        print(f"Sample {i+1}: True: {class_names[true_class]}, Predicted: {class_names[pred_class]} ({confidence:.2f}% confidence)")

# This module implements CNN architecture concepts from Andrew Ng's Deep Learning Specialization:
# - Using multiple convolutional layers with increasing filter sizes
# - Batch normalization to stabilize and speed up training
# - Dropout layers to prevent overfitting
# - MaxPooling to reduce spatial dimensions
# - Global architecture follows the typical CNN pattern: Conv -> Pool -> Conv -> Pool -> Flatten -> Dense
