"""
Transformer-based model architecture for text classification tasks.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from transformers import TFAutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config.config import TRANSFORMER_CONFIG, TEXT_CONFIG, LEARNING_RATE

def create_transformer_model(model_name=None, num_classes=None, dataset_name='imdb'):
    """
    Create a transformer-based model for text classification.
    
    Parameters:
    -----------
    model_name : str, optional
        Hugging Face model name
    num_classes : int, optional
        Number of output classes
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    tuple
        (keras.Model, transformers.PreTrainedTokenizer)
    """
    if model_name is None:
        model_name = TRANSFORMER_CONFIG['model_name']
    
    if num_classes is None:
        config = TEXT_CONFIG[dataset_name]
        num_classes = config['num_classes']
    
    print(f"Creating transformer model based on {model_name} with {num_classes} output classes")
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = TFAutoModel.from_pretrained(model_name)
    
    # Define model architecture
    input_ids = Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    # Get transformer outputs
    outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    # Use the [CLS] token output (first token)
    cls_output = outputs[:, 0, :]
    
    # Add dropout for regularization
    x = Dropout(TRANSFORMER_CONFIG['dropout_rate'])(cls_output)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    
    # Freeze the transformer layers to speed up training and prevent overfitting
    # This is especially useful if you have a small dataset
    for layer in transformer.layers:
        layer.trainable = False
    
    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model, tokenizer

def create_callbacks(model_name, patience=5):
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
        patience=3,
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

def train_model(model, train_dataset, val_dataset, epochs=5, model_name='text_model'):
    """
    Train the model.
    
    Parameters:
    -----------
    model : keras.Model
        The model to train
    train_dataset : tf.data.Dataset
        Training dataset
    val_dataset : tf.data.Dataset
        Validation dataset
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
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_dataset):
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    test_dataset : tf.data.Dataset
        Test dataset
        
    Returns:
    --------
    tuple
        (loss, accuracy)
    """
    # Evaluate the model
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(test_dataset)
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

def predict_batch_from_dataset(model, dataset, class_names):
    """
    Make predictions on a dataset.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    dataset : tf.data.Dataset
        Dataset to predict on
    class_names : list
        Names of the classes
        
    Returns:
    --------
    tuple
        (true_labels, predictions)
    """
    # Get true labels and predictions
    all_predictions = []
    all_labels = []
    
    for batch in dataset:
        inputs, labels = batch
        predictions = model.predict(inputs)
        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    return all_labels, all_predictions

def visualize_confusion_matrix(true_labels, predictions, class_names):
    """
    Visualize confusion matrix.
    
    Parameters:
    -----------
    true_labels : numpy.ndarray
        True labels
    predictions : numpy.ndarray
        Model predictions
    class_names : list
        Names of the classes
    """
    # Convert predictions to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=class_names))

def visualize_text_predictions(model, tokenizer, texts, true_labels=None, class_names=None):
    """
    Visualize model predictions on text samples.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer
    texts : list
        List of text samples
    true_labels : list, optional
        True labels
    class_names : list, optional
        Names of the classes
    """
    # Tokenize the texts
    encoded_texts = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=TRANSFORMER_CONFIG['max_length'],
        return_tensors='tf'
    )
    
    # Get predictions
    predictions = model.predict({
        'input_ids': encoded_texts['input_ids'],
        'attention_mask': encoded_texts['attention_mask']
    })
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Print results
    for i, text in enumerate(texts):
        pred_class = predicted_classes[i]
        confidence = predictions[i][pred_class] * 100
        
        print(f"Text {i+1}:")
        # Truncate long texts for display
        if len(text) > 100:
            display_text = text[:100] + "..."
        else:
            display_text = text
        print(f"  {display_text}")
        
        if true_labels is not None and class_names is not None:
            true_class = true_labels[i]
            print(f"  True class: {class_names[true_class]}")
        
        if class_names is not None:
            print(f"  Predicted class: {class_names[pred_class]} ({confidence:.2f}% confidence)")
        else:
            print(f"  Predicted class: {pred_class} ({confidence:.2f}% confidence)")
        
        print("-" * 80)

# This module implements Transformer-based models following concepts from deep learning best practices:
# - Using pre-trained transformer models (transfer learning)
# - Fine-tuning approaches with frozen base layers
# - Proper handling of attention masks for variable-length sequences
# - Using the [CLS] token output for classification
