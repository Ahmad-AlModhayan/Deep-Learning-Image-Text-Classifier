"""
Visualization utilities for deep learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import tensorflow as tf

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), normalize=True):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    class_names : list, optional
        Names of the classes
    figsize : tuple
        Figure size
    normalize : bool
        Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curves(y_true, y_scores, class_names=None, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (one-hot encoded)
    y_scores : numpy.ndarray
        Predicted probabilities
    class_names : list, optional
        Names of the classes
    figsize : tuple
        Figure size
    """
    # Convert one-hot encoded y_true to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    n_classes = y_scores.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        # For each class, compute the ROC curve using one-vs-rest approach
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    y_true_binary = tf.keras.utils.to_categorical(y_true, n_classes)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=figsize)
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        class_label = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC for {class_label} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal line for random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall_curves(y_true, y_scores, class_names=None, figsize=(10, 8)):
    """
    Plot Precision-Recall curves for multi-class classification.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (one-hot encoded)
    y_scores : numpy.ndarray
        Predicted probabilities
    class_names : list, optional
        Names of the classes
    figsize : tuple
        Figure size
    """
    # Convert one-hot encoded y_true to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    n_classes = y_scores.shape[1]
    
    # Compute precision-recall curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    for i in range(n_classes):
        # For each class, compute PR curve using one-vs-rest approach
        y_true_binary = (y_true == i).astype(int)
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_scores[:, i])
        avg_precision[i] = np.mean(precision[i])
    
    # Plot precision-recall curves
    plt.figure(figsize=figsize)
    
    # Plot PR curve for each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        class_label = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR for {class_label} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.show()

def plot_learning_curves(history, figsize=(12, 5), include_lr=False):
    """
    Plot learning curves from training history.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    figsize : tuple
        Figure size
    include_lr : bool
        Whether to include learning rate plot
    """
    # Determine number of subplots
    n_plots = 2
    if include_lr and 'lr' in history.history:
        n_plots = 3
    
    plt.figure(figsize=figsize)
    
    # Plot training & validation accuracy
    plt.subplot(1, n_plots, 1)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(alpha=0.3)
    
    # Plot training & validation loss
    plt.subplot(1, n_plots, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(alpha=0.3)
    
    # Plot learning rate if available and requested
    if include_lr and 'lr' in history.history:
        plt.subplot(1, n_plots, 3)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, layer_name, image, figsize=(12, 12), cmap='viridis'):
    """
    Visualize feature maps from a convolutional layer.
    
    Parameters:
    -----------
    model : keras.Model
        The model
    layer_name : str
        Name of the convolutional layer
    image : numpy.ndarray
        Input image (should include batch dimension)
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the feature maps
    """
    # Create a model that outputs the feature maps
    layer = model.get_layer(layer_name)
    feature_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
    
    # Get feature maps
    feature_maps = feature_model.predict(image)
    
    # Plot feature maps
    fig = plt.figure(figsize=figsize)
    
    # Number of feature maps to display
    n_features = min(64, feature_maps.shape[-1])
    size = int(np.ceil(np.sqrt(n_features)))
    
    # Display each feature map
    for i in range(n_features):
        plt.subplot(size, size, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap=cmap)
        plt.axis('off')
    
    plt.suptitle(f'Feature Maps from Layer: {layer_name}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def visualize_embeddings(embeddings, labels, class_names=None, n_components=2, figsize=(10, 8)):
    """
    Visualize embeddings using t-SNE.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Embeddings to visualize
    labels : numpy.ndarray
        Labels for the embeddings
    class_names : list, optional
        Names of the classes
    n_components : int
        Number of dimensions for t-SNE (2 or 3)
    figsize : tuple
        Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    # Apply t-SNE
    print(f"Applying t-SNE with {n_components} components...")
    tsne = TSNE(n_components=n_components, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # Visualize the results
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 2D visualization
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, 
                             cmap='tab10', alpha=0.8, s=50)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    else:
        # 3D visualization
        ax = plt.figure(figsize=figsize).add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2], 
                            c=labels, cmap='tab10', alpha=0.8, s=50)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
    
    # Add legend
    if class_names is not None:
        n_classes = len(class_names)
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names[:n_classes], loc="best", title="Classes")
    
    plt.title(f't-SNE Visualization of {n_components}D Embeddings')
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def display_model_summary(model, print_layer_info=True):
    """
    Display a summary of the model architecture.
    
    Parameters:
    -----------
    model : keras.Model
        The model to summarize
    print_layer_info : bool
        Whether to print detailed info about each layer
    """
    # Print model summary
    model.summary()
    
    if print_layer_info:
        # Print detailed info about each layer
        print("\nLayer Details:")
        print("-" * 80)
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name}")
            print(f"  Type: {layer.__class__.__name__}")
            print(f"  Input Shape: {layer.input_shape}")
            print(f"  Output Shape: {layer.output_shape}")
            print(f"  Parameters: {layer.count_params():,}")
            print(f"  Trainable: {layer.trainable}")
            print("-" * 80)
    
    # Calculate total parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")

# This module follows principles from visualization best practices:
# - Confusion matrices help understand classification errors
# - ROC curves provide insights into classifier performance across thresholds
# - Precision-recall curves are valuable for imbalanced datasets
# - t-SNE helps visualize high-dimensional embeddings
