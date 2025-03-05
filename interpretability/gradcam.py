"""
Grad-CAM implementation for visualizing CNN decisions.

Gradient-weighted Class Activation Mapping (Grad-CAM) produces a heatmap highlighting
important regions in the image for predicting a specific class.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2

from config.config import GRADCAM_CONFIG

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visualizing CNN decisions.
    
    This class implements Grad-CAM as described in the paper:
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    by Selvaraju et al.
    """
    
    def __init__(self, model, layer_name=None, class_idx=None):
        """
        Initialize GradCAM.
        
        Parameters:
        -----------
        model : keras.Model
            The model to explain
        layer_name : str, optional
            Name of the target convolutional layer
        class_idx : int, optional
            Index of the class to explain. If None, the predicted class will be used.
        """
        self.model = model
        self.layer_name = layer_name or GRADCAM_CONFIG['layer_name']
        self.class_idx = class_idx
        
        # Create a model that outputs both the predictions and the activations from target layer
        target_layer = model.get_layer(self.layer_name)
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.output, target_layer.output]
        )
    
    def compute_heatmap(self, img, class_idx=None, alpha=None):
        """
        Compute Grad-CAM heatmap for the given image.
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image (can be a single image or a batch)
        class_idx : int, optional
            Index of the class to explain
        alpha : float, optional
            Transparency factor for the heatmap overlay
            
        Returns:
        --------
        tuple
            (heatmap, superimposed_img, pred_idx, pred_confidence)
        """
        # Ensure the image has batch dimension
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
            
        # Use the provided class index or the predicted class
        class_idx = class_idx or self.class_idx
            
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass through the model
            preds, conv_outputs = self.grad_model(img)
            
            # Get the score for the target class
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            pred_confidence = float(preds[0, class_idx].numpy())
            
            # Get the prediction score for the target class
            class_channel = preds[:, class_idx]
        
        # Gradient of the class output with respect to the conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps with the gradient values
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU to the heatmap
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize the heatmap
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Convert original image to RGB if it's grayscale
        img_squeeze = np.squeeze(img[0])
        if len(img_squeeze.shape) == 2 or img_squeeze.shape[-1] == 1:
            img_rgb = np.repeat(img_squeeze[..., np.newaxis], 3, axis=-1)
        else:
            img_rgb = img_squeeze
        
        # Scale the image to [0, 255]
        img_rgb = np.uint8(255 * img_rgb)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(
            heatmap_colored, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Superimpose the heatmap on the image
        alpha = alpha or GRADCAM_CONFIG['alpha']
        superimposed_img = cv2.addWeighted(
            img_rgb, 1 - alpha, heatmap_resized, alpha, 0)
        
        return heatmap, superimposed_img, class_idx.numpy(), pred_confidence
    
    def visualize(self, img, class_idx=None, alpha=None, class_names=None):
        """
        Compute and visualize Grad-CAM heatmap.
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image
        class_idx : int, optional
            Index of the class to explain
        alpha : float, optional
            Transparency factor for the heatmap overlay
        class_names : list, optional
            Names of the classes
            
        Returns:
        --------
        numpy.ndarray
            Superimposed image
        """
        # Compute heatmap
        heatmap, superimposed_img, pred_idx, confidence = self.compute_heatmap(
            img, class_idx, alpha)
        
        # Convert to RGB for matplotlib
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Display the original image and heatmap overlay
        plt.figure(figsize=(10, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        img_to_show = np.squeeze(img)
        if len(img_to_show.shape) == 2 or img_to_show.shape[-1] == 1:
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(img_to_show)
        
        # Class label
        if class_names is not None:
            title = f"Class: {class_names[pred_idx]}\nConfidence: {confidence:.2f}"
        else:
            title = f"Class: {pred_idx}\nConfidence: {confidence:.2f}"
        plt.title(title)
        plt.axis('off')
        
        # Heatmap overlay
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return superimposed_img

def apply_gradcam_to_dataset(model, images, labels, class_names, num_samples=5, target_layer=None):
    """
    Apply Grad-CAM to a batch of images from a dataset.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    images : numpy.ndarray
        Batch of images
    labels : numpy.ndarray
        True labels (can be one-hot encoded)
    class_names : list
        Names of the classes
    num_samples : int
        Number of samples to visualize
    target_layer : str, optional
        Name of the target convolutional layer
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(labels.shape) > 1:
        true_classes = np.argmax(labels, axis=1)
    else:
        true_classes = labels
    
    # Select random indices
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    # Create GradCAM instance
    gradcam = GradCAM(model, layer_name=target_layer)
    
    # Visualize GradCAM for each selected sample
    for i, idx in enumerate(indices):
        img = images[idx:idx+1]  # Keep batch dimension
        true_class = true_classes[idx]
        
        print(f"\nSample {i+1}: True class = {class_names[true_class]}")
        print("Showing Grad-CAM for predicted class:")
        gradcam.visualize(img, class_names=class_names)
        
        # If the prediction is wrong, also show Grad-CAM for the true class
        pred_class = np.argmax(model.predict(img)[0])
        if pred_class != true_class:
            print(f"Prediction incorrect! Showing Grad-CAM for true class ({class_names[true_class]}):")
            gradcam.visualize(img, class_idx=true_class, class_names=class_names)

def gradcam_explanation_report(model, images, labels, class_names, num_samples=5, target_layer=None):
    """
    Generate a comprehensive explanation report with Grad-CAM visualizations.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    images : numpy.ndarray
        Batch of images
    labels : numpy.ndarray
        True labels (can be one-hot encoded)
    class_names : list
        Names of the classes
    num_samples : int
        Number of samples to visualize
    target_layer : str, optional
        Name of the target convolutional layer
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(labels.shape) > 1:
        true_classes = np.argmax(labels, axis=1)
    else:
        true_classes = labels
    
    # Select random indices
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    # Create GradCAM instance
    gradcam = GradCAM(model, layer_name=target_layer)
    
    # Make predictions
    predictions = model.predict(images[indices])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute overall accuracy on selected samples
    accuracy = np.mean(predicted_classes == true_classes[indices])
    print(f"Accuracy on {num_samples} selected samples: {accuracy:.2f}")
    
    # Display each sample with its Grad-CAM
    for i, idx in enumerate(indices):
        img = images[idx:idx+1]  # Keep batch dimension
        true_class = true_classes[idx]
        pred_class = predicted_classes[i]
        confidence = predictions[i][pred_class] * 100
        
        # Compute Grad-CAM
        _, superimposed_img, _, _ = gradcam.compute_heatmap(img, pred_class)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Display original image and heatmap side by side
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_to_show = np.squeeze(img)
        if len(img_to_show.shape) == 2 or img_to_show.shape[-1] == 1:
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(img_to_show)
        plt.title(f"Original: {class_names[true_class]}")
        plt.axis('off')
        
        # Predicted class heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(superimposed_img)
        result_text = "Correct" if pred_class == true_class else "Incorrect"
        plt.title(f"Prediction: {class_names[pred_class]}\n{confidence:.1f}% ({result_text})")
        plt.axis('off')
        
        # If prediction is wrong, show Grad-CAM for true class
        if pred_class != true_class:
            _, true_class_img, _, _ = gradcam.compute_heatmap(img, true_class)
            true_class_img = cv2.cvtColor(true_class_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, 3, 3)
            plt.imshow(true_class_img)
            plt.title(f"Grad-CAM for true class:\n{class_names[true_class]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print explanation
        print(f"\nSample {i+1}:")
        print(f"True class: {class_names[true_class]}")
        print(f"Predicted class: {class_names[pred_class]} with {confidence:.2f}% confidence")
        
        if pred_class == true_class:
            print("The model correctly identified this image.")
            print("The highlighted regions (in red/yellow) show what the model focused on to make this decision.")
        else:
            print("The model incorrectly identified this image.")
            print("The highlighted regions in the middle image show what the model focused on for its incorrect prediction.")
            print("The right image shows what the model should be focusing on for the correct class.")
        
        print("-" * 80)

# Grad-CAM follows principles from visualization techniques covered in advanced ML courses:
# - Using gradients to understand which parts of the input contribute to the output
# - Creating visual explanations of deep network decisions
# - Helping identify model biases or incorrect reasoning
