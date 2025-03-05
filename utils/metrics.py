"""
Evaluation metrics utilities for deep learning models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

def calculate_metrics(y_true, y_pred, y_scores=None, average='weighted'):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    y_scores : numpy.ndarray, optional
        Predicted probabilities for each class
    average : str
        Method for averaging in multi-class scenarios ('micro', 'macro', 'weighted')
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Convert one-hot encoded y_true and y_pred to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Calculate AUC-ROC if scores are provided
    if y_scores is not None:
        # For binary classification
        if y_scores.shape[1] == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores[:, 1])
        # For multi-class classification
        else:
            try:
                y_true_onehot = tf.keras.utils.to_categorical(y_true, y_scores.shape[1])
                metrics['auc_roc'] = roc_auc_score(y_true_onehot, y_scores, average=average, multi_class='ovr')
            except Exception as e:
                print(f"Could not calculate AUC-ROC: {e}")
    
    return metrics

def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print classification report with class names.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    class_names : list, optional
        Names of the classes
        
    Returns:
    --------
    str
        Classification report as string
    """
    # Convert one-hot encoded y_true and y_pred to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Create the classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names if class_names else None,
        digits=3
    )
    
    print("Classification Report:")
    print(report)
    
    return report

def plot_confusion_matrix_with_metrics(y_true, y_pred, class_names=None, figsize=(10, 8), normalize=True):
    """
    Plot confusion matrix with metrics summary.
    
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
        
    Returns:
    --------
    tuple
        (metrics_summary, confusion_matrix)
    """
    # Convert one-hot encoded y_true and y_pred to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate per-class metrics
    metrics_per_class = {}
    for cls in np.unique(y_true):
        cls_name = class_names[cls] if class_names else f"Class {cls}"
        
        # Create binary labels for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        # Calculate metrics for this class
        metrics_per_class[cls_name] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }
    
    # Calculate overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
                ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Create metrics summary table
    metrics_df = pd.DataFrame(metrics_per_class).T
    metrics_df.columns = ['Precision', 'Recall', 'F1-Score']
    
    # Add overall metrics
    metrics_df.loc['Overall (Accuracy)', :] = [overall_metrics['accuracy'], overall_metrics['accuracy'], overall_metrics['accuracy']]
    metrics_df.loc['Overall (Macro)', :] = [overall_metrics['macro_precision'], overall_metrics['macro_recall'], overall_metrics['macro_f1']]
    metrics_df.loc['Overall (Weighted)', :] = [overall_metrics['weighted_precision'], overall_metrics['weighted_recall'], overall_metrics['weighted_f1']]
    
    # Plot metrics table
    ax2.axis('off')
    ax2.table(cellText=metrics_df.values.round(3),
              rowLabels=metrics_df.index,
              colLabels=metrics_df.columns,
              cellLoc='center',
              loc='center',
              bbox=[0, 0, 1, 1])
    ax2.set_title('Metrics Summary')
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df, cm

def calculate_per_class_accuracy(y_true, y_pred, class_names=None):
    """
    Calculate per-class accuracy.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    class_names : list, optional
        Names of the classes
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with per-class accuracy
    """
    # Convert one-hot encoded y_true and y_pred to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Initialize results
    results = []
    
    # Calculate accuracy for each class
    for cls in unique_classes:
        # Get indices where true label is the current class
        indices = (y_true == cls)
        
        # Skip if no instances of this class
        if sum(indices) == 0:
            continue
        
        # Calculate accuracy for this class
        class_accuracy = accuracy_score(y_true[indices], y_pred[indices])
        
        # Get class name
        class_name = class_names[cls] if class_names else f"Class {cls}"
        
        # Store results
        results.append({
            'Class': class_name,
            'Accuracy': class_accuracy,
            'Support': sum(indices)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy (descending)
    df = df.sort_values('Accuracy', ascending=False)
    
    # Add overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    df = df.append({
        'Class': 'Overall',
        'Accuracy': overall_accuracy,
        'Support': len(y_true)
    }, ignore_index=True)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot per-class accuracy bars
    bar_plot = sns.barplot(x='Class', y='Accuracy', data=df[:-1], palette='viridis')
    
    # Add value labels on top of the bars
    for i, p in enumerate(bar_plot.patches):
        bar_plot.annotate(f'{p.get_height():.3f}',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom', rotation=0,
                          xytext=(0, 5), textcoords='offset points')
    
    # Add a horizontal line for overall accuracy
    plt.axhline(y=overall_accuracy, color='r', linestyle='--', 
                label=f'Overall Accuracy: {overall_accuracy:.3f}')
    
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df

def calibration_curve_plot(y_true, y_scores, n_bins=10, figsize=(10, 8)):
    """
    Plot calibration curve to assess model calibration.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (binary or class indices)
    y_scores : numpy.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for the calibration curve
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple
        (bins, accuracies)
    """
    # For binary classification
    if len(y_scores.shape) > 1 and y_scores.shape[1] == 2:
        y_scores = y_scores[:, 1]
    # For multi-class, use the max probability
    elif len(y_scores.shape) > 1 and y_scores.shape[1] > 2:
        y_pred = np.argmax(y_scores, axis=1)
        y_scores = np.max(y_scores, axis=1)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        y_true = (y_true == y_pred).astype(int)
    
    # Create bins and calculate accuracy in each bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_scores, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_sums = np.bincount(bin_indices, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_prob = np.bincount(bin_indices, weights=y_scores, minlength=n_bins)
    
    # Calculate mean predicted probability and accuracy in each bin
    nonzero = bin_sums > 0
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    
    prob_true[nonzero] = bin_true[nonzero] / bin_sums[nonzero]
    prob_pred[nonzero] = bin_prob[nonzero] / bin_sums[nonzero]
    
    # Plot calibration curve
    plt.figure(figsize=figsize)
    
    # Plot the calibration curve
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration curve')
    
    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    
    # Calculate the Brier score (mean squared error)
    brier_score = np.mean((y_scores - y_true) ** 2)
    
    # Calculate the calibration error
    cal_error = np.mean(np.abs(prob_pred - prob_true))
    
    plt.title(f'Calibration Curve\nBrier Score: {brier_score:.4f}, Calibration Error: {cal_error:.4f}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Plot the histogram of predicted probabilities
    ax2 = plt.gca().twinx()
    ax2.hist(y_scores, range=(0, 1), bins=n_bins, histtype='step', lw=2, color='r')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)
    
    plt.tight_layout()
    plt.show()
    
    return prob_pred, prob_true

# Implementation follows principles from evaluation metrics in machine learning:
# - Using F1 score to balance precision and recall
# - Confusion matrices to understand error patterns
# - Per-class metrics to identify problematic classes
# - Calibration curves to assess probability estimates
