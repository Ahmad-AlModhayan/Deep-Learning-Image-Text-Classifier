"""
Main entry point for the deep learning classifier with interpretability project.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import project modules
from config.config import (
    IMAGE_CONFIG, TEXT_CONFIG, TRANSFORMER_CONFIG, 
    GRADCAM_CONFIG, SHAP_CONFIG, NUM_EPOCHS, BATCH_SIZE
)
from data.image_data import (
    load_dataset as load_image_dataset,
    create_data_generators,
    visualize_dataset_samples, 
    visualize_augmented_samples
)
from data.text_data import (
    load_and_prepare_huggingface_data,
    create_tf_datasets,
    decode_and_visualize_text
)
from models.image_model import (
    create_cnn_model,
    train_model as train_image_model,
    evaluate_model as evaluate_image_model,
    visualize_training_history as visualize_image_training,
    predict_batch,
    visualize_predictions
)
from models.text_model import (
    create_transformer_model,
    train_model as train_text_model,
    evaluate_model as evaluate_text_model,
    visualize_training_history as visualize_text_training,
    visualize_confusion_matrix
)
from interpretability.gradcam import (
    GradCAM, 
    apply_gradcam_to_dataset,
    gradcam_explanation_report
)
from interpretability.shap_explainer import (
    ShapTextExplainer, 
    shap_explanation_report
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves
)
from utils.metrics import (
    calculate_metrics,
    print_classification_report,
    plot_confusion_matrix_with_metrics
)

def setup_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs('saved_models', exist_ok=True)
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Learning Classifier with Interpretability')
    
    # Main arguments
    parser.add_argument('--mode', type=str, choices=['image', 'text'], default='image',
                        help='Mode: image classification or text classification')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to use (cifar10, mnist for image; imdb for text)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs for training (default: {NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    
    # Training and evaluation options
    parser.add_argument('--train', action='store_true',
                        help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model')
    parser.add_argument('--visualize_data', action='store_true',
                        help='Visualize dataset samples')
    
    # Model options
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model')
    
    # Interpretability options
    parser.add_argument('--interpret', action='store_true',
                        help='Generate model interpretability visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to use for interpretability visualization')
    
    return parser.parse_args()

def image_classification_pipeline(args):
    """Run the image classification pipeline."""
    print(f"\n{'='*80}")
    print(f"Image Classification Pipeline: {args.dataset.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    x_train, y_train, x_test, y_test, class_names = load_image_dataset(args.dataset)
    
    # Visualize dataset if requested
    if args.visualize_data:
        print("\nVisualizing dataset samples...")
        visualize_dataset_samples(x_train, y_train, class_names)
        visualize_augmented_samples(x_train, y_train, class_names)
    
    # Create data generators
    train_generator, test_generator = create_data_generators(
        x_train, y_train, x_test, y_test)
    
    # Get input shape from the dataset
    input_shape = x_train.shape[1:]
    
    if args.train:
        # Create and train model
        print("\nCreating and training model...")
        model = create_cnn_model(args.dataset, input_shape)
        
        # Train the model
        history = train_image_model(
            model=model,
            train_generator=train_generator,
            validation_data=(x_test, y_test),
            epochs=args.epochs,
            model_name=f"{args.dataset}_cnn"
        )
        
        # Visualize training history
        visualize_image_training(history)
        
        # Save the model
        model_save_path = f"saved_models/{args.dataset}_cnn.h5"
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    elif args.model_path:
        # Load the saved model
        print(f"\nLoading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
    else:
        # Create a new model without training
        print("\nCreating model (not training)...")
        model = create_cnn_model(args.dataset, input_shape)
    
    if args.evaluate:
        # Evaluate model
        print("\nEvaluating model...")
        loss, accuracy = evaluate_image_model(model, test_generator)
        
        # Make predictions on test data
        print("\nGenerating predictions...")
        predictions, predicted_classes = predict_batch(model, x_test, class_names)
        
        # Visualize predictions
        visualize_predictions(x_test, y_test, predictions, class_names)
        
        # Print classification report
        print_classification_report(y_test, predictions, class_names)
        
        # Plot confusion matrix with metrics
        plot_confusion_matrix_with_metrics(
            np.argmax(y_test, axis=1), 
            np.argmax(predictions, axis=1),
            class_names
        )
        
        # Plot ROC curves
        plot_roc_curves(y_test, predictions, class_names)
        
        # Plot precision-recall curves
        plot_precision_recall_curves(y_test, predictions, class_names)
    
    if args.interpret:
        # Apply Grad-CAM to test samples
        print("\nGenerating Grad-CAM visualizations...")
        # Find the last convolutional layer name
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
        if conv_layers:
            target_layer = conv_layers[-1]
            print(f"Using {target_layer} as target layer for Grad-CAM")
            
            # Generate explanations
            gradcam_explanation_report(
                model, x_test, y_test, class_names, 
                num_samples=args.num_samples,
                target_layer=target_layer
            )
        else:
            print("No convolutional layers found in the model. Cannot apply Grad-CAM.")

def text_classification_pipeline(args):
    """Run the text classification pipeline."""
    print(f"\n{'='*80}")
    print(f"Text Classification Pipeline: {args.dataset.upper()}")
    print(f"{'='*80}")
    
    # Load and prepare data
    train_dataset, test_dataset, tokenizer, class_names = load_and_prepare_huggingface_data(
        dataset_name=args.dataset
    )
    
    # Create TF datasets
    train_ds, test_ds = create_tf_datasets(train_dataset, test_dataset, args.batch_size)
    
    # Visualize some examples if requested
    if args.visualize_data:
        print("\nVisualizing dataset samples...")
        # Get a batch from the test dataset
        for inputs, labels in test_ds.take(1):
            decode_and_visualize_text(inputs, tokenizer, labels.numpy(), class_names)
    
    if args.train:
        # Create and train model
        print("\nCreating and training transformer model...")
        model, _ = create_transformer_model(
            model_name=TRANSFORMER_CONFIG['model_name'],
            dataset_name=args.dataset
        )
        
        # Train the model
        history = train_text_model(
            model=model,
            train_dataset=train_ds,
            val_dataset=test_ds,
            epochs=args.epochs,
            model_name=f"{args.dataset}_transformer"
        )
        
        # Visualize training history
        visualize_text_training(history)
        
        # Save the model
        model_save_path = f"saved_models/{args.dataset}_transformer"
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    elif args.model_path:
        # Load the saved model
        print(f"\nLoading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
    else:
        # Create a new model without training
        print("\nCreating model (not training)...")
        model, _ = create_transformer_model(
            model_name=TRANSFORMER_CONFIG['model_name'],
            dataset_name=args.dataset
        )
    
    if args.evaluate:
        # Evaluate model
        print("\nEvaluating model...")
        loss, accuracy = evaluate_text_model(model, test_ds)
        
        # Get predictions and true labels from the test dataset
        all_predictions = []
        all_labels = []
        
        for inputs, labels in test_ds:
            predictions = model.predict(inputs)
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Print classification report
        print_classification_report(all_labels, all_predictions, class_names)
        
        # Plot confusion matrix
        visualize_confusion_matrix(all_labels, np.argmax(all_predictions, axis=1), class_names)
        
        # Plot confusion matrix with metrics
        plot_confusion_matrix_with_metrics(
            all_labels, 
            np.argmax(all_predictions, axis=1),
            class_names
        )
    
    if args.interpret:
        print("\nGenerating SHAP explanations...")
        
        # Create a SHAP explainer
        explainer = ShapTextExplainer(model, tokenizer, class_names)
        
        # Get a few samples from the test dataset for interpretation
        sample_texts = []
        sample_labels = []
        
        # Extract some test samples
        for inputs, labels in test_ds.take(1):
            input_ids = inputs['input_ids'].numpy()
            attention_mask = inputs['attention_mask'].numpy()
            sample_labels.extend(labels.numpy())
            
            # Decode the input IDs back to text
            for i in range(min(args.num_samples, len(input_ids))):
                # Skip padding tokens
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                sample_texts.append(text)
        
        sample_labels = sample_labels[:len(sample_texts)]
        
        # Generate explanations
        shap_explanation_report(
            model, tokenizer, sample_texts[:args.num_samples], 
            sample_labels[:args.num_samples], class_names
        )

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create necessary directories
    setup_dirs()
    
    # Print arguments
    print("\nArguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Run pipeline based on mode
    if args.mode == 'image':
        image_classification_pipeline(args)
    elif args.mode == 'text':
        text_classification_pipeline(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main()
