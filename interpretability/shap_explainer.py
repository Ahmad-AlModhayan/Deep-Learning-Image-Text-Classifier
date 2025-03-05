"""
SHAP (SHapley Additive exPlanations) implementation for explaining text model predictions.

SHAP values attribute to each feature (token in text) how much it contributed to a
particular prediction, based on cooperative game theory.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display

from config.config import SHAP_CONFIG

class ShapTextExplainer:
    """
    SHAP explainer for transformer-based text models.
    
    This class implements SHAP (SHapley Additive exPlanations) for text models
    to explain which tokens contribute most to the model's predictions.
    """
    
    def __init__(self, model, tokenizer, class_names=None):
        """
        Initialize ShapTextExplainer.
        
        Parameters:
        -----------
        model : keras.Model
            The model to explain
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer used for the model
        class_names : list, optional
            Names of the classes
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.explainer = None
    
    def _model_predict(self, inputs):
        """
        Model prediction function for SHAP.
        
        Parameters:
        -----------
        inputs : list
            List of texts
            
        Returns:
        --------
        numpy.ndarray
            Model predictions
        """
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            inputs,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='tf'
        )
        
        # Get predictions
        predictions = self.model.predict({
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask']
        })
        
        return predictions
    
    def create_explainer(self, background_data=None, num_samples=None):
        """
        Create SHAP explainer.
        
        Parameters:
        -----------
        background_data : list, optional
            Background data for the explainer
        num_samples : int, optional
            Number of samples to use for the explainer
            
        Returns:
        --------
        shap.Explainer
            SHAP explainer
        """
        # Use default parameters if not provided
        if num_samples is None:
            num_samples = SHAP_CONFIG['num_samples']
        
        # If no background data provided, create empty background
        if background_data is None:
            background_data = ['']
            
        print(f"Creating SHAP explainer with {num_samples} samples...")
        
        # Create explainer
        self.explainer = shap.Explainer(self._model_predict, background_data)
        
        return self.explainer
    
    def explain_text(self, text, target_class=None, explain_class=None):
        """
        Explain model prediction for a single text.
        
        Parameters:
        -----------
        text : str
            Text to explain
        target_class : int, optional
            Target class index
        explain_class : int, optional
            Class to explain
            
        Returns:
        --------
        shap.Explanation
            SHAP explanation
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Get prediction
        prediction = self._model_predict([text])[0]
        predicted_class = np.argmax(prediction)
        
        # Use predicted class if not specified
        if explain_class is None:
            explain_class = predicted_class if target_class is None else target_class
        
        # Get explanation
        shap_values = self.explainer([text])
        
        # Display results
        print(f"Text: {text}")
        if self.class_names is not None:
            print(f"Predicted class: {self.class_names[predicted_class]} ({prediction[predicted_class]:.4f})")
            if target_class is not None:
                print(f"Target class: {self.class_names[target_class]} ({prediction[target_class]:.4f})")
            print(f"Explaining contribution to class: {self.class_names[explain_class]}")
        else:
            print(f"Predicted class: {predicted_class} ({prediction[predicted_class]:.4f})")
            if target_class is not None:
                print(f"Target class: {target_class} ({prediction[target_class]:.4f})")
            print(f"Explaining contribution to class: {explain_class}")
        
        # Visualize the explanation
        plt.figure(figsize=(18, 4))
        shap.plots.text(shap_values[:, :, explain_class])
        plt.tight_layout()
        plt.show()
        
        return shap_values
    
    def explain_batch(self, texts, target_classes=None, top_n=None):
        """
        Explain model predictions for a batch of texts.
        
        Parameters:
        -----------
        texts : list
            List of texts to explain
        target_classes : list, optional
            List of target class indices
        top_n : int, optional
            Number of top tokens to show
            
        Returns:
        --------
        list
            List of SHAP explanations
        """
        if self.explainer is None:
            self.create_explainer()
        
        if top_n is None:
            top_n = SHAP_CONFIG['top_n_tokens']
        
        # Get predictions
        predictions = self._model_predict(texts)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Use predicted classes if target classes not provided
        if target_classes is None:
            target_classes = predicted_classes
            
        # Get explanations
        shap_values = self.explainer(texts)
        
        all_explanations = []
        
        # Display results for each text
        for i, text in enumerate(texts):
            # Get token-wise SHAP values for the target class
            token_shap_values = shap_values[i, :, target_classes[i]]
            tokens = shap_values.data[i].split()
            
            # Match SHAP values to tokens (handling potential length mismatch)
            max_len = min(len(tokens), len(token_shap_values))
            token_contributions = list(zip(tokens[:max_len], token_shap_values[:max_len]))
            
            # Sort by absolute SHAP value
            token_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Display only top N tokens
            top_tokens = token_contributions[:top_n]
            
            print(f"\nText {i+1}:")
            print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
            
            if self.class_names is not None:
                pred_class = predicted_classes[i]
                target_class = target_classes[i]
                confidence = predictions[i][pred_class] * 100
                print(f"Predicted class: {self.class_names[pred_class]} ({confidence:.2f}%)")
                print(f"Explaining contribution to class: {self.class_names[target_class]}")
            else:
                pred_class = predicted_classes[i]
                target_class = target_classes[i]
                confidence = predictions[i][pred_class] * 100
                print(f"Predicted class: {pred_class} ({confidence:.2f}%)")
                print(f"Explaining contribution to class: {target_class}")
            
            print("\nTop tokens contributing to prediction:")
            for token, value in top_tokens:
                direction = "POSITIVE" if value > 0 else "NEGATIVE"
                print(f"  {token}: {value:.4f} ({direction})")
            
            all_explanations.append({
                'text': text,
                'prediction': pred_class,
                'confidence': predictions[i][pred_class],
                'target_class': target_class,
                'top_tokens': top_tokens
            })
        
        return all_explanations
    
    def visualize_token_contribution(self, text, target_class=None, max_tokens=50):
        """
        Visualize token contributions for a text.
        
        Parameters:
        -----------
        text : str
            Text to explain
        target_class : int, optional
            Target class index
        max_tokens : int
            Maximum number of tokens to visualize
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with token contributions
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Get prediction
        prediction = self._model_predict([text])[0]
        predicted_class = np.argmax(prediction)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = predicted_class
        
        # Get explanation
        shap_values = self.explainer([text])
        
        # Get token-wise SHAP values for the target class
        token_shap_values = shap_values[0, :, target_class]
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Match SHAP values to tokens (handling potential length mismatch)
        max_len = min(len(tokens), len(token_shap_values), max_tokens)
        token_contributions = list(zip(tokens[:max_len], token_shap_values[:max_len]))
        
        # Create DataFrame
        df = pd.DataFrame(token_contributions, columns=['Token', 'SHAP Value'])
        
        # Sort by absolute SHAP value
        df['Abs SHAP Value'] = df['SHAP Value'].abs()
        df = df.sort_values('Abs SHAP Value', ascending=False).head(max_tokens)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar chart
        sns.barplot(x='SHAP Value', y='Token', data=df, 
                   palette=['red' if x < 0 else 'blue' for x in df['SHAP Value']])
        
        # Add title and labels
        if self.class_names is not None:
            plt.title(f"Top tokens contributing to class: {self.class_names[target_class]}")
        else:
            plt.title(f"Top tokens contributing to class: {target_class}")
        
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.ylabel('Token')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return df
    
    def generate_interactive_explanation(self, text, target_class=None):
        """
        Generate an interactive HTML visualization of token contributions.
        
        Parameters:
        -----------
        text : str
            Text to explain
        target_class : int, optional
            Target class index
            
        Returns:
        --------
        IPython.display.HTML
            Interactive HTML visualization
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Get prediction
        prediction = self._model_predict([text])[0]
        predicted_class = np.argmax(prediction)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = predicted_class
        
        # Get explanation
        shap_values = self.explainer([text])
        
        # Get token-wise SHAP values for the target class
        token_shap_values = shap_values[0, :, target_class].numpy()
        
        # Get individual tokens (approximate using split - this is simplified)
        tokens = text.split()
        token_count = min(len(tokens), len(token_shap_values))
        
        # Normalize SHAP values for color intensity
        max_abs_shap = max(abs(token_shap_values[:token_count]))
        if max_abs_shap > 0:
            normalized_shap = token_shap_values[:token_count] / max_abs_shap
        else:
            normalized_shap = token_shap_values[:token_count]
        
        # Generate HTML with colored spans
        html = "<div style='font-size: 16px; line-height: 1.5;'>"
        
        for i in range(token_count):
            shap_value = token_shap_values[i]
            normalized_value = normalized_shap[i]
            
            # Determine color based on SHAP value (red for negative, blue for positive)
            if shap_value >= 0:
                color = f"rgba(0, 0, 255, {abs(normalized_value) * 0.7})"
            else:
                color = f"rgba(255, 0, 0, {abs(normalized_value) * 0.7})"
            
            # Create span with tooltip
            span = f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;' title='SHAP: {shap_value:.4f}'>{tokens[i]}</span> "
            html += span
        
        html += "</div>"
        
        # Add legend
        html += """
        <div style='margin-top: 20px;'>
            <p style='font-size: 14px;'>
                <span style='background-color: rgba(0, 0, 255, 0.7); padding: 2px; border-radius: 3px;'>Blue</span>: 
                Positive contribution (increases probability of the class)
            </p>
            <p style='font-size: 14px;'>
                <span style='background-color: rgba(255, 0, 0, 0.7); padding: 2px; border-radius: 3px;'>Red</span>: 
                Negative contribution (decreases probability of the class)
            </p>
            <p style='font-size: 14px;'>
                Color intensity indicates the magnitude of the contribution.
            </p>
        </div>
        """
        
        # Display prediction results
        html += "<div style='margin-top: 20px; font-size: 14px;'>"
        if self.class_names is not None:
            html += f"<p>Predicted class: <b>{self.class_names[predicted_class]}</b> with {prediction[predicted_class]:.2%} confidence</p>"
            html += f"<p>Explaining contribution to class: <b>{self.class_names[target_class]}</b></p>"
        else:
            html += f"<p>Predicted class: <b>{predicted_class}</b> with {prediction[predicted_class]:.2%} confidence</p>"
            html += f"<p>Explaining contribution to class: <b>{target_class}</b></p>"
        html += "</div>"
        
        return HTML(html)

def shap_explanation_report(model, tokenizer, texts, true_labels=None, class_names=None):
    """
    Generate comprehensive SHAP explanation report for multiple texts.
    
    Parameters:
    -----------
    model : keras.Model
        The trained model
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer
    texts : list
        List of texts to explain
    true_labels : list, optional
        List of true labels
    class_names : list, optional
        Names of the classes
    """
    # Create explainer
    explainer = ShapTextExplainer(model, tokenizer, class_names)
    explainer.create_explainer()
    
    # Get predictions
    predictions = explainer._model_predict(texts)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Display results
    for i, text in enumerate(texts):
        pred_class = predicted_classes[i]
        confidence = predictions[i][pred_class] * 100
        
        print(f"\n{'-' * 80}")
        print(f"Text {i+1}:")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        if class_names is not None:
            print(f"Predicted class: {class_names[pred_class]} ({confidence:.2f}%)")
            if true_labels is not None:
                true_class = true_labels[i]
                print(f"True class: {class_names[true_class]}")
                print(f"Prediction {'CORRECT' if pred_class == true_class else 'INCORRECT'}")
        else:
            print(f"Predicted class: {pred_class} ({confidence:.2f}%)")
            if true_labels is not None:
                true_class = true_labels[i]
                print(f"True class: {true_class}")
                print(f"Prediction {'CORRECT' if pred_class == true_class else 'INCORRECT'}")
        
        # Generate token contribution visualization
        print("\nToken Contribution Analysis:")
        explainer.visualize_token_contribution(text, pred_class)
        
        # Show interactive explanation
        print("\nDetailed Word-Level Explanation:")
        display(explainer.generate_interactive_explanation(text, pred_class))
        
        # If prediction is incorrect and true labels are provided, also explain true class
        if true_labels is not None and pred_class != true_labels[i]:
            print(f"\nExplanation for why the model DIDN'T predict the true class ({class_names[true_labels[i]] if class_names else true_labels[i]}):")
            explainer.visualize_token_contribution(text, true_labels[i])

# SHAP follows the principles of model interpretability from advanced ML courses:
# - Attributing importance to individual features (tokens) in the prediction
# - Based on cooperative game theory principles
# - Providing both global and local explanations of model behavior
