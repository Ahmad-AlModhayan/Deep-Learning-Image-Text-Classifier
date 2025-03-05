"""
Text data handling module for loading and preprocessing text datasets.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import seaborn as sns

from config.config import TEXT_CONFIG, TRANSFORMER_CONFIG, BATCH_SIZE, RANDOM_SEED

def load_imdb_dataset(max_features=10000):
    """
    Load the IMDB movie review dataset for sentiment analysis.
    
    Parameters:
    -----------
    max_features : int
        Maximum number of words to consider in the vocabulary
        
    Returns:
    --------
    tuple
        (x_train, y_train, x_test, y_test, word_index)
    """
    print("Loading IMDB dataset...")
    
    # Load data with Keras
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Get the word index mapping
    word_index = imdb.get_word_index()
    
    # Reverse the word index to get words
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    print(f"IMDB dataset loaded: {len(x_train)} training samples, {len(x_test)} test samples")
    
    return x_train, y_train, x_test, y_test, reverse_word_index

def preprocess_keras_imdb(x_train, x_test, maxlen=256):
    """
    Preprocess IMDB data for Keras models.
    
    Parameters:
    -----------
    x_train, x_test : list
        Lists of sequences (integer word indices)
    maxlen : int
        Maximum sequence length
        
    Returns:
    --------
    tuple
        (x_train_padded, x_test_padded)
    """
    print(f"Padding sequences to length {maxlen}...")
    
    # Pad sequences
    x_train_padded = pad_sequences(x_train, maxlen=maxlen)
    x_test_padded = pad_sequences(x_test, maxlen=maxlen)
    
    return x_train_padded, x_test_padded

def preprocess_huggingface_data(x_data, y_data, tokenizer, max_length=None):
    """
    Preprocess text data using a Hugging Face tokenizer.
    
    Parameters:
    -----------
    x_data : list
        List of text samples
    y_data : list
        List of labels
    tokenizer : transformers.PreTrainedTokenizer
        Hugging Face tokenizer
    max_length : int, optional
        Maximum sequence length
        
    Returns:
    --------
    dict
        Tokenized data ready for model input
    """
    if max_length is None:
        max_length = TRANSFORMER_CONFIG['max_length']
    
    print(f"Tokenizing data with max length {max_length}...")
    
    # Tokenize the texts
    encoded_data = tokenizer(
        x_data.tolist() if isinstance(x_data, np.ndarray) else x_data,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    
    return {
        'input_ids': encoded_data['input_ids'],
        'attention_mask': encoded_data['attention_mask'],
        'labels': tf.convert_to_tensor(y_data)
    }

def load_and_prepare_huggingface_data(dataset_name='imdb', test_size=0.2, max_length=None):
    """
    Load and prepare a dataset for use with Hugging Face models.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('imdb')
    test_size : float
        Fraction of data to use for testing
    max_length : int, optional
        Maximum sequence length
        
    Returns:
    --------
    tuple
        (train_dataset, test_dataset, tokenizer, class_names)
    """
    config = TEXT_CONFIG[dataset_name]
    class_names = config['class_names']
    
    if max_length is None:
        max_length = config['max_length']
    
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_CONFIG['model_name'])
    
    if dataset_name == 'imdb':
        # Load IMDB dataset using Keras to get the raw data
        (x_train_indices, y_train), (x_test_indices, y_test) = imdb.load_data()
        
        # Convert indices back to text
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # Indices are offset by 3 in the dataset
        # Index 0 = padding, 1 = start token, 2 = unknown
        index_offset = 3
        
        x_train_texts = [' '.join([reverse_word_index.get(i - index_offset, '?') 
                                   for i in sequence if i >= index_offset]) 
                          for sequence in x_train_indices]
        
        x_test_texts = [' '.join([reverse_word_index.get(i - index_offset, '?') 
                                 for i in sequence if i >= index_offset]) 
                        for sequence in x_test_indices]
        
        # Tokenize the data
        train_dataset = preprocess_huggingface_data(x_train_texts, y_train, tokenizer, max_length)
        test_dataset = preprocess_huggingface_data(x_test_texts, y_test, tokenizer, max_length)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for Hugging Face models")
    
    return train_dataset, test_dataset, tokenizer, class_names

def create_tf_datasets(train_data, test_data, batch_size=None):
    """
    Create TensorFlow datasets from tokenized data.
    
    Parameters:
    -----------
    train_data : dict
        Tokenized training data
    test_data : dict
        Tokenized test data
    batch_size : int, optional
        Batch size
        
    Returns:
    --------
    tuple
        (train_ds, test_ds)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    train_ds = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_data['input_ids'],
            'attention_mask': train_data['attention_mask']
        },
        train_data['labels']
    )).shuffle(10000).batch(batch_size)
    
    test_ds = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': test_data['input_ids'],
            'attention_mask': test_data['attention_mask']
        },
        test_data['labels']
    )).batch(batch_size)
    
    return train_ds, test_ds

def decode_and_visualize_text(tokenized_text, tokenizer, label=None, class_names=None):
    """
    Decode and visualize tokenized text.
    
    Parameters:
    -----------
    tokenized_text : dict
        Tokenized text with input_ids and attention_mask
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to encode the text
    label : int, optional
        The label associated with the text
    class_names : list, optional
        Names of the classes
    """
    input_ids = tokenized_text['input_ids'].numpy()
    
    # Decode each sequence in the batch
    for i, sequence in enumerate(input_ids):
        # Decode the input IDs back to text
        decoded_text = tokenizer.decode(sequence, skip_special_tokens=True)
        
        # Truncate very long texts for display
        if len(decoded_text) > 500:
            decoded_text = decoded_text[:250] + "... [truncated] ..." + decoded_text[-250:]
        
        # Display with label if provided
        if label is not None and class_names is not None:
            print(f"Class: {class_names[label[i]]}")
        
        print(f"Text {i+1}:")
        print(decoded_text)
        print("-" * 80)

def visualize_text_lengths(texts, title="Text Length Distribution"):
    """
    Visualize the distribution of text lengths.
    
    Parameters:
    -----------
    texts : list
        List of text samples
    title : str
        Title for the plot
    """
    # Calculate lengths
    lengths = [len(text.split()) for text in texts]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'length': lengths})
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='length', bins=50)
    plt.title(title)
    plt.xlabel("Text Length (words)")
    plt.ylabel("Frequency")
    plt.axvline(x=np.median(lengths), color='red', linestyle='--', 
                label=f'Median: {np.median(lengths):.0f}')
    plt.axvline(x=np.mean(lengths), color='green', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.0f}')
    plt.legend()
    plt.show()
    
    # Print statistics
    print(f"Text length statistics:")
    print(f"Min: {np.min(lengths)}")
    print(f"Max: {np.max(lengths)}")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")

# This module follows principles from Andrew Ng's Deep Learning Specialization:
# - Text preprocessing involves tokenization and padding to handle variable-length sequences
# - Word embeddings are used to represent text data in a form suitable for neural networks
# - Data visualization helps understand the characteristics of the dataset
