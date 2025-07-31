"""
Data loader utilities for SmartRetail multimodal emotion analysis.
Handles loading and preprocessing of FER2013 and EmoReact datasets.
"""

import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FacialDataLoader:
    """Data loader for FER2013 facial emotion dataset."""
    
    def __init__(self, config):
        self.config = config
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self.emotions = config["emotions"]
        self.batch_size = config["batch_size"]
        
    def load_fer2013(self, data_path):
        """
        Load FER2013 dataset from CSV file.
        
        Args:
            data_path (str): Path to FER2013 CSV file
            
        Returns:
            tuple: (images, labels) where images are numpy arrays and labels are encoded
        """
        print("Loading FER2013 dataset...")
        
        # Load CSV data
        df = pd.read_csv(data_path)
        
        # Extract images and labels
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            # Convert pixel string to numpy array
            pixels = np.array([int(pixel) for pixel in row['pixels'].split()])
            image = pixels.reshape(48, 48).astype(np.uint8)
            
            # Resize to target size
            image = cv2.resize(image, self.image_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            images.append(image)
            labels.append(row['emotion'])
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        print(f"Labels distribution: {np.bincount(labels_encoded)}")
        
        return images, labels_encoded, label_encoder
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation for training.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Reshape for grayscale
        X_train_reshaped = X_train.reshape(-1, *self.image_size, 1)
        X_val_reshaped = X_val.reshape(-1, *self.image_size, 1)
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train_reshaped, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val_reshaped, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator

class TextDataLoader:
    """Data loader for EmoReact text emotion dataset."""
    
    def __init__(self, config):
        self.config = config
        self.max_sequence_length = config["max_sequence_length"]
        self.vocab_size = config["vocab_size"]
        self.embedding_dim = config["embedding_dim"]
        self.num_classes = config["num_classes"]
        self.emotions = config["emotions"]
        self.batch_size = config["batch_size"]
        self.tokenizer = None
        
    def preprocess_text(self, text):
        """
        Preprocess text data.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_vocabulary(self, texts):
        """
        Create vocabulary from text corpus.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Word to index mapping
        """
        word_counts = {}
        
        for text in texts:
            words = word_tokenize(self.preprocess_text(text))
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocabulary = {'<PAD>': 0, '<UNK>': 1}
        
        for word, count in sorted_words[:self.vocab_size - 2]:
            vocabulary[word] = len(vocabulary)
        
        return vocabulary
    
    def text_to_sequence(self, text, vocabulary):
        """
        Convert text to sequence of indices.
        
        Args:
            text (str): Input text
            vocabulary (dict): Word to index mapping
            
        Returns:
            list: Sequence of indices
        """
        words = word_tokenize(self.preprocess_text(text))
        sequence = []
        
        for word in words:
            if word in vocabulary:
                sequence.append(vocabulary[word])
            else:
                sequence.append(vocabulary['<UNK>'])
        
        return sequence
    
    def load_emoreact(self, data_path):
        """
        Load EmoReact dataset from CSV file.
        
        Args:
            data_path (str): Path to EmoReact CSV file
            
        Returns:
            tuple: (sequences, labels, vocabulary)
        """
        print("Loading EmoReact dataset...")
        
        # Load CSV data
        df = pd.read_csv(data_path)
        
        # Extract texts and labels
        texts = df['text'].tolist()
        labels = df['emotion'].tolist()
        
        # Create vocabulary
        vocabulary = self.create_vocabulary(texts)
        self.tokenizer = vocabulary
        
        # Convert texts to sequences
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text, vocabulary)
            sequences.append(sequence)
        
        # Pad sequences
        sequences_padded = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        print(f"Loaded {len(sequences_padded)} sequences with shape {sequences_padded.shape}")
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"Labels distribution: {np.bincount(labels_encoded)}")
        
        return sequences_padded, labels_encoded, vocabulary, label_encoder
    
    def save_tokenizer(self, save_path):
        """Save tokenizer to file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, load_path):
        """Load tokenizer from file."""
        with open(load_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

class MultimodalDataLoader:
    """Data loader for multimodal emotion analysis."""
    
    def __init__(self, facial_config, text_config):
        self.facial_loader = FacialDataLoader(facial_config)
        self.text_loader = TextDataLoader(text_config)
        
    def load_multimodal_data(self, facial_data_path, text_data_path):
        """
        Load both facial and text datasets.
        
        Args:
            facial_data_path (str): Path to FER2013 CSV
            text_data_path (str): Path to EmoReact CSV
            
        Returns:
            dict: Dictionary containing all data
        """
        # Load facial data
        facial_images, facial_labels, facial_label_encoder = self.facial_loader.load_fer2013(facial_data_path)
        
        # Load text data
        text_sequences, text_labels, vocabulary, text_label_encoder = self.text_loader.load_emoreact(text_data_path)
        
        return {
            'facial_images': facial_images,
            'facial_labels': facial_labels,
            'facial_label_encoder': facial_label_encoder,
            'text_sequences': text_sequences,
            'text_labels': text_labels,
            'vocabulary': vocabulary,
            'text_label_encoder': text_label_encoder
        }
    
    def create_multimodal_dataset(self, data_dict, test_size=0.2, random_state=42):
        """
        Create multimodal dataset with aligned labels.
        
        Args:
            data_dict (dict): Dictionary containing facial and text data
            test_size (float): Fraction of data for testing
            random_state (int): Random seed
            
        Returns:
            dict: Dictionary containing train/test splits
        """
        # For demonstration, we'll create synthetic multimodal data
        # In practice, you'd have paired facial-text data
        
        n_samples = min(len(data_dict['facial_images']), len(data_dict['text_sequences']))
        
        # Create synthetic multimodal pairs
        multimodal_data = {
            'facial_images': data_dict['facial_images'][:n_samples],
            'text_sequences': data_dict['text_sequences'][:n_samples],
            'labels': data_dict['facial_labels'][:n_samples]  # Using facial labels as primary
        }
        
        # Split into train/test
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=multimodal_data['labels']
        )
        
        return {
            'train': {
                'facial_images': multimodal_data['facial_images'][train_indices],
                'text_sequences': multimodal_data['text_sequences'][train_indices],
                'labels': multimodal_data['labels'][train_indices]
            },
            'test': {
                'facial_images': multimodal_data['facial_images'][test_indices],
                'text_sequences': multimodal_data['text_sequences'][test_indices],
                'labels': multimodal_data['labels'][test_indices]
            }
        } 