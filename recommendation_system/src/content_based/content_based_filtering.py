"""
Content-based filtering algorithms for recommendation system.
Implements TF-IDF, word embeddings, and product embeddings methods.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

class TFIDFContentBased:
    """Content-based filtering using TF-IDF."""
    
    def __init__(self, config):
        self.config = config
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.item_features = None
        self.similarity_matrix = None
        
    def fit(self, product_features):
        """
        Fit the TF-IDF content-based model.
        
        Args:
            product_features (dict): Product features dictionary
        """
        print("Training TF-IDF Content-Based Filtering...")
        
        if 'text' not in product_features:
            raise ValueError("Text features required for TF-IDF")
        
        # Combine all text features
        combined_texts = []
        for feature_name, feature_data in product_features['text'].items():
            combined_texts.extend(feature_data['corpus'])
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df']
        )
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
    
    def recommend(self, user_profile, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations based on user profile.
        
        Args:
            user_profile (dict): User profile with preferences
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's rated items
        rated_items = user_profile.get('rated_items', [])
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Calculate content-based scores
        item_scores = {}
        
        for item_id in range(self.tfidf_matrix.shape[0]):
            if exclude_rated and item_id in rated_items:
                continue
            
            # Calculate average similarity to rated items
            similarities = []
            for rated_item in rated_items:
                if rated_item < self.similarity_matrix.shape[0]:
                    similarity = self.similarity_matrix[item_id, rated_item]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                item_scores[item_id] = avg_similarity
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        # For TF-IDF, return items with highest average TF-IDF scores
        avg_tfidf_scores = np.mean(self.tfidf_matrix.toarray(), axis=1)
        top_indices = np.argsort(avg_tfidf_scores)[::-1][:n_recommendations]
        
        return [(idx, avg_tfidf_scores[idx]) for idx in top_indices]

class WordEmbeddingsContentBased:
    """Content-based filtering using word embeddings."""
    
    def __init__(self, config):
        self.config = config
        self.word2vec_model = None
        self.item_embeddings = None
        self.similarity_matrix = None
        
    def fit(self, product_features):
        """
        Fit the word embeddings content-based model.
        
        Args:
            product_features (dict): Product features dictionary
        """
        print("Training Word Embeddings Content-Based Filtering...")
        
        if 'text' not in product_features:
            raise ValueError("Text features required for word embeddings")
        
        # Prepare training data
        sentences = []
        for feature_name, feature_data in product_features['text'].items():
            for text in feature_data['corpus']:
                # Tokenize and clean
                tokens = word_tokenize(text.lower())
                tokens = [token for token in tokens if token.isalpha()]
                if tokens:
                    sentences.append(tokens)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=self.config['embedding_dim'],
            window=self.config['window_size'],
            min_count=self.config['min_count'],
            sg=self.config['sg'],
            epochs=self.config['epochs']
        )
        
        # Calculate item embeddings
        self.item_embeddings = self._calculate_item_embeddings(product_features)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.item_embeddings)
        
        print(f"Word2Vec vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        print(f"Item embeddings shape: {self.item_embeddings.shape}")
    
    def _calculate_item_embeddings(self, product_features):
        """Calculate embeddings for each item."""
        item_embeddings = []
        
        for feature_name, feature_data in product_features['text'].items():
            for text in feature_data['corpus']:
                # Tokenize
                tokens = word_tokenize(text.lower())
                tokens = [token for token in tokens if token.isalpha()]
                
                # Calculate average word embedding
                word_vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        word_vectors.append(self.word2vec_model.wv[token])
                
                if word_vectors:
                    item_embedding = np.mean(word_vectors, axis=0)
                else:
                    # Zero vector if no words found
                    item_embedding = np.zeros(self.config['embedding_dim'])
                
                item_embeddings.append(item_embedding)
        
        return np.array(item_embeddings)
    
    def recommend(self, user_profile, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations based on user profile.
        
        Args:
            user_profile (dict): User profile with preferences
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's rated items
        rated_items = user_profile.get('rated_items', [])
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Calculate content-based scores
        item_scores = {}
        
        for item_id in range(self.item_embeddings.shape[0]):
            if exclude_rated and item_id in rated_items:
                continue
            
            # Calculate average similarity to rated items
            similarities = []
            for rated_item in rated_items:
                if rated_item < self.similarity_matrix.shape[0]:
                    similarity = self.similarity_matrix[item_id, rated_item]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                item_scores[item_id] = avg_similarity
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        # Return items with highest average embedding magnitude
        embedding_magnitudes = np.linalg.norm(self.item_embeddings, axis=1)
        top_indices = np.argsort(embedding_magnitudes)[::-1][:n_recommendations]
        
        return [(idx, embedding_magnitudes[idx]) for idx in top_indices]

class ProductEmbeddingsModel:
    """Neural network for learning product embeddings."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.item_embeddings = None
        
    def build_model(self, n_items, n_features):
        """Build the product embeddings model."""
        inputs = layers.Input(shape=(n_features,))
        
        # Hidden layers
        x = inputs
        for units in self.config['hidden_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Embedding layer
        embeddings = layers.Dense(self.config['embedding_dim'], activation='relu')(x)
        
        # Output layer (reconstruction)
        outputs = layers.Dense(n_features, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, product_features):
        """
        Fit the product embeddings model.
        
        Args:
            product_features (dict): Product features dictionary
        """
        print("Training Product Embeddings Model...")
        
        # Prepare feature matrix
        feature_matrix = self._prepare_feature_matrix(product_features)
        
        # Build model
        n_items, n_features = feature_matrix.shape
        self.build_model(n_items, n_features)
        
        # Train model
        self.model.fit(
            feature_matrix,
            feature_matrix,  # Autoencoder
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        # Extract embeddings
        embedding_layer = self.model.layers[-2]  # Second to last layer
        self.item_embeddings = embedding_layer.predict(feature_matrix)
        
        print(f"Product embeddings shape: {self.item_embeddings.shape}")
    
    def _prepare_feature_matrix(self, product_features):
        """Prepare feature matrix from product features."""
        features_list = []
        
        # Categorical features (one-hot encoded)
        if 'categorical' in product_features:
            for feature_name, feature_data in product_features['categorical'].items():
                # One-hot encode categorical features
                n_categories = len(feature_data['unique_values'])
                one_hot = np.zeros((len(feature_data['encoded_values']), n_categories))
                
                for i, encoded_value in enumerate(feature_data['encoded_values']):
                    one_hot[i, encoded_value] = 1
                
                features_list.append(one_hot)
        
        # Numerical features
        if 'numerical' in product_features:
            for feature_name, feature_data in product_features['numerical'].items():
                # Normalize numerical features
                normalized_values = feature_data['scaled_values'].reshape(-1, 1)
                features_list.append(normalized_values)
        
        # Combine all features
        if features_list:
            feature_matrix = np.hstack(features_list)
        else:
            # Fallback: create dummy features
            n_items = max(len(feature_data['encoded_values']) 
                         for feature_data in product_features['categorical'].values())
            feature_matrix = np.random.rand(n_items, 10)
        
        return feature_matrix
    
    def recommend(self, user_profile, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations based on user profile.
        
        Args:
            user_profile (dict): User profile with preferences
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.item_embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's rated items
        rated_items = user_profile.get('rated_items', [])
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Calculate similarity to rated items
        item_scores = {}
        
        for item_id in range(self.item_embeddings.shape[0]):
            if exclude_rated and item_id in rated_items:
                continue
            
            # Calculate average similarity to rated items
            similarities = []
            for rated_item in rated_items:
                if rated_item < self.item_embeddings.shape[0]:
                    similarity = cosine_similarity(
                        [self.item_embeddings[item_id]], 
                        [self.item_embeddings[rated_item]]
                    )[0][0]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                item_scores[item_id] = avg_similarity
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        # Return items with highest average embedding magnitude
        embedding_magnitudes = np.linalg.norm(self.item_embeddings, axis=1)
        top_indices = np.argsort(embedding_magnitudes)[::-1][:n_recommendations]
        
        return [(idx, embedding_magnitudes[idx]) for idx in top_indices]

class ContentBasedEnsemble:
    """Ensemble of content-based filtering methods."""
    
    def __init__(self, configs):
        self.configs = configs
        self.models = {}
        self.weights = {}
        
        # Initialize models
        self.models['tfidf'] = TFIDFContentBased(configs['tfidf'])
        self.models['word_embeddings'] = WordEmbeddingsContentBased(configs['word_embeddings'])
        self.models['product_embeddings'] = ProductEmbeddingsModel(configs['product_embeddings'])
        
        # Set default weights
        self.weights = {
            'tfidf': 0.4,
            'word_embeddings': 0.3,
            'product_embeddings': 0.3
        }
    
    def fit(self, product_features):
        """Fit all content-based models."""
        print("Training Content-Based Filtering Ensemble...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(product_features)
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                self.weights[name] = 0  # Disable failed model
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print("Content-based models trained successfully!")
    
    def recommend(self, user_profile, n_recommendations=10, exclude_rated=True):
        """
        Generate ensemble recommendations.
        
        Args:
            user_profile (dict): User profile with preferences
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        all_predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            if self.weights[name] > 0:  # Skip disabled models
                try:
                    predictions = model.recommend(user_profile, n_recommendations * 2, exclude_rated)
                    
                    # Add weighted scores
                    for item_id, score in predictions:
                        if item_id not in all_predictions:
                            all_predictions[item_id] = 0
                        all_predictions[item_id] += score * self.weights[name]
                
                except Exception as e:
                    print(f"Warning: {name} failed for user: {e}")
                    continue
        
        # Sort by ensemble score
        sorted_items = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def set_weights(self, weights):
        """Set ensemble weights."""
        self.weights = weights 