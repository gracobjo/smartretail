"""
Collaborative filtering algorithms for recommendation system.
Implements user-based, item-based, and matrix factorization methods.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class UserBasedCF:
    """User-based collaborative filtering."""
    
    def __init__(self, config):
        self.config = config
        self.user_similarity_matrix = None
        self.user_item_matrix = None
        self.n_neighbors = config['n_neighbors']
        self.similarity_metric = config['similarity_metric']
        self.min_similarity = config['min_similarity']
        self.min_common_items = config['min_common_items']
    
    def fit(self, user_item_matrix):
        """
        Fit the user-based collaborative filtering model.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix
        """
        print("Training User-Based Collaborative Filtering...")
        
        self.user_item_matrix = user_item_matrix
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = self._calculate_user_similarity()
        
        print(f"User similarity matrix shape: {self.user_similarity_matrix.shape}")
    
    def _calculate_user_similarity(self):
        """Calculate user similarity matrix."""
        # Remove users with too few ratings
        user_rating_counts = (self.user_item_matrix != 0).sum(axis=1)
        valid_users = user_rating_counts >= self.min_common_items
        
        if not valid_users.any():
            raise ValueError("No users with sufficient ratings")
        
        filtered_matrix = self.user_item_matrix[valid_users]
        
        # Calculate similarity
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(filtered_matrix)
        elif self.similarity_metric == 'euclidean':
            # Convert to distance and then to similarity
            distance_matrix = euclidean_distances(filtered_matrix)
            similarity_matrix = 1 / (1 + distance_matrix)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Filter by minimum similarity
        similarity_matrix[similarity_matrix < self.min_similarity] = 0
        
        return similarity_matrix
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Find similar users
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users = self._get_similar_users(user_idx)
        
        if len(similar_users) == 0:
            return self._get_popular_items(n_recommendations)
        
        # Calculate predicted ratings
        predictions = self._calculate_predictions(user_idx, similar_users)
        
        # Filter out already rated items
        if exclude_rated:
            predictions = {item: score for item, score in predictions.items() 
                         if item not in rated_items}
        
        # Sort by score and return top recommendations
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _get_similar_users(self, user_idx):
        """Get similar users for a given user."""
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_user_indices = np.argsort(user_similarities)[::-1][:self.n_neighbors]
        
        # Filter by minimum similarity
        similar_users = []
        for idx in similar_user_indices:
            if user_similarities[idx] > 0:
                similar_users.append(idx)
        
        return similar_users
    
    def _calculate_predictions(self, user_idx, similar_users):
        """Calculate predicted ratings for items."""
        predictions = {}
        
        for item_id in self.user_item_matrix.columns:
            # Skip items rated by the user
            if self.user_item_matrix.iloc[user_idx, item_id] > 0:
                continue
            
            # Calculate weighted average rating
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_idx in similar_users:
                similarity = self.user_similarity_matrix[user_idx, similar_user_idx]
                rating = self.user_item_matrix.iloc[similar_user_idx, item_id]
                
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions[item_id] = predicted_rating
        
        return predictions
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        item_rating_counts = (self.user_item_matrix != 0).sum(axis=0)
        popular_items = item_rating_counts.nlargest(n_recommendations)
        
        return [(item_id, count) for item_id, count in popular_items.items()]

class ItemBasedCF:
    """Item-based collaborative filtering."""
    
    def __init__(self, config):
        self.config = config
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.n_neighbors = config['n_neighbors']
        self.similarity_metric = config['similarity_metric']
        self.min_similarity = config['min_similarity']
        self.min_common_users = config['min_common_users']
    
    def fit(self, user_item_matrix):
        """
        Fit the item-based collaborative filtering model.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix
        """
        print("Training Item-Based Collaborative Filtering...")
        
        self.user_item_matrix = user_item_matrix
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = self._calculate_item_similarity()
        
        print(f"Item similarity matrix shape: {self.item_similarity_matrix.shape}")
    
    def _calculate_item_similarity(self):
        """Calculate item similarity matrix."""
        # Remove items with too few ratings
        item_rating_counts = (self.user_item_matrix != 0).sum(axis=0)
        valid_items = item_rating_counts >= self.min_common_users
        
        if not valid_items.any():
            raise ValueError("No items with sufficient ratings")
        
        filtered_matrix = self.user_item_matrix.loc[:, valid_items]
        
        # Calculate similarity
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(filtered_matrix.T)
        elif self.similarity_metric == 'euclidean':
            # Convert to distance and then to similarity
            distance_matrix = euclidean_distances(filtered_matrix.T)
            similarity_matrix = 1 / (1 + distance_matrix)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Filter by minimum similarity
        similarity_matrix[similarity_matrix < self.min_similarity] = 0
        
        return similarity_matrix
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.item_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Calculate predicted ratings
        predictions = self._calculate_predictions(user_ratings, rated_items)
        
        # Filter out already rated items
        if exclude_rated:
            predictions = {item: score for item, score in predictions.items() 
                         if item not in rated_items}
        
        # Sort by score and return top recommendations
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _calculate_predictions(self, user_ratings, rated_items):
        """Calculate predicted ratings for items."""
        predictions = {}
        
        for item_id in self.user_item_matrix.columns:
            if item_id in rated_items:
                continue
            
            # Calculate weighted average rating
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item in rated_items:
                item_idx = self.user_item_matrix.columns.get_loc(item_id)
                rated_item_idx = self.user_item_matrix.columns.get_loc(rated_item)
                
                similarity = self.item_similarity_matrix[item_idx, rated_item_idx]
                rating = user_ratings[rated_item]
                
                if similarity > 0 and rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions[item_id] = predicted_rating
        
        return predictions
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        item_rating_counts = (self.user_item_matrix != 0).sum(axis=0)
        popular_items = item_rating_counts.nlargest(n_recommendations)
        
        return [(item_id, count) for item_id, count in popular_items.items()]

class MatrixFactorization:
    """Matrix factorization methods (SVD, NMF)."""
    
    def __init__(self, config, method='svd'):
        self.config = config
        self.method = method
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None
        
        if method == 'svd':
            self.model = TruncatedSVD(
                n_components=config['n_components'],
                random_state=config['random_state']
            )
        elif method == 'nmf':
            self.model = NMF(
                n_components=config['n_components'],
                random_state=config['random_state'],
                max_iter=config['max_iter']
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def fit(self, user_item_matrix):
        """
        Fit the matrix factorization model.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix
        """
        print(f"Training {self.method.upper()} Matrix Factorization...")
        
        self.user_item_matrix = user_item_matrix
        
        # Fit the model
        self.model.fit(user_item_matrix)
        
        # Get factor matrices
        if self.method == 'svd':
            self.user_factors = self.model.transform(user_item_matrix)
            self.item_factors = self.model.components_.T
        else:  # NMF
            self.user_factors = self.model.transform(user_item_matrix)
            self.item_factors = self.model.components_.T
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_items) == 0:
            # Cold start: return most popular items
            return self._get_popular_items(n_recommendations)
        
        # Calculate predicted ratings
        user_vector = self.user_factors[user_idx]
        predictions = np.dot(self.item_factors, user_vector)
        
        # Create item-score pairs
        item_scores = list(enumerate(predictions))
        
        # Filter out already rated items
        if exclude_rated:
            rated_indices = [self.user_item_matrix.columns.get_loc(item) for item in rated_items]
            item_scores = [(idx, score) for idx, score in item_scores if idx not in rated_indices]
        
        # Sort by score and return top recommendations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert indices back to item IDs
        recommendations = []
        for idx, score in item_scores[:n_recommendations]:
            item_id = self.user_item_matrix.columns[idx]
            recommendations.append((item_id, score))
        
        return recommendations
    
    def _get_popular_items(self, n_recommendations):
        """Get most popular items for cold start."""
        item_rating_counts = (self.user_item_matrix != 0).sum(axis=0)
        popular_items = item_rating_counts.nlargest(n_recommendations)
        
        return [(item_id, count) for item_id, count in popular_items.items()]

class CollaborativeFilteringEnsemble:
    """Ensemble of collaborative filtering methods."""
    
    def __init__(self, configs):
        self.configs = configs
        self.models = {}
        self.weights = {}
        
        # Initialize models
        self.models['user_based'] = UserBasedCF(configs['user_based'])
        self.models['item_based'] = ItemBasedCF(configs['item_based'])
        self.models['svd'] = MatrixFactorization(configs['svd'], method='svd')
        self.models['nmf'] = MatrixFactorization(configs['nmf'], method='nmf')
        
        # Set default weights
        self.weights = {
            'user_based': 0.25,
            'item_based': 0.25,
            'svd': 0.25,
            'nmf': 0.25
        }
    
    def fit(self, user_item_matrix):
        """Fit all collaborative filtering models."""
        print("Training Collaborative Filtering Ensemble...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(user_item_matrix)
        
        print("All models trained successfully!")
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate ensemble recommendations.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: List of recommended item IDs with scores
        """
        all_predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                predictions = model.recommend(user_id, n_recommendations * 2, exclude_rated)
                
                # Add weighted scores
                for item_id, score in predictions:
                    if item_id not in all_predictions:
                        all_predictions[item_id] = 0
                    all_predictions[item_id] += score * self.weights[name]
            
            except Exception as e:
                print(f"Warning: {name} failed for user {user_id}: {e}")
                continue
        
        # Sort by ensemble score
        sorted_items = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def set_weights(self, weights):
        """Set ensemble weights."""
        self.weights = weights
    
    def get_model_performance(self, test_matrix):
        """Evaluate individual model performance."""
        performances = {}
        
        for name, model in self.models.items():
            # Simple evaluation: calculate average prediction error
            total_error = 0
            count = 0
            
            for user_id in test_matrix.index:
                for item_id in test_matrix.columns:
                    true_rating = test_matrix.loc[user_id, item_id]
                    if true_rating > 0:
                        try:
                            predictions = model.recommend(user_id, 1, exclude_rated=False)
                            if predictions and predictions[0][0] == item_id:
                                predicted_score = predictions[0][1]
                                error = abs(true_rating - predicted_score)
                                total_error += error
                                count += 1
                        except:
                            continue
            
            if count > 0:
                performances[name] = total_error / count
            else:
                performances[name] = float('inf')
        
        return performances 