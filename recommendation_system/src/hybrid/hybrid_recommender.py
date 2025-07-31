"""
Hybrid recommendation system combining collaborative and content-based filtering.
Implements various fusion strategies for optimal recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class WeightedHybrid:
    """Weighted combination of collaborative and content-based recommendations."""
    
    def __init__(self, config):
        self.config = config
        self.collaborative_weight = config['collaborative_weight']
        self.content_weight = config['content_weight']
        self.dynamic_weighting = config['dynamic_weighting']
        
    def recommend(self, collaborative_recs, content_recs, user_profile=None):
        """
        Combine recommendations using weighted fusion.
        
        Args:
            collaborative_recs (list): Collaborative filtering recommendations
            content_recs (list): Content-based recommendations
            user_profile (dict): User profile for dynamic weighting
            
        Returns:
            list: Combined recommendations
        """
        # Convert to dictionaries for easier processing
        collab_dict = dict(collaborative_recs)
        content_dict = dict(content_recs)
        
        # Get all unique items
        all_items = set(collab_dict.keys()) | set(content_dict.keys())
        
        # Calculate weights
        if self.dynamic_weighting and user_profile:
            weights = self._calculate_dynamic_weights(user_profile)
        else:
            weights = {
                'collaborative': self.collaborative_weight,
                'content': self.content_weight
            }
        
        # Combine scores
        combined_scores = {}
        for item in all_items:
            collab_score = collab_dict.get(item, 0)
            content_score = content_dict.get(item, 0)
            
            combined_score = (weights['collaborative'] * collab_score + 
                           weights['content'] * content_score)
            combined_scores[item] = combined_score
        
        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items
    
    def _calculate_dynamic_weights(self, user_profile):
        """Calculate dynamic weights based on user profile."""
        interaction_count = user_profile.get('rating_count', 0)
        
        # More interactions -> higher collaborative weight
        if interaction_count < 10:
            collab_weight = 0.3
            content_weight = 0.7
        elif interaction_count < 50:
            collab_weight = 0.5
            content_weight = 0.5
        else:
            collab_weight = 0.7
            content_weight = 0.3
        
        return {
            'collaborative': collab_weight,
            'content': content_weight
        }

class SwitchingHybrid:
    """Switching between collaborative and content-based based on confidence."""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config['confidence_threshold']
        self.fallback_method = config['fallback_method']
    
    def recommend(self, collaborative_recs, content_recs, user_profile=None):
        """
        Switch between methods based on confidence.
        
        Args:
            collaborative_recs (list): Collaborative filtering recommendations
            content_recs (list): Content-based recommendations
            user_profile (dict): User profile for confidence calculation
            
        Returns:
            list: Selected recommendations
        """
        # Calculate confidence for each method
        collab_confidence = self._calculate_confidence(collaborative_recs, user_profile, 'collaborative')
        content_confidence = self._calculate_confidence(content_recs, user_profile, 'content')
        
        # Choose method with higher confidence
        if collab_confidence > content_confidence and collab_confidence > self.confidence_threshold:
            return collaborative_recs
        elif content_confidence > self.confidence_threshold:
            return content_recs
        else:
            # Fallback to specified method
            if self.fallback_method == 'collaborative':
                return collaborative_recs
            else:
                return content_recs
    
    def _calculate_confidence(self, recommendations, user_profile, method_type):
        """Calculate confidence score for recommendations."""
        if not recommendations:
            return 0.0
        
        # Base confidence on recommendation scores
        scores = [score for _, score in recommendations]
        avg_score = np.mean(scores)
        score_variance = np.var(scores)
        
        # Higher average score and lower variance = higher confidence
        confidence = avg_score * (1 - score_variance)
        
        # Adjust based on user profile
        if user_profile:
            interaction_count = user_profile.get('rating_count', 0)
            
            if method_type == 'collaborative':
                # More interactions = higher confidence for collaborative
                confidence *= min(1.0, interaction_count / 50)
            else:
                # Content-based works better with fewer interactions
                confidence *= max(0.5, 1 - interaction_count / 50)
        
        return confidence

class CascadeHybrid:
    """Cascade hybrid: apply primary method, then secondary for remaining items."""
    
    def __init__(self, config):
        self.config = config
        self.primary_method = config['primary_method']
        self.secondary_method = config['secondary_method']
        self.min_recommendations = config['min_recommendations']
    
    def recommend(self, collaborative_recs, content_recs, user_profile=None):
        """
        Apply cascade strategy.
        
        Args:
            collaborative_recs (list): Collaborative filtering recommendations
            content_recs (list): Content-based recommendations
            user_profile (dict): User profile
            
        Returns:
            list: Cascade recommendations
        """
        # Get primary recommendations
        if self.primary_method == 'collaborative':
            primary_recs = collaborative_recs
            secondary_recs = content_recs
        else:
            primary_recs = content_recs
            secondary_recs = collaborative_recs
        
        # If primary has enough recommendations, use them
        if len(primary_recs) >= self.min_recommendations:
            return primary_recs
        
        # Otherwise, combine with secondary
        combined_recs = primary_recs.copy()
        
        # Add secondary recommendations (excluding already recommended items)
        primary_items = set(item for item, _ in primary_recs)
        
        for item, score in secondary_recs:
            if item not in primary_items:
                combined_recs.append((item, score))
        
        # Sort by score
        combined_recs.sort(key=lambda x: x[1], reverse=True)
        
        return combined_recs

class FeatureCombinationHybrid:
    """Feature-level combination of collaborative and content-based features."""
    
    def __init__(self, config):
        self.config = config
        self.collaborative_features = config['collaborative_features']
        self.content_features = config['content_features']
        self.fusion_method = config['fusion_method']
    
    def recommend(self, collaborative_recs, content_recs, user_profile=None):
        """
        Combine at feature level.
        
        Args:
            collaborative_recs (list): Collaborative filtering recommendations
            content_recs (list): Content-based recommendations
            user_profile (dict): User profile
            
        Returns:
            list: Feature-combined recommendations
        """
        # Convert to feature vectors
        collab_features = self._recommendations_to_features(collaborative_recs)
        content_features = self._recommendations_to_features(content_recs)
        
        # Combine features
        if self.fusion_method == 'concatenate':
            combined_features = np.concatenate([
                collab_features * self.collaborative_features,
                content_features * self.content_features
            ], axis=1)
        elif self.fusion_method == 'average':
            combined_features = (collab_features * self.collaborative_features + 
                              content_features * self.content_features) / 2
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Convert back to recommendations
        return self._features_to_recommendations(combined_features)
    
    def _recommendations_to_features(self, recommendations):
        """Convert recommendations to feature vectors."""
        if not recommendations:
            return np.zeros((1, 10))  # Default feature vector
        
        # Extract scores as features
        scores = [score for _, score in recommendations]
        
        # Pad or truncate to fixed length
        feature_length = 10
        if len(scores) < feature_length:
            scores.extend([0] * (feature_length - len(scores)))
        else:
            scores = scores[:feature_length]
        
        return np.array(scores).reshape(1, -1)
    
    def _features_to_recommendations(self, features):
        """Convert feature vectors back to recommendations."""
        # This is a simplified conversion
        # In practice, you'd need a mapping back to items
        scores = features.flatten()
        return [(i, score) for i, score in enumerate(scores) if score > 0]

class HybridRecommender:
    """Main hybrid recommendation system."""
    
    def __init__(self, collaborative_model, content_model, hybrid_config):
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.hybrid_config = hybrid_config
        
        # Initialize hybrid strategies
        self.weighted_hybrid = WeightedHybrid(hybrid_config['weighted'])
        self.switching_hybrid = SwitchingHybrid(hybrid_config['switching'])
        self.cascade_hybrid = CascadeHybrid(hybrid_config['cascade'])
        self.feature_hybrid = FeatureCombinationHybrid(hybrid_config['feature_combination'])
        
        self.current_strategy = 'weighted'  # Default strategy
    
    def set_strategy(self, strategy):
        """Set the hybrid fusion strategy."""
        valid_strategies = ['weighted', 'switching', 'cascade', 'feature_combination']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from: {valid_strategies}")
        
        self.current_strategy = strategy
    
    def recommend(self, user_id, user_profile, n_recommendations=10, exclude_rated=True):
        """
        Generate hybrid recommendations.
        
        Args:
            user_id (int): User ID
            user_profile (dict): User profile
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Exclude items already rated by user
            
        Returns:
            list: Hybrid recommendations
        """
        # Get recommendations from both models
        try:
            collaborative_recs = self.collaborative_model.recommend(
                user_id, n_recommendations * 2, exclude_rated
            )
        except Exception as e:
            print(f"Collaborative filtering failed: {e}")
            collaborative_recs = []
        
        try:
            content_recs = self.content_model.recommend(
                user_profile, n_recommendations * 2, exclude_rated
            )
        except Exception as e:
            print(f"Content-based filtering failed: {e}")
            content_recs = []
        
        # Apply hybrid fusion strategy
        if self.current_strategy == 'weighted':
            combined_recs = self.weighted_hybrid.recommend(
                collaborative_recs, content_recs, user_profile
            )
        elif self.current_strategy == 'switching':
            combined_recs = self.switching_hybrid.recommend(
                collaborative_recs, content_recs, user_profile
            )
        elif self.current_strategy == 'cascade':
            combined_recs = self.cascade_hybrid.recommend(
                collaborative_recs, content_recs, user_profile
            )
        elif self.current_strategy == 'feature_combination':
            combined_recs = self.feature_hybrid.recommend(
                collaborative_recs, content_recs, user_profile
            )
        else:
            raise ValueError(f"Unknown strategy: {self.current_strategy}")
        
        # Return top recommendations
        return combined_recs[:n_recommendations]
    
    def get_recommendation_explanation(self, user_id, user_profile, recommendations):
        """
        Generate explanations for recommendations.
        
        Args:
            user_id (int): User ID
            user_profile (dict): User profile
            recommendations (list): Generated recommendations
            
        Returns:
            dict: Explanation for each recommendation
        """
        explanations = {}
        
        for item_id, score in recommendations:
            explanation = {
                'item_id': item_id,
                'score': score,
                'reasons': []
            }
            
            # Collaborative filtering explanation
            try:
                collab_recs = self.collaborative_model.recommend(user_id, 1, exclude_rated=False)
                if any(item == item_id for item, _ in collab_recs):
                    explanation['reasons'].append("Recommended by similar users")
            except:
                pass
            
            # Content-based explanation
            try:
                content_recs = self.content_model.recommend(user_profile, 1, exclude_rated=False)
                if any(item == item_id for item, _ in content_recs):
                    explanation['reasons'].append("Similar to items you liked")
            except:
                pass
            
            # Hybrid strategy explanation
            explanation['reasons'].append(f"Combined using {self.current_strategy} strategy")
            
            explanations[item_id] = explanation
        
        return explanations
    
    def evaluate_strategy_performance(self, test_users, test_matrix):
        """
        Evaluate performance of different hybrid strategies.
        
        Args:
            test_users (list): List of test user IDs
            test_matrix (pd.DataFrame): Test user-item matrix
            
        Returns:
            dict: Performance metrics for each strategy
        """
        strategies = ['weighted', 'switching', 'cascade', 'feature_combination']
        results = {}
        
        for strategy in strategies:
            self.set_strategy(strategy)
            
            total_precision = 0
            total_recall = 0
            count = 0
            
            for user_id in test_users[:10]:  # Test with first 10 users
                try:
                    # Get user profile
                    user_ratings = test_matrix.loc[user_id]
                    rated_items = user_ratings[user_ratings > 0].index.tolist()
                    
                    user_profile = {
                        'rated_items': rated_items,
                        'rating_count': len(rated_items)
                    }
                    
                    # Generate recommendations
                    recommendations = self.recommend(user_id, user_profile, 10)
                    recommended_items = [item for item, _ in recommendations]
                    
                    # Calculate metrics
                    relevant_items = set(rated_items)
                    recommended_set = set(recommended_items)
                    
                    if len(recommended_set) > 0:
                        precision = len(relevant_items & recommended_set) / len(recommended_set)
                        recall = len(relevant_items & recommended_set) / len(relevant_items) if len(relevant_items) > 0 else 0
                        
                        total_precision += precision
                        total_recall += recall
                        count += 1
                
                except Exception as e:
                    print(f"Error evaluating user {user_id}: {e}")
                    continue
            
            if count > 0:
                results[strategy] = {
                    'precision': total_precision / count,
                    'recall': total_recall / count
                }
        
        return results 