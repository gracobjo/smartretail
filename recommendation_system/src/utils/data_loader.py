"""
Data loader utilities for SmartRetail hybrid recommendation system.
Handles loading and preprocessing of user-item interactions and product features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RecommendationDataLoader:
    """Data loader for recommendation system datasets."""
    
    def __init__(self, config):
        self.config = config
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_interaction_data(self, file_path):
        """
        Load user-item interaction data.
        
        Args:
            file_path (str): Path to interaction data file
            
        Returns:
            tuple: (user_item_matrix, user_mapping, item_mapping)
        """
        print("Loading interaction data...")
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Ensure required columns exist
        required_columns = ['user_id', 'item_id', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter data based on configuration
        df = self._filter_interactions(df)
        
        # Encode users and items
        df['user_id_encoded'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_id_encoded'] = self.item_encoder.fit_transform(df['item_id'])
        
        # Create user-item matrix
        user_item_matrix = df.pivot_table(
            index='user_id_encoded',
            columns='item_id_encoded',
            values='rating',
            fill_value=0
        )
        
        print(f"Loaded {len(df)} interactions")
        print(f"Users: {user_item_matrix.shape[0]}")
        print(f"Items: {user_item_matrix.shape[1]}")
        print(f"Sparsity: {1 - (user_item_matrix != 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.3f}")
        
        return user_item_matrix, self.user_encoder, self.item_encoder
    
    def load_product_features(self, file_path):
        """
        Load product features data.
        
        Args:
            file_path (str): Path to product features file
            
        Returns:
            dict: Dictionary containing different types of features
        """
        print("Loading product features...")
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Ensure item_id column exists
        if 'item_id' not in df.columns:
            raise ValueError("Missing 'item_id' column in product features")
        
        # Align with item encoder
        if hasattr(self, 'item_encoder'):
            df = df[df['item_id'].isin(self.item_encoder.classes_)]
            df['item_id_encoded'] = self.item_encoder.transform(df['item_id'])
        
        features = {}
        
        # Text features
        text_columns = [col for col in self.config['text_columns'] if col in df.columns]
        if text_columns:
            features['text'] = self._process_text_features(df, text_columns)
        
        # Categorical features
        categorical_columns = [col for col in self.config['categorical_columns'] if col in df.columns]
        if categorical_columns:
            features['categorical'] = self._process_categorical_features(df, categorical_columns)
        
        # Numerical features
        numerical_columns = [col for col in self.config['numerical_columns'] if col in df.columns]
        if numerical_columns:
            features['numerical'] = self._process_numerical_features(df, numerical_columns)
        
        print(f"Processed features: {list(features.keys())}")
        
        return features
    
    def _filter_interactions(self, df):
        """Filter interactions based on configuration."""
        # Filter by minimum ratings per user
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.config['min_ratings_per_user']].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter by minimum ratings per item
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.config['min_ratings_per_item']].index
        df = df[df['item_id'].isin(valid_items)]
        
        return df
    
    def _process_text_features(self, df, text_columns):
        """Process text features for content-based filtering."""
        text_features = {}
        
        for column in text_columns:
            # Clean text
            df[f'{column}_clean'] = df[column].fillna('').astype(str).apply(self._clean_text)
            
            # Create text corpus
            text_corpus = df[f'{column}_clean'].tolist()
            text_features[column] = {
                'corpus': text_corpus,
                'raw_text': df[column].fillna('').astype(str).tolist()
            }
        
        return text_features
    
    def _process_categorical_features(self, df, categorical_columns):
        """Process categorical features."""
        categorical_features = {}
        
        for column in categorical_columns:
            # Encode categorical values
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(df[column].fillna('unknown'))
            
            categorical_features[column] = {
                'encoded_values': encoded_values,
                'encoder': encoder,
                'unique_values': encoder.classes_
            }
        
        return categorical_features
    
    def _process_numerical_features(self, df, numerical_columns):
        """Process numerical features."""
        numerical_features = {}
        
        for column in numerical_columns:
            # Fill missing values with median
            median_value = df[column].median()
            df[f'{column}_filled'] = df[column].fillna(median_value)
            
            # Scale numerical features
            scaled_values = self.scaler.fit_transform(df[f'{column}_filled'].values.reshape(-1, 1))
            
            numerical_features[column] = {
                'scaled_values': scaled_values.flatten(),
                'scaler': self.scaler,
                'median_value': median_value
            }
        
        return numerical_features
    
    def _clean_text(self, text):
        """Clean text for processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def split_data(self, user_item_matrix, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix
            test_size (float): Fraction of data for testing
            random_state (int): Random seed
            
        Returns:
            tuple: (train_matrix, test_matrix, train_interactions, test_interactions)
        """
        print("Splitting data into train and test sets...")
        
        # Convert matrix to interactions
        interactions = []
        for user_id in user_item_matrix.index:
            for item_id in user_item_matrix.columns:
                rating = user_item_matrix.loc[user_id, item_id]
                if rating > 0:  # Only positive interactions
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating
                    })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Split interactions
        train_interactions, test_interactions = train_test_split(
            interactions_df,
            test_size=test_size,
            random_state=random_state,
            stratify=interactions_df['user_id']
        )
        
        # Convert back to matrices
        train_matrix = self._interactions_to_matrix(train_interactions, user_item_matrix.shape)
        test_matrix = self._interactions_to_matrix(test_interactions, user_item_matrix.shape)
        
        print(f"Train interactions: {len(train_interactions)}")
        print(f"Test interactions: {len(test_interactions)}")
        
        return train_matrix, test_matrix, train_interactions, test_interactions
    
    def _interactions_to_matrix(self, interactions_df, original_shape):
        """Convert interactions DataFrame back to matrix."""
        matrix = pd.DataFrame(0, index=range(original_shape[0]), columns=range(original_shape[1]))
        
        for _, row in interactions_df.iterrows():
            matrix.loc[row['user_id'], row['item_id']] = row['rating']
        
        return matrix
    
    def create_user_profiles(self, user_item_matrix, product_features):
        """
        Create user profiles based on their interactions and product features.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item matrix
            product_features (dict): Product features
            
        Returns:
            dict: User profiles
        """
        print("Creating user profiles...")
        
        user_profiles = {}
        
        for user_id in user_item_matrix.index:
            # Get items rated by user
            user_items = user_item_matrix.loc[user_id]
            rated_items = user_items[user_items > 0].index.tolist()
            
            if not rated_items:
                continue
            
            # Calculate user preferences
            profile = {
                'rated_items': rated_items,
                'avg_rating': user_items[user_items > 0].mean(),
                'rating_count': len(rated_items),
                'preferences': {}
            }
            
            # Extract preferences from product features
            if 'categorical' in product_features:
                for feature_name, feature_data in product_features['categorical'].items():
                    item_values = []
                    for item_id in rated_items:
                        if item_id < len(feature_data['encoded_values']):
                            item_values.append(feature_data['encoded_values'][item_id])
                    
                    if item_values:
                        # Most common category for this user
                        profile['preferences'][feature_name] = max(set(item_values), key=item_values.count)
            
            if 'numerical' in product_features:
                for feature_name, feature_data in product_features['numerical'].items():
                    item_values = []
                    for item_id in rated_items:
                        if item_id < len(feature_data['scaled_values']):
                            item_values.append(feature_data['scaled_values'][item_id])
                    
                    if item_values:
                        # Average numerical preference
                        profile['preferences'][feature_name] = np.mean(item_values)
            
            user_profiles[user_id] = profile
        
        print(f"Created profiles for {len(user_profiles)} users")
        
        return user_profiles
    
    def classify_user_profile(self, user_profile, profile_config):
        """
        Classify user profile based on interaction patterns.
        
        Args:
            user_profile (dict): User profile
            profile_config (dict): Profile configuration
            
        Returns:
            str: User profile type
        """
        interaction_count = user_profile['rating_count']
        
        for profile_type, config in profile_config['profile_types'].items():
            if (config['min_interactions'] <= interaction_count <= 
                config['max_interactions']):
                return profile_type
        
        return 'active_user'  # Default fallback

class SyntheticDataGenerator:
    """Generate synthetic data for testing and demonstration."""
    
    def __init__(self, n_users=1000, n_items=500, n_interactions=10000):
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        
    def generate_interaction_data(self):
        """Generate synthetic user-item interactions."""
        print("Generating synthetic interaction data...")
        
        # Generate random interactions
        np.random.seed(42)
        
        # Create user-item pairs
        user_ids = np.random.randint(0, self.n_users, self.n_interactions)
        item_ids = np.random.randint(0, self.n_items, self.n_interactions)
        
        # Generate ratings (1-5 scale)
        ratings = np.random.choice([1, 2, 3, 4, 5], self.n_interactions, p=[0.1, 0.2, 0.3, 0.3, 0.1])
        
        # Create DataFrame
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings
        })
        
        # Remove duplicates
        interactions_df = interactions_df.drop_duplicates(['user_id', 'item_id'])
        
        print(f"Generated {len(interactions_df)} interactions")
        
        return interactions_df
    
    def generate_product_features(self):
        """Generate synthetic product features."""
        print("Generating synthetic product features...")
        
        # Categories and brands
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Food', 'Toys']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH']
        colors = ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Purple', 'Orange']
        sizes = ['S', 'M', 'L', 'XL', 'XXL', 'One Size']
        
        # Generate product data
        products = []
        for item_id in range(self.n_items):
            product = {
                'item_id': item_id,
                'title': f'Product {item_id}',
                'description': f'This is a description for product {item_id}. It is a great product with amazing features.',
                'category': np.random.choice(categories),
                'brand': np.random.choice(brands),
                'color': np.random.choice(colors),
                'size': np.random.choice(sizes),
                'price': np.random.uniform(10, 500),
                'rating': np.random.uniform(1, 5),
                'review_count': np.random.randint(0, 100)
            }
            products.append(product)
        
        products_df = pd.DataFrame(products)
        
        print(f"Generated features for {len(products_df)} products")
        
        return products_df 