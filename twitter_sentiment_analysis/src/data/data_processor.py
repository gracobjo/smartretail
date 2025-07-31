"""
Data processor for Twitter sentiment analysis.
Handles data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TwitterDataProcessor:
    """Data processor for Twitter sentiment analysis."""
    
    def __init__(self, config=None):
        """Initialize data processor with configuration."""
        self.config = config or {
            'text_cleaning': {
                'remove_urls': True,
                'remove_mentions': True,
                'remove_hashtags': False,
                'remove_emojis': False,
                'lowercase': True,
                'remove_numbers': False
            },
            'preprocessing': {
                'min_length': 10,
                'max_length': 280,
                'remove_duplicates': True
            }
        }
    
    def load_twitter_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Twitter data from various formats.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with Twitter data
        """
        print(f"Loading Twitter data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        print(f"Loaded {len(df)} tweets")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for Twitter data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        # Common column mappings
        column_mappings = {
            'text': ['text', 'tweet_text', 'content', 'message'],
            'created_at': ['created_at', 'timestamp', 'date', 'created_date'],
            'user_id': ['user_id', 'author_id', 'user', 'author'],
            'sentiment': ['sentiment', 'label', 'target', 'class'],
            'id': ['id', 'tweet_id', 'tweet_id_str']
        }
        
        # Rename columns based on mappings
        for standard_name, possible_names in column_mappings.items():
            for col in possible_names:
                if col in df.columns:
                    df = df.rename(columns={col: standard_name})
                    break
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Remove URLs
        if self.config['text_cleaning']['remove_urls']:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions
        if self.config['text_cleaning']['remove_mentions']:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text)
        if self.config['text_cleaning']['remove_hashtags']:
            text = re.sub(r'#\w+', '', text)
        else:
            # Keep hashtag text but remove the #
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        if self.config['text_cleaning']['remove_emojis']:
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        
        # Convert to lowercase
        if self.config['text_cleaning']['lowercase']:
            text = text.lower()
        
        # Remove numbers
        if self.config['text_cleaning']['remove_numbers']:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Twitter data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("Preprocessing tweets...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Filter by length
        min_length = self.config['preprocessing']['min_length']
        max_length = self.config['preprocessing']['max_length']
        
        df = df[
            (df['cleaned_text'].str.len() >= min_length) &
            (df['cleaned_text'].str.len() <= max_length)
        ]
        
        # Remove duplicates
        if self.config['preprocessing']['remove_duplicates']:
            df = df.drop_duplicates(subset=['cleaned_text'])
        
        # Convert timestamp
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Add time-based features
        if 'created_at' in df.columns:
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['month'] = df['created_at'].dt.month
            df['year'] = df['created_at'].dt.year
        
        print(f"Preprocessed {len(df)} tweets")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from Twitter data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting features...")
        
        # Text length features
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Hashtag features
        df['hashtag_count'] = df['text'].str.count(r'#\w+')
        df['hashtags'] = df['text'].str.findall(r'#\w+')
        
        # Mention features
        df['mention_count'] = df['text'].str.count(r'@\w+')
        df['mentions'] = df['text'].str.findall(r'@\w+')
        
        # URL features
        df['url_count'] = df['text'].str.count(r'http[s]?://')
        
        # Exclamation and question marks
        df['exclamation_count'] = df['text'].str.count(r'!')
        df['question_count'] = df['text'].str.count(r'\?')
        
        # Emoji features
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        df['emoji_count'] = df['text'].apply(lambda x: len(emoji_pattern.findall(str(x))))
        
        # Sentiment indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated']
        
        df['positive_word_count'] = df['cleaned_text'].apply(
            lambda x: sum(1 for word in x.split() if word in positive_words)
        )
        df['negative_word_count'] = df['cleaned_text'].apply(
            lambda x: sum(1 for word in x.split() if word in negative_words)
        )
        
        # Sentiment score (simple)
        df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']
        
        print("Features extracted successfully")
        return df
    
    def create_sample_data(self, n_tweets: int = 10000) -> pd.DataFrame:
        """
        Create sample Twitter data for testing.
        
        Args:
            n_tweets: Number of tweets to generate
            
        Returns:
            DataFrame with sample Twitter data
        """
        print(f"Creating sample data with {n_tweets} tweets...")
        
        # Sample tweets with different sentiments
        sample_tweets = [
            "I love this new product! It's amazing! 😍 #awesome #loveit",
            "This is terrible, worst experience ever! 😠 #bad #hate",
            "The weather is nice today, feeling good! ☀️ #weather #happy",
            "I'm so frustrated with this service! 😤 #frustrated #bad",
            "Great movie, highly recommend! 🎬 #movie #recommend",
            "This food is delicious! 🍕 #food #yummy",
            "I'm disappointed with the quality 😞 #disappointed",
            "Amazing concert last night! 🎵 #concert #amazing",
            "This app is so buggy and slow! 😡 #buggy #slow",
            "Wonderful day at the beach! 🏖️ #beach #wonderful"
        ]
        
        sentiments = ['positive', 'negative', 'positive', 'negative', 'positive', 
                     'positive', 'negative', 'positive', 'negative', 'positive']
        
        # Generate sample data
        data = []
        for i in range(n_tweets):
            # Randomly select a tweet template
            tweet_idx = np.random.randint(0, len(sample_tweets))
            base_tweet = sample_tweets[tweet_idx]
            sentiment = sentiments[tweet_idx]
            
            # Add some randomness
            random_words = ['really', 'very', 'so', 'quite', 'extremely']
            random_word = np.random.choice(random_words)
            
            # Modify tweet
            if 'love' in base_tweet:
                modified_tweet = base_tweet.replace('love', f'{random_word} love')
            elif 'terrible' in base_tweet:
                modified_tweet = base_tweet.replace('terrible', f'{random_word} terrible')
            else:
                modified_tweet = base_tweet
            
            # Add random hashtags
            hashtags = ['#random', '#test', '#sample', '#data', '#twitter']
            random_hashtag = np.random.choice(hashtags)
            modified_tweet += f' {random_hashtag}'
            
            # Generate timestamp
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Generate user ID
            user_id = f'user_{np.random.randint(1000, 9999)}'
            
            data.append({
                'id': f'tweet_{i:06d}',
                'text': modified_tweet,
                'created_at': timestamp,
                'user_id': user_id,
                'sentiment': sentiment,
                'cleaned_text': self.clean_text(modified_tweet)
            })
        
        df = pd.DataFrame(data)
        
        # Extract features
        df = self.extract_features(df)
        
        print(f"Created sample data with {len(df)} tweets")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
        """
        print(f"Saving processed data to {output_path}")
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported output format: {output_path}")
        
        print("Data saved successfully")
    
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """
        Load processed data from file.
        
        Args:
            file_path: Path to processed data file
            
        Returns:
            Loaded DataFrame
        """
        print(f"Loading processed data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Convert timestamp column
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        print(f"Loaded {len(df)} processed tweets")
        return df

def main():
    """Test data processor with sample data."""
    processor = TwitterDataProcessor()
    
    # Create sample data
    sample_df = processor.create_sample_data(n_tweets=1000)
    
    # Save sample data
    processor.save_processed_data(sample_df, 'data/processed/sample_tweets.csv')
    
    print("\nSample data created and saved!")
    print(f"Total tweets: {len(sample_df)}")
    print(f"Sentiment distribution:")
    print(sample_df['sentiment'].value_counts())

if __name__ == "__main__":
    main() 