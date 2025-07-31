"""
PySpark pipeline for distributed Twitter sentiment analysis.
Handles large-scale data processing with streaming capabilities.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class SparkSentimentPipeline:
    """PySpark pipeline for distributed sentiment analysis."""
    
    def __init__(self, app_name: str = "TwitterSentimentAnalysis", 
                 master: str = "local[*]", config: Dict = None):
        """
        Initialize Spark pipeline.
        
        Args:
            app_name: Spark application name
            master: Spark master URL
            config: Configuration dictionary
        """
        self.app_name = app_name
        self.master = master
        self.config = config or {
            'spark': {
                'sql.adaptive.enabled': 'true',
                'sql.adaptive.coalescePartitions.enabled': 'true',
                'sql.adaptive.skewJoin.enabled': 'true',
                'sql.adaptive.localShuffleReader.enabled': 'true'
            },
            'processing': {
                'batch_size': 1000,
                'window_duration': '5 minutes',
                'slide_duration': '1 minute'
            }
        }
        
        # Initialize Spark session
        self.spark = self._create_spark_session()
        
        print(f"Spark session initialized: {self.spark.version}")
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        
        builder = SparkSession.builder \
            .appName(self.app_name) \
            .master(self.master)
        
        # Add Spark configurations
        for key, value in self.config['spark'].items():
            builder = builder.config(f"spark.{key}", value)
        
        # Add additional configurations for performance
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                       .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
                       .config("spark.sql.adaptive.enabled", "true") \
                       .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        return builder.getOrCreate()
    
    def create_sample_data(self, n_tweets: int = 10000) -> None:
        """
        Create sample Twitter data for testing.
        
        Args:
            n_tweets: Number of tweets to generate
        """
        print(f"Creating sample data with {n_tweets} tweets...")
        
        # Sample tweets
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
            tweet_idx = np.random.randint(0, len(sample_tweets))
            base_tweet = sample_tweets[tweet_idx]
            sentiment = sentiments[tweet_idx]
            
            # Add randomness
            random_words = ['really', 'very', 'so', 'quite', 'extremely']
            random_word = np.random.choice(random_words)
            
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
            
            data.append({
                'id': f'tweet_{i:06d}',
                'text': modified_tweet,
                'created_at': timestamp,
                'user_id': f'user_{np.random.randint(1000, 9999)}',
                'sentiment': sentiment
            })
        
        # Convert to Spark DataFrame
        df = self.spark.createDataFrame(data)
        
        # Save to parquet format
        df.write.mode('overwrite').parquet('data/raw/sample_tweets.parquet')
        
        print(f"Sample data created and saved with {len(data)} tweets")
    
    def load_data(self, file_path: str) -> 'DataFrame':
        """
        Load data into Spark DataFrame.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Spark DataFrame
        """
        print(f"Loading data from {file_path}")
        
        if file_path.endswith('.parquet'):
            df = self.spark.read.parquet(file_path)
        elif file_path.endswith('.csv'):
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)
        elif file_path.endswith('.json'):
            df = self.spark.read.json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Loaded {df.count()} records")
        return df
    
    def preprocess_text(self, df: 'DataFrame') -> 'DataFrame':
        """
        Preprocess text data using Spark SQL functions.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            Preprocessed Spark DataFrame
        """
        print("Preprocessing text data...")
        
        # Clean text using Spark SQL functions
        df_cleaned = df.withColumn(
            'cleaned_text',
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            lower(col('text')),
                            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ''
                        ),
                        r'@\w+', ''
                    ),
                    r'#(\w+)', '$1'  # Keep hashtag text, remove #
                ),
                r'\s+', ' '  # Remove extra whitespace
            )
        ).withColumn(
            'cleaned_text',
            trim(col('cleaned_text'))
        )
        
        # Add text features
        df_features = df_cleaned.withColumn(
            'text_length', length(col('cleaned_text'))
        ).withColumn(
            'word_count', size(split(col('cleaned_text'), ' '))
        ).withColumn(
            'hashtag_count', size(regexp_extract_all(col('text'), r'#\w+'))
        ).withColumn(
            'mention_count', size(regexp_extract_all(col('text'), r'@\w+'))
        ).withColumn(
            'url_count', size(regexp_extract_all(col('text'), r'http[s]?://'))
        ).withColumn(
            'exclamation_count', length(regexp_replace(col('text'), r'[^!]', ''))
        ).withColumn(
            'question_count', length(regexp_replace(col('text'), r'[^?]', ''))
        )
        
        # Filter by length
        df_filtered = df_features.filter(
            (col('text_length') >= 10) & 
            (col('text_length') <= 280) &
            (col('cleaned_text') != '')
        )
        
        print(f"Preprocessed {df_filtered.count()} tweets")
        return df_filtered
    
    def create_text_features(self, df: 'DataFrame') -> 'DataFrame':
        """
        Create text features using Spark ML.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            DataFrame with text features
        """
        print("Creating text features...")
        
        # Tokenize text
        tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokens")
        
        # Remove stop words
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        
        # Create TF-IDF features
        cv = CountVectorizer(inputCol="filtered_tokens", outputCol="tf", minDF=2)
        idf = IDF(inputCol="tf", outputCol="tfidf")
        
        # Create pipeline
        pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
        
        # Fit and transform
        model = pipeline.fit(df)
        df_features = model.transform(df)
        
        print("Text features created successfully")
        return df_features
    
    def analyze_sentiment_batch(self, df: 'DataFrame', 
                               analyzer_func) -> 'DataFrame':
        """
        Analyze sentiment for a batch of tweets.
        
        Args:
            df: Input Spark DataFrame
            analyzer_func: Function to analyze sentiment
            
        Returns:
            DataFrame with sentiment analysis results
        """
        print("Analyzing sentiment for batch...")
        
        # Convert to pandas for sentiment analysis
        pandas_df = df.toPandas()
        
        # Analyze sentiment
        results_df = analyzer_func(pandas_df)
        
        # Convert back to Spark DataFrame
        spark_df = self.spark.createDataFrame(results_df)
        
        print("Sentiment analysis completed")
        return spark_df
    
    def create_streaming_query(self, input_path: str, output_path: str,
                              analyzer_func) -> 'StreamingQuery':
        """
        Create a streaming query for real-time sentiment analysis.
        
        Args:
            input_path: Path to input data
            output_path: Path to output results
            analyzer_func: Function to analyze sentiment
            
        Returns:
            Streaming query
        """
        print("Creating streaming query...")
        
        # Read streaming data
        streaming_df = self.spark.readStream \
            .format("parquet") \
            .option("maxFilesPerTrigger", 10) \
            .load(input_path)
        
        # Preprocess data
        processed_df = self.preprocess_text(streaming_df)
        
        # Create text features
        featured_df = self.create_text_features(processed_df)
        
        # Analyze sentiment
        def analyze_batch(batch_df, batch_id):
            if batch_df.count() > 0:
                # Convert to pandas for sentiment analysis
                pandas_df = batch_df.toPandas()
                
                # Analyze sentiment
                results_df = analyzer_func(pandas_df)
                
                # Convert back to Spark DataFrame
                spark_df = self.spark.createDataFrame(results_df)
                
                # Write results
                spark_df.write.mode("append").parquet(output_path)
        
        # Create query
        query = featured_df.writeStream \
            .foreachBatch(analyze_batch) \
            .outputMode("append") \
            .trigger(processingTime="1 minute") \
            .start()
        
        print("Streaming query created successfully")
        return query
    
    def aggregate_sentiment_stats(self, df: 'DataFrame') -> Dict:
        """
        Aggregate sentiment statistics.
        
        Args:
            df: Spark DataFrame with sentiment results
            
        Returns:
            Dictionary with sentiment statistics
        """
        print("Aggregating sentiment statistics...")
        
        # Sentiment distribution
        sentiment_dist = df.groupBy('sentiment_prediction') \
            .count() \
            .toPandas()
        
        # Confidence statistics
        confidence_stats = df.select(
            mean('sentiment_confidence').alias('mean_confidence'),
            stddev('sentiment_confidence').alias('std_confidence'),
            min('sentiment_confidence').alias('min_confidence'),
            max('sentiment_confidence').alias('max_confidence')
        ).toPandas()
        
        # Time-based statistics
        time_stats = df.groupBy(
            window('created_at', '1 hour'),
            'sentiment_prediction'
        ).count().toPandas()
        
        # Hashtag analysis
        hashtag_stats = df.select(
            explode(regexp_extract_all(col('text'), r'#\w+')).alias('hashtag'),
            'sentiment_prediction'
        ).groupBy('hashtag', 'sentiment_prediction') \
         .count() \
         .orderBy('count', ascending=False) \
         .limit(20) \
         .toPandas()
        
        results = {
            'sentiment_distribution': sentiment_dist.to_dict('records'),
            'confidence_statistics': confidence_stats.to_dict('records')[0],
            'time_based_statistics': time_stats.to_dict('records'),
            'hashtag_statistics': hashtag_stats.to_dict('records')
        }
        
        print("Sentiment statistics aggregated successfully")
        return results
    
    def save_results(self, df: 'DataFrame', output_path: str, 
                    format: str = 'parquet'):
        """
        Save results to file.
        
        Args:
            df: Spark DataFrame to save
            output_path: Output file path
            format: Output format
        """
        print(f"Saving results to {output_path}")
        
        if format == 'parquet':
            df.write.mode('overwrite').parquet(output_path)
        elif format == 'csv':
            df.write.mode('overwrite').csv(output_path, header=True)
        elif format == 'json':
            df.write.mode('overwrite').json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print("Results saved successfully")
    
    def create_trend_analysis(self, df: 'DataFrame') -> 'DataFrame':
        """
        Create trend analysis for sentiment over time.
        
        Args:
            df: Spark DataFrame with sentiment results
            
        Returns:
            DataFrame with trend analysis
        """
        print("Creating trend analysis...")
        
        # Group by time windows and sentiment
        trend_df = df.groupBy(
            window('created_at', '1 hour'),
            'sentiment_prediction'
        ).agg(
            count('*').alias('count'),
            mean('sentiment_confidence').alias('avg_confidence')
        ).orderBy('window')
        
        # Calculate sentiment percentages
        window_spec = Window.partitionBy('window')
        
        trend_df = trend_df.withColumn(
            'total_count', sum('count').over(window_spec)
        ).withColumn(
            'percentage', (col('count') / col('total_count')) * 100
        )
        
        print("Trend analysis created successfully")
        return trend_df
    
    def create_word_cloud_data(self, df: 'DataFrame') -> 'DataFrame':
        """
        Create data for word cloud visualization.
        
        Args:
            df: Spark DataFrame with text data
            
        Returns:
            DataFrame with word frequencies
        """
        print("Creating word cloud data...")
        
        # Extract words from cleaned text
        word_df = df.select(
            explode(split(col('cleaned_text'), ' ')).alias('word'),
            'sentiment_prediction'
        ).filter(
            length(col('word')) > 2  # Filter short words
        ).groupBy('word', 'sentiment_prediction') \
         .count() \
         .orderBy('count', ascending=False)
        
        print("Word cloud data created successfully")
        return word_df
    
    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped")

def main():
    """Test Spark pipeline with sample data."""
    from src.models.sentiment_analyzer import SentimentAnalyzer
    
    # Initialize Spark pipeline
    pipeline = SparkSentimentPipeline()
    
    try:
        # Create sample data
        pipeline.create_sample_data(n_tweets=1000)
        
        # Load data
        df = pipeline.load_data('data/raw/sample_tweets.parquet')
        
        # Preprocess data
        processed_df = pipeline.preprocess_text(df)
        
        # Create text features
        featured_df = pipeline.create_text_features(processed_df)
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Analyze sentiment
        def analyze_sentiment(pandas_df):
            return analyzer.analyze_tweets(pandas_df)
        
        results_df = pipeline.analyze_sentiment_batch(featured_df, analyze_sentiment)
        
        # Aggregate statistics
        stats = pipeline.aggregate_sentiment_stats(results_df)
        
        # Save results
        pipeline.save_results(results_df, 'data/processed/sentiment_results.parquet')
        
        print("\nSpark pipeline completed successfully!")
        print(f"Sentiment distribution: {stats['sentiment_distribution']}")
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main() 