"""
Sentiment analyzer using BERT/DistilBERT for Twitter data.
Implements modern NLP techniques for sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import re
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """BERT-based sentiment analyzer for Twitter data."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 max_length: int = 128, batch_size: int = 32, use_fallback: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Name of the BERT model to use
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            use_fallback: Whether to use fallback method if BERT fails
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fallback = use_fallback
        self.use_bert = True
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3  # positive, negative, neutral
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentiment labels
            self.sentiment_labels = ['negative', 'neutral', 'positive']
            
            print("BERT model loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            if self.use_fallback:
                print("Using fallback sentiment analysis method...")
                self.use_bert = False
                self._setup_fallback_analyzer()
            else:
                raise e
    
    def _setup_fallback_analyzer(self):
        """Setup fallback sentiment analysis using simple keyword-based approach."""
        # Positive and negative keywords
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'love', 'like', 'happy', 'joy', 'pleased', 'satisfied',
            'best', 'perfect', 'outstanding', 'brilliant', 'superb', 'terrific',
            'positive', 'optimistic', 'excited', 'thrilled', 'delighted'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
            'hate', 'dislike', 'sad', 'angry', 'frustrated', 'upset', 'annoyed',
            'negative', 'pessimistic', 'depressed', 'miserable', 'awful', 'dreadful',
            'terrible', 'horrible', 'atrocious', 'abysmal', 'lousy'
        }
        
        # Neutral indicators
        self.neutral_words = {
            'okay', 'fine', 'alright', 'normal', 'average', 'neutral',
            'indifferent', 'neither', 'nor', 'but', 'however'
        }
        
        print("Fallback sentiment analyzer initialized!")
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """
        Simple keyword-based sentiment analysis as fallback.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment prediction
        """
        if not text or text.strip() == '':
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        neutral_count = sum(1 for word in words if word in self.neutral_words)
        
        total_words = len(words)
        if total_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        # Calculate scores
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = neutral_count / total_words
        
        # Determine sentiment
        if positive_score > negative_score and positive_score > 0.1:
            sentiment = 'positive'
            confidence = min(positive_score * 2, 0.95)
        elif negative_score > positive_score and negative_score > 0.1:
            sentiment = 'negative'
            confidence = min(negative_score * 2, 0.95)
        else:
            sentiment = 'neutral'
            confidence = max(0.3, 1 - (positive_score + negative_score))
        
        return {
            'sentiment': sentiment,
            'confidence': confidence
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BERT model.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Basic cleaning
        text = str(text).strip()
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough estimate
            text = text[:self.max_length * 4]
        
        return text
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment prediction
        """
        if not text or text.strip() == '':
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
        
        # Use fallback method if BERT is not available
        if not self.use_bert:
            result = self._fallback_sentiment_analysis(text)
            scores = {
                'negative': 0.0,
                'neutral': 0.0,
                'positive': 0.0
            }
            scores[result['sentiment']] = result['confidence']
            # Distribute remaining probability
            remaining = 1.0 - result['confidence']
            other_labels = [l for l in scores.keys() if l != result['sentiment']]
            for other_label in other_labels:
                scores[other_label] = remaining / len(other_labels)
            
            return {
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'scores': scores
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        try:
            # Get prediction
            result = self.sentiment_pipeline(processed_text)[0]
            
            # Map labels
            label = result['label'].lower()
            confidence = result['score']
            
            # Create scores dictionary
            scores = {
                'negative': 0.0,
                'neutral': 0.0,
                'positive': 0.0
            }
            
            if label in scores:
                scores[label] = confidence
                # Distribute remaining probability
                remaining = 1.0 - confidence
                other_labels = [l for l in scores.keys() if l != label]
                for other_label in other_labels:
                    scores[other_label] = remaining / len(other_labels)
            
            return {
                'sentiment': label,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment predictions
        """
        print(f"Predicting sentiment for {len(texts)} texts...")
        
        results = []
        
        # Use fallback method if BERT is not available
        if not self.use_bert:
            for text in texts:
                result = self._fallback_sentiment_analysis(text)
                scores = {
                    'negative': 0.0,
                    'neutral': 0.0,
                    'positive': 0.0
                }
                scores[result['sentiment']] = result['confidence']
                # Distribute remaining probability
                remaining = 1.0 - result['confidence']
                other_labels = [l for l in scores.keys() if l != result['sentiment']]
                for other_label in other_labels:
                    scores[other_label] = remaining / len(other_labels)
                
                results.append({
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'scores': scores
                })
            return results
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Preprocess batch
            processed_texts = [self.preprocess_text(text) for text in batch_texts]
            
            try:
                # Get predictions for batch
                batch_results = self.sentiment_pipeline(processed_texts)
                
                for j, result in enumerate(batch_results):
                    label = result['label'].lower()
                    confidence = result['score']
                    
                    # Create scores dictionary
                    scores = {
                        'negative': 0.0,
                        'neutral': 0.0,
                        'positive': 0.0
                    }
                    
                    if label in scores:
                        scores[label] = confidence
                        # Distribute remaining probability
                        remaining = 1.0 - confidence
                        other_labels = [l for l in scores.keys() if l != label]
                        for other_label in other_labels:
                            scores[other_label] = remaining / len(other_labels)
                    
                    results.append({
                        'sentiment': label,
                        'confidence': confidence,
                        'scores': scores
                    })
                    
            except Exception as e:
                print(f"Error in batch prediction: {e}")
                # Add neutral predictions for failed batch
                for _ in batch_texts:
                    results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
                    })
        
        return results
    
    def analyze_tweets(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of tweets.
        
        Args:
            df: DataFrame with tweets
            text_column: Column containing text data
            
        Returns:
            DataFrame with sentiment analysis results
        """
        print(f"Analyzing sentiment for {len(df)} tweets...")
        
        # Get texts
        texts = df[text_column].tolist()
        
        # Predict sentiment
        predictions = self.predict_batch(texts)
        
        # Add results to DataFrame
        df['sentiment_prediction'] = [pred['sentiment'] for pred in predictions]
        df['sentiment_confidence'] = [pred['confidence'] for pred in predictions]
        df['sentiment_scores'] = [pred['scores'] for pred in predictions]
        
        # Extract individual scores
        df['negative_score'] = [pred['scores']['negative'] for pred in predictions]
        df['neutral_score'] = [pred['scores']['neutral'] for pred in predictions]
        df['positive_score'] = [pred['scores']['positive'] for pred in predictions]
        
        print("Sentiment analysis completed!")
        return df
    
    def evaluate_model(self, df: pd.DataFrame, true_column: str = 'sentiment', 
                      pred_column: str = 'sentiment_prediction') -> Dict:
        """
        Evaluate model performance.
        
        Args:
            df: DataFrame with true and predicted labels
            true_column: Column with true labels
            pred_column: Column with predicted labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Get true and predicted labels
        y_true = df[true_column].tolist()
        y_pred = df[pred_column].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.sentiment_labels)
        
        # Calculate per-class metrics
        class_metrics = {}
        for label in self.sentiment_labels:
            if label in report:
                class_metrics[label] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1_score': report[label]['f1-score'],
                    'support': report[label]['support']
                }
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'overall_metrics': {
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score']
            }
        }
        
        print(f"Model evaluation completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {results['overall_metrics']['macro_f1']:.4f}")
        
        return results
    
    def get_sentiment_distribution(self, df: pd.DataFrame, 
                                 pred_column: str = 'sentiment_prediction') -> pd.Series:
        """
        Get sentiment distribution.
        
        Args:
            df: DataFrame with predictions
            pred_column: Column with predicted sentiments
            
        Returns:
            Series with sentiment distribution
        """
        return df[pred_column].value_counts()
    
    def get_confidence_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get confidence statistics.
        
        Args:
            df: DataFrame with confidence scores
            
        Returns:
            Dictionary with confidence statistics
        """
        confidence_stats = {
            'mean_confidence': df['sentiment_confidence'].mean(),
            'median_confidence': df['sentiment_confidence'].median(),
            'std_confidence': df['sentiment_confidence'].std(),
            'min_confidence': df['sentiment_confidence'].min(),
            'max_confidence': df['sentiment_confidence'].max()
        }
        
        return confidence_stats
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        print(f"Saving model to {model_path}")
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print("Model saved successfully!")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Recreate pipeline
        self.sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("Model loaded successfully!")

class StreamingSentimentAnalyzer(SentimentAnalyzer):
    """Streaming version of sentiment analyzer for real-time processing."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 max_length: int = 128, batch_size: int = 32):
        """Initialize streaming sentiment analyzer."""
        super().__init__(model_name, max_length, batch_size)
        self.processed_count = 0
        self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def process_stream(self, text: str) -> Dict:
        """
        Process a single text in streaming mode.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment and streaming stats
        """
        # Predict sentiment
        prediction = self.predict_sentiment(text)
        
        # Update streaming stats
        self.processed_count += 1
        self.sentiment_counts[prediction['sentiment']] += 1
        
        # Calculate streaming metrics
        total = self.processed_count
        sentiment_percentages = {
            sentiment: (count / total) * 100 
            for sentiment, count in self.sentiment_counts.items()
        }
        
        # Add streaming stats to result
        result = prediction.copy()
        result.update({
            'processed_count': self.processed_count,
            'sentiment_counts': self.sentiment_counts.copy(),
            'sentiment_percentages': sentiment_percentages,
            'timestamp': pd.Timestamp.now()
        })
        
        return result
    
    def get_streaming_stats(self) -> Dict:
        """
        Get current streaming statistics.
        
        Returns:
            Dictionary with streaming stats
        """
        total = self.processed_count
        if total == 0:
            return {
                'processed_count': 0,
                'sentiment_counts': self.sentiment_counts,
                'sentiment_percentages': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        
        sentiment_percentages = {
            sentiment: (count / total) * 100 
            for sentiment, count in self.sentiment_counts.items()
        }
        
        return {
            'processed_count': total,
            'sentiment_counts': self.sentiment_counts.copy(),
            'sentiment_percentages': sentiment_percentages
        }
    
    def reset_streaming_stats(self):
        """Reset streaming statistics."""
        self.processed_count = 0
        self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

def main():
    """Test sentiment analyzer with sample data."""
    from src.data.data_processor import TwitterDataProcessor
    
    # Create sample data
    processor = TwitterDataProcessor()
    sample_df = processor.create_sample_data(n_tweets=100)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    results_df = analyzer.analyze_tweets(sample_df)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print(f"Total tweets: {len(results_df)}")
    print("\nSentiment distribution:")
    print(results_df['sentiment_prediction'].value_counts())
    
    print("\nConfidence statistics:")
    confidence_stats = analyzer.get_confidence_stats(results_df)
    for stat, value in confidence_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Evaluate if true labels are available
    if 'sentiment' in results_df.columns:
        print("\nModel Evaluation:")
        evaluation = analyzer.evaluate_model(results_df)
        print(f"Accuracy: {evaluation['accuracy']:.4f}")

if __name__ == "__main__":
    main() 