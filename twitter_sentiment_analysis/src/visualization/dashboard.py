"""
Twitter Sentiment Analysis Dashboard using Dash.
Interactive dashboard with real-time visualizations and streaming capabilities.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import local modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_processor import TwitterDataProcessor
from models.sentiment_analyzer import SentimentAnalyzer, StreamingSentimentAnalyzer

class TwitterSentimentDashboard:
    """Interactive dashboard for Twitter sentiment analysis."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ])
        
        self.app.title = "Twitter Sentiment Analysis Dashboard - SmartRetail"
        
        # Initialize components
        self.data_processor = TwitterDataProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.streaming_analyzer = StreamingSentimentAnalyzer()
        
        # Load or generate data
        self.load_data()
        
        # Setup layout
        self.setup_layout()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def load_data(self):
        """Load or generate sample data."""
        try:
            # Try to load existing data
            self.df = self.data_processor.load_processed_data('data/processed/sample_tweets.csv')
            
            # Analyze sentiment if not already done
            if 'sentiment_prediction' not in self.df.columns:
                self.df = self.sentiment_analyzer.analyze_tweets(self.df)
                self.data_processor.save_processed_data(self.df, 'data/processed/sample_tweets.csv')
            
            print("Loaded existing data")
            
        except FileNotFoundError:
            # Generate new data
            print("Generating sample data...")
            self.df = self.data_processor.create_sample_data(n_tweets=5000)
            self.df = self.sentiment_analyzer.analyze_tweets(self.df)
            self.data_processor.save_processed_data(self.df, 'data/processed/sample_tweets.csv')
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Twitter Sentiment Analysis Dashboard", className="text-center mb-4"),
                html.P("Real-time sentiment analysis with BERT and PySpark", 
                      className="text-center text-muted")
            ], className="container-fluid bg-primary text-white py-3"),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Date Range:", className="form-label"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=self.df['created_at'].min().date(),
                        end_date=self.df['created_at'].max().date(),
                        display_format='YYYY-MM-DD'
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Sentiment Filter:", className="form-label"),
                    dcc.Dropdown(
                        id='sentiment-filter',
                        options=[
                            {'label': 'All Sentiments', 'value': 'all'},
                            {'label': 'Positive', 'value': 'positive'},
                            {'label': 'Negative', 'value': 'negative'},
                            {'label': 'Neutral', 'value': 'neutral'}
                        ],
                        value='all',
                        placeholder="Select sentiment"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Confidence Threshold:", className="form-label"),
                    dcc.Slider(
                        id='confidence-threshold',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Text Length:", className="form-label"),
                    dcc.RangeSlider(
                        id='text-length-range',
                        min=self.df['text_length'].min(),
                        max=self.df['text_length'].max(),
                        step=10,
                        value=[self.df['text_length'].min(), self.df['text_length'].max()],
                        marks={
                            int(self.df['text_length'].min()): str(int(self.df['text_length'].min())),
                            int(self.df['text_length'].max()): str(int(self.df['text_length'].max()))
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="col-md-3")
            ], className="row mb-4"),
            
            # KPI Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Total Tweets", className="card-title"),
                        html.H2(id="total-tweets", className="text-primary"),
                        html.P("Analyzed tweets", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("Positive Sentiment", className="card-title"),
                        html.H2(id="positive-percentage", className="text-success"),
                        html.P("Percentage", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("Negative Sentiment", className="card-title"),
                        html.H2(id="negative-percentage", className="text-danger"),
                        html.P("Percentage", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("Avg Confidence", className="card-title"),
                        html.H2(id="avg-confidence", className="text-info"),
                        html.P("Model confidence", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3")
            ], className="row mb-4", id="kpi-cards"),
            
            # Charts Row 1
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Sentiment Distribution", className="card-title"),
                        dcc.Graph(id="sentiment-distribution")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Confidence Distribution", className="card-title"),
                        dcc.Graph(id="confidence-distribution")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Charts Row 2
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Sentiment Over Time", className="card-title"),
                        dcc.Graph(id="sentiment-timeline")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Text Length vs Sentiment", className="card-title"),
                        dcc.Graph(id="length-sentiment")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Charts Row 3
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Hashtag Analysis", className="card-title"),
                        dcc.Graph(id="hashtag-analysis")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Word Cloud Data", className="card-title"),
                        dcc.Graph(id="word-frequency")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Streaming Section
            html.Div([
                html.Div([
                    html.H4("Real-time Sentiment Analysis", className="card-title"),
                    html.P("Enter text to analyze sentiment in real-time:", className="text-muted"),
                    dcc.Textarea(
                        id='streaming-input',
                        placeholder='Enter text here...',
                        style={'width': '100%', 'height': 100}
                    ),
                    html.Button('Analyze Sentiment', id='analyze-button', 
                               className='btn btn-primary mt-2'),
                    html.Div(id='streaming-output', className='mt-3')
                ], className="card-body")
            ], className="col-md-12 mb-4"),
            
            # Sample Tweets Table
            html.Div([
                html.Div([
                    html.H5("Sample Tweets", className="card-title"),
                    html.Div(id="tweets-table")
                ], className="card-body")
            ], className="col-md-12 mb-4"),
            
            # Footer
            html.Div([
                html.P("SmartRetail Twitter Sentiment Analysis", 
                      className="text-center text-muted")
            ], className="container-fluid bg-light py-3")
            
        ], className="container-fluid")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("total-tweets", "children"),
             Output("positive-percentage", "children"),
             Output("negative-percentage", "children"),
             Output("avg-confidence", "children")],
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_kpi_cards(start_date, end_date, sentiment, confidence_threshold, length_range):
            """Update KPI cards based on filters."""
            
            # Apply filters
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            # Calculate KPIs
            total_tweets = len(filtered_df)
            
            if total_tweets > 0:
                positive_pct = (filtered_df['sentiment_prediction'] == 'positive').sum() / total_tweets * 100
                negative_pct = (filtered_df['sentiment_prediction'] == 'negative').sum() / total_tweets * 100
                avg_confidence = filtered_df['sentiment_confidence'].mean()
            else:
                positive_pct = 0
                negative_pct = 0
                avg_confidence = 0
            
            return [
                f"{total_tweets:,}",
                f"{positive_pct:.1f}%",
                f"{negative_pct:.1f}%",
                f"{avg_confidence:.3f}"
            ]
        
        @self.app.callback(
            Output("sentiment-distribution", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_sentiment_distribution(start_date, end_date, sentiment, 
                                       confidence_threshold, length_range):
            """Update sentiment distribution chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            sentiment_counts = filtered_df['sentiment_prediction'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=['#28a745', '#dc3545', '#6c757d']
            )])
            
            fig.update_layout(
                title="Sentiment Distribution",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("confidence-distribution", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_confidence_distribution(start_date, end_date, sentiment, 
                                        confidence_threshold, length_range):
            """Update confidence distribution chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_df['sentiment_confidence'],
                nbinsx=30,
                name='Confidence Distribution',
                marker_color='#007bff'
            ))
            
            fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("sentiment-timeline", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_sentiment_timeline(start_date, end_date, sentiment, 
                                    confidence_threshold, length_range):
            """Update sentiment timeline chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            # Group by hour and sentiment
            filtered_df['hour'] = filtered_df['created_at'].dt.hour
            timeline_data = filtered_df.groupby(['hour', 'sentiment_prediction']).size().reset_index(name='count')
            
            fig = go.Figure()
            
            for sentiment_type in ['positive', 'negative', 'neutral']:
                sentiment_data = timeline_data[timeline_data['sentiment_prediction'] == sentiment_type]
                if not sentiment_data.empty:
                    fig.add_trace(go.Scatter(
                        x=sentiment_data['hour'],
                        y=sentiment_data['count'],
                        mode='lines+markers',
                        name=sentiment_type.capitalize(),
                        line=dict(width=3)
                    ))
            
            fig.update_layout(
                title="Sentiment Over Time (Hourly)",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Tweets",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("length-sentiment", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_length_sentiment(start_date, end_date, sentiment, 
                                 confidence_threshold, length_range):
            """Update text length vs sentiment chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            fig = go.Figure()
            
            for sentiment_type in ['positive', 'negative', 'neutral']:
                sentiment_data = filtered_df[filtered_df['sentiment_prediction'] == sentiment_type]
                if not sentiment_data.empty:
                    fig.add_trace(go.Box(
                        y=sentiment_data['text_length'],
                        name=sentiment_type.capitalize(),
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                title="Text Length by Sentiment",
                yaxis_title="Text Length (characters)",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("hashtag-analysis", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_hashtag_analysis(start_date, end_date, sentiment, 
                                  confidence_threshold, length_range):
            """Update hashtag analysis chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            # Extract hashtags
            hashtags = []
            for text in filtered_df['text']:
                import re
                hashtags.extend(re.findall(r'#\w+', text))
            
            if hashtags:
                hashtag_counts = pd.Series(hashtags).value_counts().head(10)
                
                fig = go.Figure(data=[go.Bar(
                    x=hashtag_counts.values,
                    y=hashtag_counts.index,
                    orientation='h',
                    marker_color='#17a2b8'
                )])
                
                fig.update_layout(
                    title="Top Hashtags",
                    xaxis_title="Count",
                    yaxis_title="Hashtag",
                    template="plotly_white"
                )
            else:
                fig = go.Figure()
                fig.update_layout(
                    title="No Hashtags Found",
                    template="plotly_white"
                )
            
            return fig
        
        @self.app.callback(
            Output("word-frequency", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_word_frequency(start_date, end_date, sentiment, 
                               confidence_threshold, length_range):
            """Update word frequency chart."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            # Extract words from cleaned text
            words = []
            for text in filtered_df['cleaned_text']:
                if pd.notna(text):
                    words.extend(text.lower().split())
            
            if words:
                # Remove common words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                words = [word for word in words if word not in stop_words and len(word) > 2]
                
                word_counts = pd.Series(words).value_counts().head(15)
                
                fig = go.Figure(data=[go.Bar(
                    x=word_counts.values,
                    y=word_counts.index,
                    orientation='h',
                    marker_color='#28a745'
                )])
                
                fig.update_layout(
                    title="Most Common Words",
                    xaxis_title="Frequency",
                    yaxis_title="Word",
                    template="plotly_white"
                )
            else:
                fig = go.Figure()
                fig.update_layout(
                    title="No Words Found",
                    template="plotly_white"
                )
            
            return fig
        
        @self.app.callback(
            Output("streaming-output", "children"),
            [Input("analyze-button", "n_clicks")],
            [Input("streaming-input", "value")]
        )
        def analyze_streaming_text(n_clicks, text):
            """Analyze text in real-time."""
            if n_clicks and text:
                # Analyze sentiment
                result = self.streaming_analyzer.process_stream(text)
                
                # Create output
                sentiment_color = {
                    'positive': 'success',
                    'negative': 'danger',
                    'neutral': 'secondary'
                }
                
                return html.Div([
                    html.H5("Analysis Results:", className="mb-3"),
                    html.Div([
                        html.Span("Sentiment: ", className="fw-bold"),
                        html.Span(result['sentiment'].title(), 
                                className=f"badge bg-{sentiment_color[result['sentiment']]}")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Confidence: ", className="fw-bold"),
                        html.Span(f"{result['confidence']:.3f}")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Processed Tweets: ", className="fw-bold"),
                        html.Span(f"{result['processed_count']:,}")
                    ], className="mb-2"),
                    html.H6("Current Statistics:", className="mt-3"),
                    html.Div([
                        html.Span(f"{sentiment.title()}: {percentage:.1f}%", 
                                className=f"badge bg-{sentiment_color[sentiment]} me-2")
                        for sentiment, percentage in result['sentiment_percentages'].items()
                    ])
                ])
            
            return ""
        
        @self.app.callback(
            Output("tweets-table", "children"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("sentiment-filter", "value"),
             Input("confidence-threshold", "value"),
             Input("text-length-range", "value")]
        )
        def update_tweets_table(start_date, end_date, sentiment, 
                              confidence_threshold, length_range):
            """Update sample tweets table."""
            
            filtered_df = self.apply_filters(start_date, end_date, sentiment, 
                                           confidence_threshold, length_range)
            
            # Get sample tweets
            sample_tweets = filtered_df.head(10)[['text', 'sentiment_prediction', 
                                                'sentiment_confidence', 'created_at']]
            
            if sample_tweets.empty:
                return html.P("No tweets found with current filters.")
            
            # Create table
            table_rows = []
            for _, tweet in sample_tweets.iterrows():
                sentiment_color = {
                    'positive': 'success',
                    'negative': 'danger',
                    'neutral': 'secondary'
                }
                
                row = html.Tr([
                    html.Td(tweet['text'][:100] + "..." if len(tweet['text']) > 100 else tweet['text']),
                    html.Td(html.Span(tweet['sentiment_prediction'].title(), 
                                    className=f"badge bg-{sentiment_color[tweet['sentiment_prediction']]}")),
                    html.Td(f"{tweet['sentiment_confidence']:.3f}"),
                    html.Td(tweet['created_at'].strftime('%Y-%m-%d %H:%M'))
                ])
                table_rows.append(row)
            
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Text"),
                        html.Th("Sentiment"),
                        html.Th("Confidence"),
                        html.Th("Date")
                    ])
                ]),
                html.Tbody(table_rows)
            ], className="table table-striped")
    
    def apply_filters(self, start_date, end_date, sentiment, confidence_threshold, length_range):
        """Apply filters to the dataset."""
        filtered_df = self.df.copy()
        
        # Date filter
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[
                (filtered_df['created_at'] >= start_date) &
                (filtered_df['created_at'] <= end_date)
            ]
        
        # Sentiment filter
        if sentiment != 'all':
            filtered_df = filtered_df[filtered_df['sentiment_prediction'] == sentiment]
        
        # Confidence filter
        filtered_df = filtered_df[filtered_df['sentiment_confidence'] >= confidence_threshold]
        
        # Length filter
        if length_range:
            filtered_df = filtered_df[
                (filtered_df['text_length'] >= length_range[0]) &
                (filtered_df['text_length'] <= length_range[1])
            ]
        
        return filtered_df
    
    def run(self, debug=True, port=8051):
        """Run the dashboard."""
        self.app.run(debug=debug, port=port)

def main():
    """Run the Twitter sentiment analysis dashboard."""
    dashboard = TwitterSentimentDashboard()
    print("Starting Twitter Sentiment Analysis Dashboard...")
    print("Open http://localhost:8051 in your browser")
    dashboard.run()

if __name__ == "__main__":
    main() 