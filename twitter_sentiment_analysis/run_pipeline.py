#!/usr/bin/env python3
"""
Script principal para ejecutar el Pipeline de Análisis de Sentimiento en Twitter.
"""

import os
import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_directories():
    """Crear directorios necesarios."""
    directories = [
        'data/raw',
        'data/processed',
        'data/sample',
        'models/saved',
        'results/visualizations',
        'results/reports',
        'results/exports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directorios creados correctamente!")

def generate_sample_data():
    """Generar datos de ejemplo si no existen."""
    try:
        from src.data.data_processor import TwitterDataProcessor
        
        # Verificar si ya existen datos
        if not os.path.exists('data/processed/sample_tweets.csv'):
            print("Generando datos de ejemplo...")
            processor = TwitterDataProcessor()
            sample_df = processor.create_sample_data(n_tweets=10000)
            processor.save_processed_data(sample_df, 'data/processed/sample_tweets.csv')
            
            print("Datos generados exitosamente!")
            print(f"- Tweets: {len(sample_df)}")
            print(f"- Sentiment distribution:")
            print(sample_df['sentiment'].value_counts())
        else:
            print("Datos de ejemplo ya existen.")
            
    except Exception as e:
        print(f"Error generando datos: {e}")
        return False
    
    return True

def run_sentiment_analysis():
    """Ejecutar análisis de sentimiento."""
    try:
        from src.data.data_processor import TwitterDataProcessor
        from src.models.sentiment_analyzer import SentimentAnalyzer
        
        print("Ejecutando análisis de sentimiento...")
        
        # Cargar datos
        processor = TwitterDataProcessor()
        df = processor.load_processed_data('data/processed/sample_tweets.csv')
        
        # Analizar sentimiento
        analyzer = SentimentAnalyzer()
        results_df = analyzer.analyze_tweets(df)
        
        # Guardar resultados
        processor.save_processed_data(results_df, 'data/processed/sentiment_results.csv')
        
        # Mostrar estadísticas
        print("\nResultados del análisis:")
        print(f"Total tweets: {len(results_df)}")
        print("\nDistribución de sentimientos:")
        print(results_df['sentiment_prediction'].value_counts())
        
        confidence_stats = analyzer.get_confidence_stats(results_df)
        print(f"\nEstadísticas de confianza:")
        for stat, value in confidence_stats.items():
            print(f"{stat}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return False

def run_spark_pipeline():
    """Ejecutar pipeline de PySpark."""
    try:
        from src.spark.spark_pipeline import SparkSentimentPipeline
        from src.models.sentiment_analyzer import SentimentAnalyzer
        
        print("Ejecutando pipeline de PySpark...")
        
        # Inicializar pipeline
        pipeline = SparkSentimentPipeline()
        
        try:
            # Crear datos de ejemplo
            pipeline.create_sample_data(n_tweets=5000)
            
            # Cargar datos
            df = pipeline.load_data('data/raw/sample_tweets.parquet')
            
            # Preprocesar datos
            processed_df = pipeline.preprocess_text(df)
            
            # Crear características de texto
            featured_df = pipeline.create_text_features(processed_df)
            
            # Inicializar analizador de sentimiento
            analyzer = SentimentAnalyzer()
            
            # Analizar sentimiento
            def analyze_sentiment(pandas_df):
                return analyzer.analyze_tweets(pandas_df)
            
            results_df = pipeline.analyze_sentiment_batch(featured_df, analyze_sentiment)
            
            # Agregar estadísticas
            stats = pipeline.aggregate_sentiment_stats(results_df)
            
            # Guardar resultados
            pipeline.save_results(results_df, 'data/processed/spark_sentiment_results.parquet')
            
            print("\nPipeline de PySpark completado exitosamente!")
            print(f"Estadísticas: {stats['sentiment_distribution']}")
            
            return True
            
        finally:
            pipeline.stop()
        
    except Exception as e:
        print(f"Error en pipeline de PySpark: {e}")
        return False

def run_dashboard():
    """Ejecutar dashboard interactivo."""
    try:
        from src.visualization.dashboard import TwitterSentimentDashboard
        
        print("Iniciando Twitter Sentiment Analysis Dashboard...")
        print("URL: http://localhost:8051")
        print("Presiona Ctrl+C para detener el servidor")
        
        dashboard = TwitterSentimentDashboard()
        dashboard.run(debug=True, port=8051)
        
        return True
        
    except Exception as e:
        print(f"Error ejecutando dashboard: {e}")
        return False

def run_streaming_analysis():
    """Ejecutar análisis de streaming."""
    try:
        from src.models.sentiment_analyzer import StreamingSentimentAnalyzer
        
        print("Iniciando análisis de streaming...")
        
        analyzer = StreamingSentimentAnalyzer()
        
        # Ejemplos de tweets para probar
        sample_tweets = [
            "I love this new product! It's amazing! 😍",
            "This is terrible, worst experience ever! 😠",
            "The weather is nice today, feeling good! ☀️",
            "I'm so frustrated with this service! 😤",
            "Great movie, highly recommend! 🎬"
        ]
        
        print("\nAnalizando tweets de ejemplo:")
        for i, tweet in enumerate(sample_tweets, 1):
            result = analyzer.process_stream(tweet)
            print(f"\nTweet {i}: {tweet}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processed count: {result['processed_count']}")
        
        # Mostrar estadísticas finales
        final_stats = analyzer.get_streaming_stats()
        print(f"\nEstadísticas finales:")
        print(f"Total procesados: {final_stats['processed_count']}")
        print(f"Distribución: {final_stats['sentiment_percentages']}")
        
        return True
        
    except Exception as e:
        print(f"Error en análisis de streaming: {e}")
        return False

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analysis Pipeline')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['data', 'sentiment', 'spark', 'dashboard', 'streaming', 'all'],
                       help='Modo de ejecución')
    parser.add_argument('--port', type=int, default=8051, help='Puerto del dashboard')
    parser.add_argument('--setup-only', action='store_true', help='Solo configurar directorios')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TWITTER SENTIMENT ANALYSIS PIPELINE - SmartRetail")
    print("=" * 60)
    
    # Configurar directorios
    setup_directories()
    
    if args.setup_only:
        print("Configuración completada.")
        return
    
    success = True
    
    if args.mode in ['data', 'all']:
        print("\n1. Generando datos de ejemplo...")
        success &= generate_sample_data()
    
    if args.mode in ['sentiment', 'all']:
        print("\n2. Ejecutando análisis de sentimiento...")
        success &= run_sentiment_analysis()
    
    if args.mode in ['spark', 'all']:
        print("\n3. Ejecutando pipeline de PySpark...")
        success &= run_spark_pipeline()
    
    if args.mode in ['streaming', 'all']:
        print("\n4. Ejecutando análisis de streaming...")
        success &= run_streaming_analysis()
    
    if args.mode in ['dashboard', 'all']:
        print("\n5. Iniciando dashboard...")
        success &= run_dashboard()
    
    if success:
        print("\n✅ Pipeline completado exitosamente!")
    else:
        print("\n❌ Pipeline completado con errores.")

if __name__ == "__main__":
    main() 