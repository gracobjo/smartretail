# Documentación Técnica - run_pipeline.py

## 📋 Descripción General

El archivo `run_pipeline.py` es el **orquestador principal** del sistema de análisis de sentimiento de Twitter. Actúa como punto de entrada único que coordina todos los componentes del sistema y permite ejecutar diferentes modos de operación.

## 🎯 Propósito

- **Orquestación**: Coordina todos los módulos del sistema
- **Configuración**: Gestiona la configuración inicial del entorno
- **Ejecución Modular**: Permite ejecutar componentes específicos
- **Monitoreo**: Proporciona feedback en tiempo real del progreso

## 🏗️ Arquitectura del Script

### Estructura Principal

```python
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
```

### Funciones Principales

#### 1. **setup_directories()**
```python
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
```

**Propósito**: Crea la estructura de directorios necesaria para el funcionamiento del sistema.

**Directorios Creados**:
- `data/raw`: Datos crudos de Twitter
- `data/processed`: Datos procesados y limpios
- `data/sample`: Datos de ejemplo para testing
- `models/saved`: Modelos entrenados
- `results/visualizations`: Visualizaciones generadas
- `results/reports`: Reportes de análisis
- `results/exports`: Exportaciones de datos
- `logs`: Archivos de log del sistema

#### 2. **generate_sample_data()**
```python
def generate_sample_data():
    """Generar datos de ejemplo si no existen."""
    try:
        from src.data.data_processor import TwitterDataProcessor
        
        if not os.path.exists('data/processed/sample_tweets.csv'):
            processor = TwitterDataProcessor()
            sample_df = processor.create_sample_data(n_tweets=10000)
            processor.save_processed_data(sample_df, 'data/processed/sample_tweets.csv')
```

**Propósito**: Genera datos de ejemplo para testing y demostración.

**Características**:
- Verifica si ya existen datos de ejemplo
- Genera 10,000 tweets de ejemplo
- Incluye distribución de sentimientos realista
- Guarda en formato CSV para fácil acceso

#### 3. **run_sentiment_analysis()**
```python
def run_sentiment_analysis():
    """Ejecutar análisis de sentimiento."""
    try:
        from src.data.data_processor import TwitterDataProcessor
        from src.models.sentiment_analyzer import SentimentAnalyzer
        
        # Cargar datos
        processor = TwitterDataProcessor()
        df = processor.load_processed_data('data/processed/sample_tweets.csv')
        
        # Analizar sentimiento
        analyzer = SentimentAnalyzer()
        results_df = analyzer.analyze_tweets(df)
```

**Propósito**: Ejecuta el análisis de sentimiento completo.

**Flujo de Procesamiento**:
1. **Carga de Datos**: Carga tweets procesados
2. **Análisis BERT**: Aplica modelo de sentimiento
3. **Cálculo de Confianza**: Calcula puntuación de confianza
4. **Estadísticas**: Genera métricas de rendimiento
5. **Almacenamiento**: Guarda resultados procesados

#### 4. **run_spark_pipeline()**
```python
def run_spark_pipeline():
    """Ejecutar pipeline de PySpark."""
    try:
        from src.spark.spark_pipeline import SparkSentimentPipeline
        from src.models.sentiment_analyzer import SentimentAnalyzer
        
        # Inicializar pipeline
        pipeline = SparkSentimentPipeline()
        
        # Crear datos de ejemplo
        pipeline.create_sample_data(n_tweets=5000)
        
        # Cargar y procesar datos
        df = pipeline.load_data('data/raw/sample_tweets.parquet')
        processed_df = pipeline.preprocess_text(df)
        featured_df = pipeline.create_text_features(processed_df)
```

**Propósito**: Ejecuta el pipeline distribuido con PySpark.

**Características**:
- **Procesamiento Distribuido**: Utiliza Spark para escalabilidad
- **Optimizaciones**: Particionamiento y caching automático
- **Streaming**: Soporte para procesamiento en tiempo real
- **Fault Tolerance**: Tolerancia a fallos distribuidos

#### 5. **run_dashboard()**
```python
def run_dashboard():
    """Ejecutar dashboard interactivo."""
    try:
        from src.visualization.dashboard import TwitterSentimentDashboard
        
        print("Iniciando Twitter Sentiment Analysis Dashboard...")
        print("URL: http://localhost:8051")
        
        dashboard = TwitterSentimentDashboard()
        dashboard.run(debug=True, port=8051)
```

**Propósito**: Inicia el dashboard interactivo web.

**Características**:
- **Interfaz Web**: Dashboard accesible vía navegador
- **Visualizaciones Interactivas**: Gráficos dinámicos
- **Filtros en Tiempo Real**: Filtrado dinámico de datos
- **Streaming de Datos**: Actualizaciones automáticas

#### 6. **run_streaming_analysis()**
```python
def run_streaming_analysis():
    """Ejecutar análisis de streaming."""
    try:
        from src.models.sentiment_analyzer import StreamingSentimentAnalyzer
        
        analyzer = StreamingSentimentAnalyzer()
        
        # Ejemplos de tweets para probar
        sample_tweets = [
            "I love this new product! It's amazing! 😍",
            "This is terrible, worst experience ever! 😠",
            "The weather is nice today, feeling good! ☀️"
        ]
```

**Propósito**: Ejecuta análisis de sentimiento en tiempo real.

**Características**:
- **Procesamiento Inmediato**: Análisis instantáneo
- **Métricas de Rendimiento**: Tiempo de procesamiento
- **Estadísticas Acumulativas**: Métricas del stream
- **Ejemplos Interactivos**: Demostración con tweets de ejemplo

## ⚙️ Configuración de Argumentos

### Parser de Argumentos
```python
def main():
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analysis Pipeline')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['data', 'sentiment', 'spark', 'dashboard', 'streaming', 'all'],
                       help='Modo de ejecución')
    parser.add_argument('--port', type=int, default=8051, help='Puerto del dashboard')
    parser.add_argument('--setup-only', action='store_true', help='Solo configurar directorios')
```

### Modos de Ejecución

| Modo | Descripción | Comando |
|------|-------------|---------|
| `data` | Solo generar datos de ejemplo | `--mode data` |
| `sentiment` | Solo análisis de sentimiento | `--mode sentiment` |
| `spark` | Solo pipeline PySpark | `--mode spark` |
| `dashboard` | Solo dashboard interactivo | `--mode dashboard` |
| `streaming` | Solo análisis de streaming | `--mode streaming` |
| `all` | Ejecutar todo el pipeline | `--mode all` |

### Opciones Adicionales

| Opción | Descripción | Uso |
|--------|-------------|-----|
| `--port` | Puerto del dashboard | `--port 8051` |
| `--setup-only` | Solo configuración | `--setup-only` |

## 🔄 Flujo de Ejecución

### Diagrama de Flujo

```
┌─────────────────┐
│   Inicio        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Setup Directories│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Parse Arguments │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Mode      │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Data    │ │ Sentiment│
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Spark   │ │ Dashboard│
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│Streaming│ │  End    │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │   Success   │
    └─────────────┘
```

### Lógica de Control

```python
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
```

## 📊 Manejo de Errores

### Estructura de Try-Catch
```python
def run_sentiment_analysis():
    """Ejecutar análisis de sentimiento."""
    try:
        # Código de análisis
        return True
    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return False
```

### Tipos de Errores Manejados

1. **Import Errors**: Errores de importación de módulos
2. **File Not Found**: Archivos de datos no encontrados
3. **Model Loading Errors**: Errores al cargar modelos BERT
4. **Spark Configuration Errors**: Errores de configuración Spark
5. **Dashboard Errors**: Errores del servidor web

### Logging y Feedback
```python
print("=" * 60)
print("TWITTER SENTIMENT ANALYSIS PIPELINE - SmartRetail")
print("=" * 60)

if success:
    print("\n✅ Pipeline completado exitosamente!")
else:
    print("\n❌ Pipeline completado con errores.")
```

## 🚀 Ejemplos de Uso

### 1. **Ejecución Completa**
```bash
# Ejecutar todo el pipeline
python run_pipeline.py --mode all
```

**Salida Esperada**:
```
============================================================
TWITTER SENTIMENT ANALYSIS PIPELINE - SmartRetail
============================================================
Directorios creados correctamente!

1. Generando datos de ejemplo...
Generando datos de ejemplo...
Datos generados exitosamente!
- Tweets: 10000
- Sentiment distribution:
positive    4500
negative    2500
neutral     3000

2. Ejecutando análisis de sentimiento...
Ejecutando análisis de sentimiento...
Resultados del análisis:
Total tweets: 10000
Distribución de sentimientos:
positive    4500
negative    2500
neutral     3000

3. Ejecutando pipeline de PySpark...
Ejecutando pipeline de PySpark...
Pipeline de PySpark completado exitosamente!

4. Ejecutando análisis de streaming...
Iniciando análisis de streaming...
Analizando tweets de ejemplo:
Tweet 1: I love this new product! It's amazing! 😍
Sentiment: positive
Confidence: 0.892

5. Iniciando dashboard...
Iniciando Twitter Sentiment Analysis Dashboard...
URL: http://localhost:8051

✅ Pipeline completado exitosamente!
```

### 2. **Solo Dashboard**
```bash
# Ejecutar solo el dashboard
python run_pipeline.py --mode dashboard --port 8051
```

### 3. **Solo Análisis de Sentimiento**
```bash
# Ejecutar solo análisis de sentimiento
python run_pipeline.py --mode sentiment
```

### 4. **Solo Configuración**
```bash
# Solo crear directorios
python run_pipeline.py --setup-only
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Configurar variables de entorno
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
export SPARK_HOME="/path/to/spark"
export TWITTER_API_KEY="your_api_key"
```

### Configuración de Logging
```python
import logging

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### Configuración de Rendimiento
```python
# Configuraciones de rendimiento
performance_config = {
    'batch_size': 32,
    'max_workers': 4,
    'memory_limit': '4GB',
    'timeout': 300
}
```

## 📈 Métricas y Monitoreo

### Métricas de Rendimiento
```python
# Métricas típicas del pipeline
pipeline_metrics = {
    'total_execution_time': 45.2,  # segundos
    'data_generation_time': 5.1,
    'sentiment_analysis_time': 12.3,
    'spark_pipeline_time': 18.7,
    'dashboard_startup_time': 3.2,
    'memory_usage_mb': 2048,
    'cpu_usage_percent': 65
}
```

### Logs de Ejecución
```python
# Estructura de logs
log_structure = {
    'timestamp': '2024-01-15T10:30:00Z',
    'mode': 'all',
    'status': 'success',
    'execution_time': 45.2,
    'errors': [],
    'warnings': []
}
```

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. **Error de Importación**
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solución: Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
```

#### 2. **Error de Spark**
```bash
# Error: SparkSession not initialized
# Solución: Verificar SPARK_HOME
export SPARK_HOME="/path/to/spark"
```

#### 3. **Error de Puerto**
```bash
# Error: Port already in use
# Solución: Cambiar puerto
python run_pipeline.py --mode dashboard --port 8052
```

### Debugging
```bash
# Ejecutar con logging detallado
python run_pipeline.py --mode all --log-level DEBUG

# Ver logs en tiempo real
tail -f logs/pipeline.log
```

## 📚 Referencias

### Dependencias Principales
- **argparse**: Parsing de argumentos de línea de comandos
- **pathlib**: Manipulación de rutas de archivos
- **sys**: Funcionalidades del sistema
- **os**: Interfaz con el sistema operativo

### Módulos del Sistema
- **src.data.data_processor**: Procesamiento de datos
- **src.models.sentiment_analyzer**: Análisis de sentimiento
- **src.spark.spark_pipeline**: Pipeline de Spark
- **src.visualization.dashboard**: Dashboard interactivo

---

*Documentación técnica del orquestador principal - SmartRetail Twitter Sentiment Analysis System* 