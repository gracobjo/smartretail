# Sistema de Análisis de Sentimiento en Twitter - SmartRetail

## Descripción del Proyecto

Este sistema implementa un pipeline completo de análisis de sentimiento en Twitter usando PySpark y técnicas modernas de NLP:

1. **PySpark Streaming**: Procesamiento en tiempo real de tweets
2. **NLP Moderno**: BERT/DistilBERT para embeddings y clasificación
3. **Análisis de Sentimiento**: Clasificación multiclase (positivo, negativo, neutro)
4. **Visualización**: Tendencias temporales y temas emergentes
5. **Streaming Analytics**: Análisis en tiempo real

## Características Principales

### Tecnologías Utilizadas

#### Procesamiento de Datos
- **PySpark**: Procesamiento distribuido y streaming
- **Spark Streaming**: Análisis en tiempo real
- **Spark SQL**: Consultas y agregaciones

#### NLP y Machine Learning
- **Transformers**: BERT/DistilBERT para embeddings
- **Hugging Face**: Modelos pre-entrenados
- **Scikit-learn**: Pipeline de ML
- **NLTK/Spacy**: Preprocesamiento de texto

#### Visualización y Analytics
- **Plotly**: Gráficos interactivos
- **WordCloud**: Nubes de palabras
- **Dash**: Dashboard en tiempo real
- **Matplotlib/Seaborn**: Visualizaciones estáticas

#### Streaming y Almacenamiento
- **Kafka**: Streaming de datos
- **MongoDB**: Almacenamiento de resultados
- **Redis**: Cache en tiempo real
- **Elasticsearch**: Búsqueda y análisis

### Funcionalidades del Sistema

#### 1. Captura de Datos
- **Twitter API**: Streaming de tweets en tiempo real
- **Kafka Integration**: Procesamiento de eventos
- **Rate Limiting**: Manejo de límites de API
- **Data Validation**: Validación de datos

#### 2. Preprocesamiento de Texto
- **Text Cleaning**: Limpieza de texto
- **Tokenization**: Tokenización avanzada
- **Stop Words**: Eliminación de palabras vacías
- **Lemmatization**: Lematización
- **Emoji Processing**: Procesamiento de emojis

#### 3. Análisis de Sentimiento
- **BERT Embeddings**: Embeddings contextuales
- **Sentiment Classification**: Clasificación multiclase
- **Confidence Scoring**: Puntuación de confianza
- **Entity Recognition**: Reconocimiento de entidades

#### 4. Análisis de Tendencias
- **Temporal Analysis**: Análisis temporal
- **Topic Modeling**: Modelado de temas
- **Trend Detection**: Detección de tendencias
- **Hashtag Analysis**: Análisis de hashtags

#### 5. Visualización
- **Real-time Dashboard**: Dashboard en tiempo real
- **Interactive Charts**: Gráficos interactivos
- **Word Clouds**: Nubes de palabras
- **Trend Graphs**: Gráficos de tendencias

## Estructura del Proyecto

```
twitter_sentiment_analysis/
├── src/
│   ├── data_ingestion/
│   │   ├── twitter_streamer.py      # Captura de tweets
│   │   ├── kafka_producer.py        # Productor Kafka
│   │   └── data_validator.py        # Validación de datos
│   ├── processing/
│   │   ├── text_preprocessor.py     # Preprocesamiento de texto
│   │   ├── nlp_pipeline.py          # Pipeline de NLP
│   │   └── sentiment_analyzer.py    # Análisis de sentimiento
│   ├── streaming/
│   │   ├── spark_streaming.py       # Spark Streaming
│   │   ├── real_time_processor.py   # Procesamiento en tiempo real
│   │   └── stream_aggregator.py     # Agregaciones de streaming
│   ├── analytics/
│   │   ├── trend_analyzer.py        # Análisis de tendencias
│   │   ├── topic_modeler.py         # Modelado de temas
│   │   └── hashtag_analyzer.py      # Análisis de hashtags
│   ├── visualization/
│   │   ├── dashboard.py             # Dashboard principal
│   │   ├── charts.py                # Gráficos y visualizaciones
│   │   └── wordcloud_generator.py   # Generador de nubes de palabras
│   └── utils/
│       ├── config.py                # Configuración
│       ├── database.py              # Conexiones a BD
│       └── helpers.py               # Utilidades
├── notebooks/
│   ├── sentiment_analysis_demo.ipynb
│   ├── trend_analysis.ipynb
│   └── visualization_examples.ipynb
├── data/
│   ├── raw/                         # Datos crudos
│   ├── processed/                   # Datos procesados
│   └── models/                      # Modelos entrenados
├── results/
│   ├── visualizations/              # Visualizaciones
│   ├── reports/                     # Reportes
│   └── analytics/                   # Resultados de análisis
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_processing.py
│   └── test_analytics.py
├── config/
│   ├── spark_config.yaml            # Configuración de Spark
│   ├── twitter_config.yaml          # Configuración de Twitter
│   └── model_config.yaml            # Configuración de modelos
├── main.py                          # Script principal
├── run_streaming.py                 # Ejecutar streaming
├── run_batch.py                     # Ejecutar procesamiento por lotes
├── requirements.txt                 # Dependencias
└── README.md                       # Documentación
```

## Instalación

### Prerrequisitos

- Python 3.8+
- Apache Spark 3.2+
- Apache Kafka 2.8+
- MongoDB 4.4+
- Redis 6.0+

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd twitter_sentiment_analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### Configuración de Spark

```bash
# Descargar Spark
wget https://downloads.apache.org/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar -xzf spark-3.2.0-bin-hadoop3.2.tgz
export SPARK_HOME=/path/to/spark-3.2.0-bin-hadoop3.2
export PATH=$PATH:$SPARK_HOME/bin
```

## Uso

### 1. Ejecutar Streaming en Tiempo Real

```bash
python run_streaming.py
```

### 2. Procesamiento por Lotes

```bash
python run_batch.py --input data/raw/tweets.json --output data/processed/
```

### 3. Dashboard en Tiempo Real

```bash
python src/visualization/dashboard.py
```

### 4. Análisis de Tendencias

```python
from src.analytics.trend_analyzer import TrendAnalyzer

analyzer = TrendAnalyzer()
trends = analyzer.analyze_trends(hours=24)
analyzer.plot_trends(trends)
```

## Configuración

### Twitter API

```yaml
# config/twitter_config.yaml
twitter:
  api_key: "your_api_key"
  api_secret: "your_api_secret"
  access_token: "your_access_token"
  access_token_secret: "your_access_token_secret"
  bearer_token: "your_bearer_token"
  
streaming:
  languages: ["es", "en"]
  keywords: ["python", "data science", "AI"]
  max_tweets: 10000
```

### Spark Configuration

```yaml
# config/spark_config.yaml
spark:
  app_name: "TwitterSentimentAnalysis"
  master: "local[*]"
  streaming:
    batch_duration: 10
    checkpoint_location: "checkpoints/"
  
  kafka:
    bootstrap_servers: "localhost:9092"
    topic: "twitter-stream"
```

## Características Avanzadas

### 1. Procesamiento en Tiempo Real

- **Spark Streaming**: Procesamiento de micro-lotes
- **Kafka Integration**: Streaming de eventos
- **Real-time Analytics**: Análisis en tiempo real
- **Dynamic Scaling**: Escalado automático

### 2. NLP Moderno

- **BERT Embeddings**: Embeddings contextuales
- **DistilBERT**: Modelo más rápido
- **Custom Training**: Entrenamiento personalizado
- **Multi-language**: Soporte multiidioma

### 3. Visualización Interactiva

- **Real-time Dashboard**: Dashboard en tiempo real
- **Interactive Charts**: Gráficos interactivos
- **Word Clouds**: Nubes de palabras dinámicas
- **Trend Analysis**: Análisis de tendencias

### 4. Análisis de Tendencias

- **Topic Detection**: Detección de temas
- **Trend Prediction**: Predicción de tendencias
- **Sentiment Evolution**: Evolución del sentimiento
- **Viral Content**: Contenido viral

## API Endpoints

### REST API

```python
# Obtener análisis de sentimiento
GET /api/sentiment/{tweet_id}

# Obtener tendencias
GET /api/trends?hours=24

# Obtener temas emergentes
GET /api/topics?limit=10

# Obtener estadísticas
GET /api/stats
```

### WebSocket API

```python
# Streaming de sentimientos en tiempo real
ws://localhost:8000/ws/sentiment

# Streaming de tendencias
ws://localhost:8000/ws/trends
```

## Monitoreo y Logging

### Métricas

- **Throughput**: Tweets procesados por segundo
- **Latency**: Latencia de procesamiento
- **Accuracy**: Precisión del modelo
- **Error Rate**: Tasa de errores

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/twitter_analysis.log'),
        logging.StreamHandler()
    ]
)
```

## Escalabilidad

### Horizontal Scaling

- **Spark Cluster**: Escalado horizontal
- **Kafka Partitions**: Particionamiento
- **Load Balancing**: Balanceo de carga
- **Auto-scaling**: Escalado automático

### Performance Optimization

- **Caching**: Cache en Redis
- **Partitioning**: Particionamiento de datos
- **Compression**: Compresión de datos
- **Indexing**: Indexación optimizada

## Seguridad

### Data Protection

- **Encryption**: Encriptación de datos
- **Access Control**: Control de acceso
- **Audit Logging**: Logs de auditoría
- **GDPR Compliance**: Cumplimiento GDPR

### API Security

- **Rate Limiting**: Límites de tasa
- **Authentication**: Autenticación
- **Authorization**: Autorización
- **Input Validation**: Validación de entrada

## Contribución

1. Fork el repositorio
2. Crear una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit los cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

- **Email**: contacto@smartretail.com
- **GitHub**: https://github.com/smartretail/twitter-sentiment-analysis
- **Documentación**: https://docs.smartretail.com/twitter-analysis 