# SmartRetail - Análisis Avanzado de Datos y Machine Learning

## 🎯 Descripción del Proyecto

SmartRetail es una plataforma completa de análisis de datos y machine learning que incluye:

1. **Dashboard de Análisis de Ventas** - KPIs avanzados con CLV, Churn Rate y Cohort Analysis
2. **Pipeline de Análisis de Sentimiento en Twitter** - Procesamiento distribuido con PySpark y BERT

## 🚀 Características Principales

### 📊 Dashboard de Análisis de Ventas
- **Customer Lifetime Value (CLV)**: Análisis del valor del cliente
- **Churn Rate**: Tasa de abandono y predicción
- **Cohort Analysis**: Análisis de retención por cohortes
- **RFM Segmentation**: Segmentación de clientes
- **Visualizaciones Interactivas**: Dashboards dinámicos con Dash

### 🐦 Pipeline de Análisis de Sentimiento en Twitter
- **Procesamiento Distribuido**: PySpark para grandes volúmenes
- **NLP Moderno**: BERT/DistilBERT para análisis de sentimiento
- **Streaming en Tiempo Real**: Procesamiento continuo
- **Visualizaciones Dinámicas**: Word clouds y gráficos interactivos

## 📁 Estructura del Proyecto

```
SmartRetail/
├── sales_analytics_dashboard/          # Dashboard de análisis de ventas
│   ├── src/
│   │   ├── data/                      # Procesamiento de datos
│   │   ├── analytics/                 # Cálculo de KPIs
│   │   └── visualization/             # Dashboard interactivo
│   ├── notebooks/                     # Jupyter notebooks
│   ├── data/                         # Datos de ejemplo
│   └── run_dashboard.py              # Script principal
├── twitter_sentiment_analysis/        # Pipeline de análisis de sentimiento
│   ├── src/
│   │   ├── data/                     # Procesamiento de datos
│   │   ├── models/                   # Modelos BERT
│   │   ├── spark/                    # Pipeline de PySpark
│   │   └── visualization/            # Dashboard interactivo
│   ├── data/                         # Datos de ejemplo
│   └── run_pipeline.py               # Script principal
├── requirements.txt                   # Dependencias principales
└── README.md                         # Documentación
```

## 🛠️ Tecnologías Utilizadas

### Backend y Procesamiento
- **Python 3.8+**: Lenguaje principal
- **PySpark**: Procesamiento distribuido
- **BERT/DistilBERT**: Análisis de sentimiento
- **Pandas/NumPy**: Manipulación de datos
- **Scikit-learn**: Machine Learning

### Frontend y Visualización
- **Dash**: Framework web para dashboards
- **Plotly**: Gráficos interactivos
- **Bootstrap**: Diseño responsivo

### Deep Learning y NLP
- **Transformers**: Modelos de lenguaje
- **Torch**: Framework de deep learning
- **NLTK/Spacy**: Procesamiento de texto

## 📦 Instalación

### Prerrequisitos
- Python 3.8+
- Java 8+ (para PySpark)
- Git

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

### 1. Dashboard de Análisis de Ventas

```bash
cd sales_analytics_dashboard

# Generar datos de ejemplo
python run_dashboard.py --generate-data

# Ejecutar dashboard
python run_dashboard.py
```

**Acceder a**: http://localhost:8050

### 2. Pipeline de Análisis de Sentimiento en Twitter

```bash
cd twitter_sentiment_analysis

# Ejecutar pipeline completo
python run_pipeline.py --mode all

# Solo dashboard
python run_pipeline.py --mode dashboard
```

**Acceder a**: http://localhost:8051

## 📊 KPIs y Métricas

### Dashboard de Ventas
- **CLV (Customer Lifetime Value)**: Valor total del cliente
- **Churn Rate**: Tasa de abandono mensual/trimestral
- **Cohort Retention**: Retención por cohortes temporales
- **RFM Segmentation**: Recency, Frequency, Monetary
- **Revenue Metrics**: MRR, ARR, Growth Rate

### Análisis de Sentimiento
- **Sentiment Distribution**: Positive, Negative, Neutral
- **Confidence Scores**: Precisión del modelo
- **Trend Analysis**: Evolución temporal
- **Hashtag Analysis**: Tendencias emergentes
- **Word Clouds**: Visualización de palabras

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Para PySpark
export SPARK_HOME=/path/to/spark
export PYSPARK_PYTHON=python3

# Para modelos BERT
export TRANSFORMERS_CACHE=/path/to/cache
```

### Configuración de Spark
```python
# En spark_pipeline.py
config = {
    'spark.sql.adaptive.enabled': 'true',
    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
    'spark.sql.adaptive.skewJoin.enabled': 'true'
}
```

## 📈 Características Avanzadas

### Escalabilidad
- **Procesamiento Distribuido**: PySpark para grandes volúmenes
- **Streaming en Tiempo Real**: Análisis continuo
- **Caching Inteligente**: Optimización de rendimiento

### Machine Learning
- **Modelos Pre-entrenados**: BERT para análisis de sentimiento
- **Fine-tuning**: Adaptación a dominios específicos
- **Evaluación Automática**: Métricas de rendimiento

### Visualización
- **Dashboards Interactivos**: Filtros dinámicos
- **Gráficos en Tiempo Real**: Actualizaciones automáticas
- **Exportación de Datos**: PDF, Excel, CSV

## 🤝 Contribución

1. Fork el repositorio
2. Crear una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit los cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

- **GitHub**: [@gracobjo](https://github.com/gracobjo)
- **Repositorio**: [SmartRetail](https://github.com/gracobjo/smartretail.git)

## 🙏 Agradecimientos

- **Hugging Face**: Por los modelos BERT
- **Apache Spark**: Por el procesamiento distribuido
- **Plotly**: Por las visualizaciones interactivas
- **Dash**: Por el framework de dashboards

---

**SmartRetail** - Transformando datos en insights inteligentes 🚀 