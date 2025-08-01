# Dockerfile para SmartRetail
FROM python:3.9-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .
COPY fraud_detection/requirements.txt fraud_detection/
COPY recommendation_system/requirements.txt recommendation_system/
COPY sales_analytics_dashboard/requirements.txt sales_analytics_dashboard/
COPY twitter_sentiment_analysis/requirements.txt twitter_sentiment_analysis/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r fraud_detection/requirements.txt
RUN pip install --no-cache-dir -r recommendation_system/requirements.txt
RUN pip install --no-cache-dir -r sales_analytics_dashboard/requirements.txt
RUN pip install --no-cache-dir -r twitter_sentiment_analysis/requirements.txt

# Instalar dependencias adicionales
RUN pip install --no-cache-dir \
    xgboost \
    lightgbm \
    shap \
    optuna \
    streamlit \
    plotly \
    nltk \
    textblob \
    vaderSentiment \
    pyspark \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn

# Copiar el código del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p \
    fraud_detection/data \
    fraud_detection/results \
    fraud_detection/models \
    fraud_detection/logs \
    recommendation_system/data \
    recommendation_system/results \
    sales_analytics_dashboard/data \
    sales_analytics_dashboard/results \
    twitter_sentiment_analysis/data \
    twitter_sentiment_analysis/results

# Exponer puertos
EXPOSE 8501 8888 5000

# Comando por defecto
CMD ["bash"] 