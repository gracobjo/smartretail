# 📚 Documentación - Twitter Sentiment Analysis System

## 🎯 Descripción General

Este directorio contiene toda la documentación del **Sistema de Análisis de Sentimiento en Twitter**, una plataforma completa que combina tecnologías modernas de NLP, procesamiento distribuido y visualización interactiva.

## 📋 Índice de Documentación

### 📖 **Documentación Funcional**
- **Archivo**: [`functional_documentation.md`](functional_documentation.md)
- **Descripción**: Documentación completa de todas las funcionalidades del sistema
- **Audiencia**: Desarrolladores, arquitectos, usuarios técnicos
- **Contenido**:
  - Arquitectura del sistema
  - Funcionalidades principales
  - Pipeline de procesamiento
  - Módulos y componentes
  - API y endpoints
  - Casos de uso
  - Monitoreo y métricas

### 🔧 **Documentación Técnica - run_pipeline.py**
- **Archivo**: [`run_pipeline_documentation.md`](run_pipeline_documentation.md)
- **Descripción**: Documentación técnica específica del orquestador principal
- **Audiencia**: Desarrolladores, DevOps
- **Contenido**:
  - Arquitectura del script
  - Funciones principales
  - Configuración de argumentos
  - Flujo de ejecución
  - Manejo de errores
  - Ejemplos de uso
  - Troubleshooting

### 🚀 **Guía de Inicio Rápido**
- **Archivo**: [`quick_start_guide.md`](quick_start_guide.md)
- **Descripción**: Guía práctica para comenzar rápidamente
- **Audiencia**: Usuarios finales, desarrolladores nuevos
- **Contenido**:
  - Instalación rápida
  - Modos de uso
  - Dashboard features
  - Casos de uso típicos
  - Solución de problemas
  - Mejores prácticas

## 🎯 Audiencias Objetivo

### 👥 **Desarrolladores**
- **Documentación Funcional**: Entender la arquitectura completa
- **Documentación Técnica**: Implementar y mantener el código
- **Guía de Inicio Rápido**: Configurar el entorno rápidamente

### 👥 **Arquitectos de Software**
- **Documentación Funcional**: Diseñar integraciones
- **Documentación Técnica**: Evaluar escalabilidad
- **Guía de Inicio Rápido**: Prototipar rápidamente

### 👥 **Usuarios Finales**
- **Guía de Inicio Rápido**: Usar el sistema efectivamente
- **Documentación Funcional**: Entender capacidades
- **Documentación Técnica**: Configuración avanzada

### 👥 **DevOps/Infraestructura**
- **Documentación Técnica**: Despliegue y monitoreo
- **Documentación Funcional**: Requisitos de sistema
- **Guía de Inicio Rápido**: Configuración inicial

## 📊 Estructura del Sistema

### 🏗️ **Arquitectura General**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Twitter API   │───▶│  Data Ingestion │───▶│  Preprocessing  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│  Visualization  │◀───│ Sentiment Model │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Analytics     │◀───│  Spark Pipeline │◀───│  Data Storage   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 📦 **Componentes Principales**

#### 1. **Data Processing**
- **Módulo**: `src/data/data_processor.py`
- **Función**: Procesamiento y limpieza de datos
- **Tecnologías**: Pandas, NumPy, Regex

#### 2. **Sentiment Analysis**
- **Módulo**: `src/models/sentiment_analyzer.py`
- **Función**: Análisis de sentimiento con BERT
- **Tecnologías**: Transformers, Hugging Face, PyTorch

#### 3. **Spark Pipeline**
- **Módulo**: `src/spark/spark_pipeline.py`
- **Función**: Procesamiento distribuido
- **Tecnologías**: PySpark, Spark Streaming

#### 4. **Dashboard**
- **Módulo**: `src/visualization/dashboard.py`
- **Función**: Visualización interactiva
- **Tecnologías**: Dash, Plotly, Bootstrap

## 🚀 Inicio Rápido

### **1. Instalación**
```bash
# Clonar repositorio
git clone <repository-url>
cd twitter_sentiment_analysis

# Instalar dependencias
pip install -r requirements.txt

# Configurar directorios
python run_pipeline.py --setup-only
```

### **2. Ejecución Básica**
```bash
# Pipeline completo
python run_pipeline.py --mode all

# Solo dashboard
python run_pipeline.py --mode dashboard
```

### **3. Acceso al Dashboard**
- **URL**: http://localhost:8051
- **Puerto por defecto**: 8051
- **Cambiar puerto**: `--port 8052`

## 📈 Casos de Uso Principales

### 🎯 **Análisis de Marca**
```bash
python run_pipeline.py --mode streaming --keywords "@marca #marca"
```

### 📊 **Análisis de Campañas**
```bash
python run_pipeline.py --mode sentiment --hashtags "#campaña2024"
```

### 🔍 **Análisis de Competencia**
```bash
python run_pipeline.py --mode sentiment --competitors "@comp1 @comp2"
```

### ⚠️ **Detección de Crisis**
```bash
python run_pipeline.py --mode streaming --alert-keywords "crisis problema"
```

## 🔧 Configuración Avanzada

### **Variables de Entorno**
```bash
export SPARK_HOME="/path/to/spark"
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
export TWITTER_API_KEY="your_api_key"
```

### **Configuración de Spark**
```yaml
spark.driver.memory 4g
spark.executor.memory 4g
spark.sql.adaptive.enabled true
```

### **Configuración de Modelos**
```python
bert_config = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'confidence_threshold': 0.7
}
```

## 📊 Métricas y Monitoreo

### **Métricas de Rendimiento**
- **Throughput**: Tweets procesados por segundo
- **Latency**: Tiempo de procesamiento
- **Accuracy**: Precisión del modelo
- **Memory Usage**: Uso de memoria

### **Métricas de Negocio**
- **Sentiment Distribution**: Distribución de sentimientos
- **Trend Analysis**: Análisis de tendencias
- **Engagement Metrics**: Métricas de engagement
- **Crisis Detection**: Detección de crisis

## 🐛 Troubleshooting

### **Problemas Comunes**

#### 1. **Error de Importación**
```bash
# Solución
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
```

#### 2. **Error de Spark**
```bash
# Solución
export SPARK_HOME="/path/to/spark"
```

#### 3. **Error de Puerto**
```bash
# Solución
python run_pipeline.py --mode dashboard --port 8052
```

### **Debug Mode**
```bash
# Logging detallado
python run_pipeline.py --mode all --log-level DEBUG

# Ver logs
tail -f logs/twitter_analysis.log
```

## 📚 Recursos Adicionales

### **Documentación Externa**
- [PySpark Documentation](https://spark.apache.org/docs/latest/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Dash Documentation](https://dash.plotly.com/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

### **Comunidad**
- **GitHub Issues**: Reportar bugs y solicitar features
- **Stack Overflow**: Preguntas técnicas
- **Discord Community**: Comunidad de desarrolladores

### **Tutoriales**
- **Getting Started**: Guía de inicio rápido
- **Advanced Configuration**: Configuración avanzada
- **API Reference**: Referencia de APIs
- **Best Practices**: Mejores prácticas

## 🔄 Actualizaciones

### **Versión Actual**: 1.0.0
- **Fecha**: Enero 2024
- **Cambios Principales**:
  - Implementación completa del pipeline
  - Dashboard interactivo
  - Análisis de streaming
  - Documentación completa

### **Próximas Versiones**
- **v1.1.0**: Mejoras en rendimiento
- **v1.2.0**: Nuevas visualizaciones
- **v1.3.0**: Integración con más APIs

## 📞 Contacto y Soporte

### **Equipo de Desarrollo**
- **Email**: desarrollo@smartretail.com
- **GitHub**: https://github.com/smartretail/twitter-sentiment-analysis
- **Documentación**: https://docs.smartretail.com/twitter-analysis

### **Canales de Soporte**
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: soporte@smartretail.com

---

## 📋 Checklist de Documentación

### ✅ **Completado**
- [x] Documentación funcional completa
- [x] Documentación técnica del orquestador
- [x] Guía de inicio rápido
- [x] Índice de documentación
- [x] Ejemplos de uso
- [x] Troubleshooting
- [x] Configuración avanzada

### 🔄 **En Progreso**
- [ ] Tutoriales en video
- [ ] Casos de uso específicos por industria
- [ ] Guías de integración
- [ ] Documentación de APIs

### 📋 **Pendiente**
- [ ] Documentación de despliegue
- [ ] Guías de optimización
- [ ] Documentación de testing
- [ ] Guías de contribución

---

*Documentación del Sistema de Análisis de Sentimiento en Twitter - SmartRetail* 