# 🚀 Guía de Inicio Rápido - Twitter Sentiment Analysis

## 📋 Resumen Ejecutivo

Este sistema analiza sentimientos en tweets usando **BERT**, **PySpark** y **Dash**. Procesa datos en tiempo real y proporciona visualizaciones interactivas.

## ⚡ Instalación Rápida

### 1. **Requisitos Previos**
```bash
# Python 3.8+
python --version

# Instalar dependencias
pip install -r requirements.txt
```

### 2. **Configuración Inicial**
```bash
# Solo crear directorios
python run_pipeline.py --setup-only
```

## 🎯 Modos de Uso

### **Modo 1: Pipeline Completo** ⭐
```bash
# Ejecutar todo el sistema
python run_pipeline.py --mode all
```
**Resultado**: Genera datos, analiza sentimientos, ejecuta Spark y abre dashboard.

### **Modo 2: Solo Dashboard**
```bash
# Dashboard interactivo
python run_pipeline.py --mode dashboard
```
**Acceso**: http://localhost:8051

### **Modo 3: Solo Análisis**
```bash
# Análisis de sentimiento
python run_pipeline.py --mode sentiment
```

### **Modo 4: Solo Spark**
```bash
# Pipeline distribuido
python run_pipeline.py --mode spark
```

## 📊 Dashboard - Características Principales

### **KPIs en Tiempo Real**
- 📈 Total de tweets procesados
- 😊 Porcentaje de sentimientos positivos
- 😞 Porcentaje de sentimientos negativos
- 🎯 Promedio de confianza

### **Visualizaciones Interactivas**
- 🥧 **Distribución de Sentimientos**: Gráfico circular
- 📈 **Línea de Tiempo**: Evolución temporal
- 📊 **Análisis de Confianza**: Histograma
- 🔤 **Análisis de Hashtags**: Palabras más frecuentes
- 📏 **Longitud vs Sentimiento**: Scatter plot

### **Filtros Dinámicos**
- 📅 **Rango de Fechas**: Seleccionar período
- 🎭 **Filtro de Sentimiento**: Positivo/Negativo/Neutral
- 🎯 **Umbral de Confianza**: Filtrar por confianza
- 📏 **Rango de Longitud**: Filtrar por longitud de texto

## 🔧 Configuración Avanzada

### **Variables de Entorno**
```bash
# Configurar Spark
export SPARK_HOME="/path/to/spark"

# Configurar Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
```

### **Configuración de Puerto**
```bash
# Cambiar puerto del dashboard
python run_pipeline.py --mode dashboard --port 8052
```

## 📈 Casos de Uso Típicos

### **1. Análisis de Marca**
```bash
# Monitorear marca específica
python run_pipeline.py --mode streaming --keywords "@marca #marca"
```

### **2. Análisis de Campaña**
```bash
# Analizar hashtag de campaña
python run_pipeline.py --mode sentiment --hashtags "#campaña2024"
```

### **3. Análisis de Competencia**
```bash
# Comparar competidores
python run_pipeline.py --mode sentiment --competitors "@comp1 @comp2"
```

## 🐛 Solución de Problemas

### **Error: "Module not found"**
```bash
# Solución: Configurar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/twitter_sentiment_analysis"
```

### **Error: "Port already in use"**
```bash
# Solución: Cambiar puerto
python run_pipeline.py --mode dashboard --port 8052
```

### **Error: "Spark not found"**
```bash
# Solución: Configurar SPARK_HOME
export SPARK_HOME="/path/to/spark"
```

## 📊 Interpretación de Resultados

### **Métricas de Sentimiento**
- **Positive**: Sentimientos positivos (😊)
- **Negative**: Sentimientos negativos (😞)
- **Neutral**: Sentimientos neutros (😐)

### **Confianza del Modelo**
- **0.9+**: Muy alta confianza
- **0.7-0.9**: Alta confianza
- **0.5-0.7**: Confianza media
- **<0.5**: Baja confianza

### **Tendencias Temporales**
- **Picos**: Eventos importantes
- **Valles**: Períodos de baja actividad
- **Tendencias**: Cambios sostenidos

## 🎯 Mejores Prácticas

### **Para Análisis de Marca**
1. Usar filtros específicos de marca
2. Monitorear tendencias temporales
3. Analizar impacto de influencers
4. Detectar crisis temprano

### **Para Campañas de Marketing**
1. Definir hashtags específicos
2. Medir engagement en tiempo real
3. Analizar sentimiento por región
4. Evaluar ROI de campaña

### **Para Análisis de Competencia**
1. Monitorear competidores clave
2. Comparar métricas de sentimiento
3. Analizar estrategias exitosas
4. Identificar oportunidades

## 📞 Soporte

### **Logs del Sistema**
```bash
# Ver logs en tiempo real
tail -f logs/twitter_analysis.log
```

### **Debug Mode**
```bash
# Ejecutar con logging detallado
python run_pipeline.py --mode all --log-level DEBUG
```

### **Documentación Completa**
- 📖 [Documentación Funcional](functional_documentation.md)
- 🔧 [Documentación Técnica](run_pipeline_documentation.md)
- 📚 [README Principal](../README.md)

## 🚀 Próximos Pasos

1. **Configurar Twitter API**: Para datos reales
2. **Personalizar Modelos**: Para tu dominio específico
3. **Configurar Alertas**: Para monitoreo automático
4. **Integrar con BI**: Para reportes avanzados

---

*Guía de inicio rápido - SmartRetail Twitter Sentiment Analysis System* 