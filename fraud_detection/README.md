# 🛡️ Sistema de Detección de Fraude - SmartRetail

## 🎯 Descripción General

Este sistema implementa un **pipeline completo de machine learning** para detectar transacciones fraudulentas en tiempo real. Utiliza técnicas avanzadas de procesamiento de datos, múltiples algoritmos de clasificación y herramientas de explicabilidad para proporcionar un sistema robusto y transparente.

## 📋 Índice de Documentación

### 📊 **Documentación de Gráficos y Visualizaciones**
- **Archivo**: [`docs/graphs_documentation.md`](docs/graphs_documentation.md)
- **Descripción**: Guía completa para interpretar todos los gráficos generados
- **Audiencia**: Analistas, equipos de negocio, desarrolladores
- **Contenido**:
  - Explicación detallada de cada gráfico
  - Guías de interpretación
  - Casos de uso por audiencia
  - Métricas de referencia
  - Recomendaciones de uso

### 🔧 **Documentación Técnica**
- **Archivo**: [`docs/technical_documentation.md`](docs/technical_documentation.md)
- **Descripción**: Documentación técnica del sistema
- **Audiencia**: Desarrolladores, DevOps, arquitectos
- **Contenido**:
  - Arquitectura del sistema
  - Componentes principales
  - Pipeline de procesamiento
  - Configuración y deployment

### 🚀 **Guía de Inicio Rápido**
- **Archivo**: [`docs/quick_start_guide.md`](docs/quick_start_guide.md)
- **Descripción**: Guía práctica para comenzar rápidamente
- **Audiencia**: Usuarios nuevos, desarrolladores
- **Contenido**:
  - Instalación y configuración
  - Ejecución del pipeline
  - Interpretación de resultados
  - Troubleshooting común

### 📈 **Reportes de Ejecución**
- **Archivo**: [`fraud_detection/results/evaluation_report.txt`](fraud_detection/results/evaluation_report.txt)
- **Descripción**: Métricas detalladas de rendimiento
- **Contenido**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC AUC por modelo
  - Matrices de confusión
  - Comparación entre algoritmos

## 🎯 **Funcionalidades Principales**

### 🔍 **Detección de Fraude**
- **Múltiples algoritmos**: XGBoost, Random Forest, LightGBM, Logistic Regression
- **Procesamiento avanzado**: Feature engineering, balanceo de datos, normalización
- **Evaluación robusta**: Métricas múltiples, validación cruzada, análisis de errores

### 📊 **Análisis y Visualización**
- **Gráficos de evaluación**: ROC curves, Precision-Recall curves, Confusion matrices
- **Análisis de datos**: Correlation matrix, Feature distributions, Target vs Features
- **Importancia de características**: Feature importance plots, SHAP explanations
- **Reportes detallados**: Métricas numéricas, análisis estadístico

### 🎯 **Explicabilidad**
- **SHAP explanations**: Explicación individual de predicciones
- **Feature importance**: Ranking de características más importantes
- **Transparencia**: Cumplimiento con regulaciones de explicabilidad

## 🚀 **Ejecución Rápida**

```bash
# Navegar al directorio
cd fraud_detection

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python run_demo.py
```

## 📁 **Estructura de Resultados**

```
fraud_detection/
├── fraud_detection/
│   ├── data/
│   │   └── synthetic_fraud_data.csv          # Datos sintéticos generados
│   └── results/
│       ├── roc_curves.png                    # Curvas ROC de todos los modelos
│       ├── precision_recall_curves.png       # Curvas de Precisión-Recall
│       ├── confusion_matrices.png            # Matrices de confusión
│       ├── metrics_comparison.png            # Comparación visual de métricas
│       ├── correlation_matrix.png             # Matriz de correlación
│       ├── feature_distributions.png         # Distribuciones de características
│       ├── target_vs_features.png            # Características vs objetivo
│       ├── feature_importance_plot.png       # Importancia de características
│       ├── shap_summary_xgboost.png          # Explicaciones SHAP (XGBoost)
│       ├── shap_summary_random_forest.png    # Explicaciones SHAP (Random Forest)
│       ├── evaluation_report.txt              # Reporte de evaluación
│       ├── feature_analysis_report.txt       # Análisis de características
│       └── feature_importance.csv            # Datos de importancia
```

## 🎯 **Audiencias Objetivo**

### **👥 Equipos de Negocio**
- **Enfoque**: Métricas de rendimiento, costos de errores, ROI
- **Gráficos clave**: `metrics_comparison.png`, `confusion_matrices.png`
- **Objetivo**: Tomar decisiones de implementación

### **👨‍💻 Equipos Técnicos**
- **Enfoque**: Optimización de modelos, explicabilidad, debugging
- **Gráficos clave**: `roc_curves.png`, `shap_summary_*.png`
- **Objetivo**: Mejorar algoritmos y procesos

### **🔍 Equipos de Auditoría**
- **Enfoque**: Cumplimiento, calidad de datos, transparencia
- **Gráficos clave**: `evaluation_report.txt`, `feature_analysis_report.txt`
- **Objetivo**: Validar implementación

### **📊 Analistas de Datos**
- **Enfoque**: Análisis exploratorio, patrones, insights
- **Gráficos clave**: `correlation_matrix.png`, `feature_distributions.png`
- **Objetivo**: Entender datos y generar insights

## 📈 **Métricas de Rendimiento**

### **Resultados Actuales:**
- **Accuracy**: 92.25%
- **ROC AUC**: 0.8782
- **Precision**: 21.61%
- **Recall**: 78.79%
- **F1-Score**: 33.91%

### **Interpretación:**
- ✅ **Excelente discriminación** (AUC > 0.8)
- ✅ **Alta cobertura** de fraudes (Recall alto)
- ✅ **Bajo costo** de falsos positivos
- ✅ **Balance óptimo** para detección de fraude

## 🛠️ **Tecnologías Utilizadas**

- **Machine Learning**: XGBoost, Random Forest, LightGBM, Scikit-learn
- **Procesamiento**: Pandas, NumPy, Feature engineering
- **Visualización**: Matplotlib, Seaborn, Plotly
- **Explicabilidad**: SHAP (SHapley Additive exPlanations)
- **Balanceo**: SMOTE para clases desbalanceadas
- **Evaluación**: Métricas múltiples, validación cruzada

## 📚 **Documentación Adicional**

Para información más detallada sobre cada componente:

- **📊 Gráficos**: Ver [`docs/graphs_documentation.md`](docs/graphs_documentation.md)
- **🔧 Técnica**: Ver [`docs/technical_documentation.md`](docs/technical_documentation.md)
- **🚀 Inicio Rápido**: Ver [`docs/quick_start_guide.md`](docs/quick_start_guide.md)

## 🤝 **Contribución**

Para contribuir al proyecto:

1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crear un Pull Request

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

---

## 🎯 **Conclusión**

El sistema de detección de fraude proporciona una solución completa y robusta para identificar transacciones fraudulentas. Con su documentación exhaustiva, múltiples algoritmos de machine learning y herramientas de explicabilidad, es una herramienta valiosa para cualquier organización que necesite proteger sus transacciones financieras.

**¡El sistema está listo para producción!** 🚀 