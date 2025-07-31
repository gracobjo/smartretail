# Sistema de Detección de Fraude Financiero - SmartRetail

## Descripción del Proyecto

Este sistema implementa un detector de fraude financiero avanzado que combina:

1. **Modelos Supervisados**: XGBoost y LightGBM para clasificación
2. **Explicabilidad**: SHAP values para interpretación de modelos
3. **Análisis Exploratorio**: Limpieza y visualización de datos
4. **Evaluación Completa**: Métricas precisas y curvas ROC
5. **Optimización**: Tuning de hiperparámetros con Optuna

## Características Principales

### Algoritmos Implementados

#### Modelos de Machine Learning
- **XGBoost**: Gradient boosting con regularización
- **LightGBM**: Gradient boosting optimizado para velocidad
- **Random Forest**: Ensemble de árboles de decisión
- **Logistic Regression**: Modelo lineal interpretable

#### Técnicas de Explicabilidad
- **SHAP Values**: Shapley Additive Explanations
- **Feature Importance**: Importancia de variables
- **Partial Dependence Plots**: Dependencias parciales
- **LIME**: Local Interpretable Model-agnostic Explanations

#### Preprocesamiento de Datos
- **Limpieza**: Manejo de valores faltantes y outliers
- **Feature Engineering**: Creación de variables derivadas
- **Balanceo**: Técnicas para datos desbalanceados
- **Escalado**: Normalización de variables

### Métricas de Evaluación

- **Precision**: Precisión de las predicciones positivas
- **Recall**: Sensibilidad del modelo
- **F1-Score**: Media armónica de precision y recall
- **ROC Curve**: Curva de características operativas
- **AUC**: Área bajo la curva ROC
- **Confusion Matrix**: Matriz de confusión detallada

## Estructura del Proyecto

```
fraud_detection/
├── src/
│   ├── data_processing/    # Limpieza y preprocesamiento
│   ├── feature_engineering/ # Ingeniería de características
│   ├── models/            # Modelos de ML
│   ├── explainability/    # SHAP y explicabilidad
│   ├── evaluation/        # Métricas y evaluación
│   └── utils/             # Utilidades
├── data/                  # Datasets
├── models/                # Modelos entrenados
├── results/               # Resultados y visualizaciones
├── notebooks/             # Jupyter notebooks
├── tests/                 # Tests unitarios
└── docs/                  # Documentación
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento de Modelos

```python
from src.models.fraud_detector import FraudDetector

# Inicializar detector
detector = FraudDetector()

# Entrenar modelo
detector.train(X_train, y_train)

# Generar predicciones
predictions = detector.predict(X_test)

# Obtener explicaciones SHAP
explanations = detector.explain_predictions(X_test)
```

### Evaluación

```python
from src.evaluation.evaluator import FraudEvaluator

# Evaluar modelo
evaluator = FraudEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, probabilities)
```

## Configuración

### Parámetros del Sistema

```python
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
}
```

## Resultados Esperados

- **Precision**: >0.85
- **Recall**: >0.80
- **F1-Score**: >0.82
- **AUC**: >0.90
- **False Positive Rate**: <0.15

## Tecnologías Utilizadas

- **XGBoost/LightGBM**: Modelos de gradient boosting
- **SHAP**: Explicabilidad de modelos
- **Scikit-learn**: Preprocesamiento y métricas
- **Pandas/Seaborn**: Análisis exploratorio
- **Optuna**: Optimización de hiperparámetros
- **Imbalanced-learn**: Balanceo de datos

## Características Avanzadas

### Selección de Variables
- Análisis de correlación
- Feature importance ranking
- Recursive feature elimination
- SHAP-based feature selection

### Tuning de Hiperparámetros
- Optimización bayesiana con Optuna
- Cross-validation estratificado
- Grid search y random search
- Early stopping

### Interpretabilidad
- SHAP summary plots
- Force plots para casos individuales
- Waterfall plots
- Dependence plots

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

- Nuevos algoritmos de ML
- Técnicas de explicabilidad adicionales
- Métricas de evaluación personalizadas
- Integración con sistemas externos 