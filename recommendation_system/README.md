# Sistema de Recomendación Híbrido - SmartRetail

## Descripción del Proyecto

Este sistema implementa un recomendador híbrido que combina:

1. **Filtrado Colaborativo**: Basado en similitud de usuarios y productos
2. **Filtrado Basado en Contenido**: Usando embeddings de productos
3. **Sistema Híbrido**: Fusión inteligente de ambos enfoques

## Características Principales

### Algoritmos Implementados

#### Filtrado Colaborativo
- **User-Based CF**: Recomendaciones basadas en usuarios similares
- **Item-Based CF**: Recomendaciones basadas en productos similares
- **Matrix Factorization**: SVD, NMF para factorización de matrices
- **Neighborhood Methods**: K-NN con diferentes métricas de similitud

#### Filtrado Basado en Contenido
- **TF-IDF**: Para características textuales de productos
- **Word Embeddings**: Word2Vec, GloVe para representación semántica
- **Product Embeddings**: Embeddings aprendidos de características de productos
- **Content Similarity**: Similitud basada en características de productos

#### Sistema Híbrido
- **Weighted Hybrid**: Combinación ponderada de recomendaciones
- **Switching Hybrid**: Selección dinámica del mejor método
- **Cascade Hybrid**: Aplicación secuencial de métodos
- **Feature Combination**: Fusión a nivel de características

### Métricas de Evaluación

- **Precision@k**: Precisión en las top-k recomendaciones
- **Recall@k**: Recall en las top-k recomendaciones
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Diversidad**: Variedad en las recomendaciones
- **Coverage**: Cobertura de productos recomendados
- **Serendipity**: Sorpresa en las recomendaciones

## Estructura del Proyecto

```
recommendation_system/
├── src/
│   ├── collaborative/     # Filtrado colaborativo
│   ├── content_based/     # Filtrado basado en contenido
│   ├── hybrid/           # Sistema híbrido
│   ├── evaluation/       # Métricas de evaluación
│   └── utils/            # Utilidades
├── data/                 # Datasets
├── models/               # Modelos entrenados
├── results/              # Resultados y visualizaciones
├── notebooks/            # Jupyter notebooks
├── tests/                # Tests unitarios
└── docs/                 # Documentación
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento de Modelos

```python
from src.hybrid.hybrid_recommender import HybridRecommender

# Inicializar recomendador híbrido
recommender = HybridRecommender()

# Entrenar modelos
recommender.train(user_item_matrix, product_features)

# Generar recomendaciones
recommendations = recommender.recommend(user_id, n_recommendations=10)
```

### Evaluación

```python
from src.evaluation.evaluator import RecommendationEvaluator

# Evaluar recomendaciones
evaluator = RecommendationEvaluator()
metrics = evaluator.evaluate(recommendations, ground_truth)
```

## Configuración

### Parámetros del Sistema

```python
HYBRID_CONFIG = {
    "collaborative_weight": 0.6,
    "content_weight": 0.4,
    "similarity_metric": "cosine",
    "n_neighbors": 50,
    "n_factors": 100,
    "embedding_dim": 128
}
```

## Resultados Esperados

- **Precision@10**: >0.3
- **Recall@10**: >0.2
- **Diversidad**: >0.7
- **Coverage**: >0.8

## Tecnologías Utilizadas

- **Scikit-learn**: Algoritmos de ML y métricas
- **Pandas**: Manipulación de datos
- **TensorFlow/Keras**: Deep learning para embeddings
- **NLTK**: Procesamiento de texto
- **Plotly**: Visualizaciones interactivas

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

- Nuevos algoritmos de filtrado colaborativo
- Diferentes técnicas de embeddings
- Métricas de evaluación adicionales
- Integración con sistemas externos 