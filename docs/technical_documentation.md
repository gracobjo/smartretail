# SmartRetail - Documentación Técnica Multimodal

## 1. Arquitectura del Sistema

### 1.1 Visión General

El sistema SmartRetail implementa un análisis multimodal de emociones que combina:

- **Análisis Facial**: CNN para clasificación de emociones en imágenes (FER2013)
- **Análisis de Texto**: RNN/Transformer para clasificación de emociones en texto (EmoReact)
- **Integración Multimodal**: Fusión de ambos modelos para clasificación combinada

### 1.2 Arquitectura de Alto Nivel

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Facial CNN    │    │   Text RNN      │    │  Multimodal     │
│   (FER2013)     │    │   (EmoReact)    │    │   Fusion        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Final Output  │
                    │   (Emotion)     │
                    └─────────────────┘
```

## 2. Modelos Implementados

### 2.1 CNN para Análisis Facial

#### Arquitectura
```python
Input (48x48x1) 
    ↓
Conv2D(32, 3x3) + BatchNorm + ReLU
    ↓
MaxPool2D(2x2) + Dropout(0.25)
    ↓
Conv2D(64, 3x3) + BatchNorm + ReLU
    ↓
MaxPool2D(2x2) + Dropout(0.25)
    ↓
Conv2D(128, 3x3) + BatchNorm + ReLU
    ↓
MaxPool2D(2x2) + Dropout(0.25)
    ↓
Attention Layer
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Dense(7, softmax)  # 7 emociones
```

#### Características Clave
- **Input**: Imágenes 48x48 píxeles en escala de grises
- **Data Augmentation**: Rotación, traslación, zoom, flip horizontal
- **Attention Mechanism**: Capa de atención personalizada para features
- **Batch Normalization**: Para estabilizar el entrenamiento
- **Dropout**: Para prevenir overfitting

### 2.2 RNN para Análisis de Texto

#### Arquitectura
```python
Input (max_sequence_length)
    ↓
Embedding(vocab_size, embedding_dim)
    ↓
Bidirectional LSTM(128) + Dropout(0.2)
    ↓
Bidirectional LSTM(64) + Dropout(0.2)
    ↓
Attention Mechanism (opcional)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Dense(64) + Dropout(0.2)
    ↓
Dense(6, softmax)  # 6 emociones
```

#### Características Clave
- **Input**: Secuencias de texto tokenizadas y paddeadas
- **Embedding**: Capa de embedding aprendida
- **Bidirectional LSTM**: Procesamiento bidireccional de secuencias
- **Attention**: Mecanismo de atención para palabras importantes
- **Vocabulary**: Vocabulario construido a partir del corpus

### 2.3 Transformer para Análisis de Texto

#### Arquitectura
```python
Input (max_sequence_length)
    ↓
Embedding(vocab_size, embedding_dim)
    ↓
Positional Encoding
    ↓
Transformer Block × 4:
    ├── Multi-Head Attention
    ├── Add & Norm
    ├── Feed Forward
    └── Add & Norm
    ↓
Global Average Pooling
    ↓
Dense(128) + Dropout(0.1)
    ↓
Dense(64) + Dropout(0.1)
    ↓
Dense(6, softmax)
```

#### Características Clave
- **Multi-Head Attention**: 8 heads de atención
- **Positional Encoding**: Codificación posicional sinusoidal
- **Layer Normalization**: Normalización por capas
- **Feed Forward**: Redes feed-forward con activación ReLU

### 2.4 Modelo Multimodal

#### Estrategias de Fusión

1. **Concatenación Simple**
```python
fused_features = concatenate([facial_features, text_features])
```

2. **Fusión con Atención**
```python
attention_weights = attention_layer([facial_features, text_features])
fused_features = weighted_combination(facial_features, text_features, attention_weights)
```

3. **Fusión Ponderada**
```python
learnable_weights = [facial_weight, text_weight]
fused_features = weighted_sum(facial_features, text_features, learnable_weights)
```

#### Arquitectura Final
```python
[facial_input, text_input]
    ↓
[Facial CNN, Text RNN/Transformer]
    ↓
Fusion Layer (concatenate/attention/weighted)
    ↓
Dense(256) + Dropout(0.3)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Dense(7, softmax)
```

## 3. Preprocesamiento de Datos

### 3.1 Datos Faciales (FER2013)

#### Pipeline de Preprocesamiento
1. **Carga**: CSV con píxeles como strings
2. **Conversión**: String → Array numpy
3. **Reshape**: (2304,) → (48, 48)
4. **Resize**: (48, 48) → target_size
5. **Normalización**: [0, 255] → [0, 1]
6. **Data Augmentation**: Para training

#### Data Augmentation
```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
```

### 3.2 Datos de Texto (EmoReact)

#### Pipeline de Preprocesamiento
1. **Limpieza**: Remover caracteres especiales y números
2. **Tokenización**: NLTK word_tokenize
3. **Vocabulario**: Construir mapping palabra → índice
4. **Secuenciación**: Convertir texto a secuencia de índices
5. **Padding**: Pad sequences a longitud fija

#### Preprocesamiento de Texto
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
```

## 4. Métricas de Evaluación

### 4.1 Métricas Implementadas

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall
- **Confusion Matrix**: Matriz de confusión
- **ROC Curves**: Curvas ROC por clase
- **AUC**: Área bajo la curva ROC

### 4.2 Visualizaciones

- **Training History**: Accuracy y loss durante entrenamiento
- **Confusion Matrix**: Heatmap de clasificaciones
- **ROC Curves**: Curvas ROC para cada emoción
- **Model Comparison**: Comparación de métricas entre modelos
- **Interactive Dashboard**: Dashboard interactivo con Plotly

## 5. Configuración y Hiperparámetros

### 5.1 Configuración Facial
```python
FACIAL_CONFIG = {
    "image_size": (48, 48),
    "num_classes": 7,
    "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2
}
```

### 5.2 Configuración Texto
```python
TEXT_CONFIG = {
    "max_sequence_length": 100,
    "vocab_size": 10000,
    "embedding_dim": 128,
    "num_classes": 6,
    "emotions": ["joy", "sadness", "anger", "fear", "surprise", "neutral"],
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001
}
```

### 5.3 Configuración Multimodal
```python
MULTIMODAL_CONFIG = {
    "fusion_method": "concatenate",  # "concatenate", "attention", "weighted"
    "hidden_dim": 256,
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 0.0005
}
```

## 6. Optimizaciones y Técnicas Avanzadas

### 6.1 Callbacks de Entrenamiento
- **Early Stopping**: Detener entrenamiento si no mejora
- **ReduceLROnPlateau**: Reducir learning rate en plateaus
- **ModelCheckpoint**: Guardar mejor modelo

### 6.2 Regularización
- **Dropout**: En capas densas y LSTM
- **Batch Normalization**: En capas convolucionales
- **Weight Decay**: Regularización L2 implícita

### 6.3 Técnicas de Fusión Avanzadas
- **Cross-Modal Attention**: Atención entre modalidades
- **Gated Fusion**: Fusión con gates aprendibles
- **Hierarchical Fusion**: Fusión jerárquica de features

## 7. Rendimiento Esperado

### 7.1 Métricas Objetivo
- **CNN Facial**: >65% accuracy en FER2013
- **RNN Texto**: >70% accuracy en EmoReact
- **Multimodal**: >75% accuracy combinada

### 7.2 Comparación de Modelos
| Modelo | Accuracy | F1-Score | Tiempo Entrenamiento |
|--------|----------|----------|---------------------|
| CNN Facial | ~68% | ~0.65 | ~2 horas |
| RNN Texto | ~72% | ~0.70 | ~1 hora |
| Multimodal | ~78% | ~0.75 | ~4 horas |

## 8. Uso y Deployment

### 8.1 Entrenamiento
```bash
# Entrenar modelo facial
python src/facial/train_cnn.py

# Entrenar modelo texto
python src/text/train_rnn.py

# Entrenar modelo multimodal
python src/multimodal/train_multimodal.py

# Pipeline completo
python main.py --all
```

### 8.2 Evaluación
```bash
# Evaluación completa
python src/utils/evaluate.py

# Tests unitarios
python -m pytest tests/ -v
```

### 8.3 Predicción
```python
# Predicción facial
emotion, confidence, probs = facial_model.predict_single_image(image)

# Predicción texto
emotion, confidence, probs = text_model.predict_single_text(sequence)

# Predicción multimodal
emotion, confidence, probs = multimodal_model.predict_multimodal(image, text)
```

## 9. Extensibilidad y Mejoras Futuras

### 9.1 Posibles Mejoras
- **Arquitecturas Avanzadas**: ResNet, EfficientNet para facial
- **Transformers**: BERT, RoBERTa para texto
- **Ensemble Methods**: Voting, stacking de modelos
- **Transfer Learning**: Pre-trained models
- **Real-time Processing**: Optimización para tiempo real

### 9.2 Nuevas Modalidades
- **Audio**: Análisis de voz y entonación
- **Video**: Análisis temporal de expresiones
- **Physiological**: Datos de sensores (HRV, GSR)

### 9.3 Aplicaciones
- **Retail Analytics**: Análisis de emociones en tiendas
- **Customer Service**: Detección de satisfacción
- **Healthcare**: Monitoreo de estado emocional
- **Education**: Análisis de engagement

## 10. Consideraciones Técnicas

### 10.1 Requisitos de Sistema
- **GPU**: Recomendado para entrenamiento (8GB+ VRAM)
- **RAM**: 16GB+ para datasets grandes
- **Storage**: 10GB+ para datasets y modelos

### 10.2 Optimizaciones
- **Mixed Precision**: Para acelerar entrenamiento
- **Data Pipeline**: tf.data para eficiencia
- **Model Compression**: Quantization, pruning
- **Batch Processing**: Procesamiento por lotes

### 10.3 Monitoreo
- **TensorBoard**: Visualización de entrenamiento
- **MLflow**: Tracking de experimentos
- **Weights & Biases**: Experiment tracking
- **Custom Logging**: Logs personalizados

Esta documentación proporciona una visión completa de la arquitectura, implementación y uso del sistema SmartRetail para análisis multimodal de emociones. 