"""
Multimodal model for emotion analysis combining facial and text features.
Implements various fusion strategies: concatenation, attention, and weighted fusion.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path

class AttentionFusion(layers.Layer):
    """Attention-based fusion layer for multimodal features."""
    
    def __init__(self, hidden_dim, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.attention_weights = None
        
    def build(self, input_shape):
        # Attention mechanism for fusion
        self.attention_dense = layers.Dense(self.hidden_dim, activation='tanh')
        self.attention_output = layers.Dense(1, activation='sigmoid')
        super(AttentionFusion, self).build(input_shape)
        
    def call(self, inputs):
        # inputs should be a list of [facial_features, text_features]
        facial_features, text_features = inputs
        
        # Concatenate features for attention
        combined_features = tf.concat([facial_features, text_features], axis=1)
        
        # Calculate attention weights
        attention_input = self.attention_dense(combined_features)
        attention_weights = self.attention_output(attention_input)
        
        # Apply attention
        weighted_facial = facial_features * attention_weights
        weighted_text = text_features * (1 - attention_weights)
        
        # Combine weighted features
        fused_features = tf.concat([weighted_facial, weighted_text], axis=1)
        
        self.attention_weights = attention_weights
        
        return fused_features

class WeightedFusion(layers.Layer):
    """Weighted fusion layer for multimodal features."""
    
    def __init__(self, **kwargs):
        super(WeightedFusion, self).__init__(**kwargs)
        self.facial_weight = None
        self.text_weight = None
        
    def build(self, input_shape):
        # Learnable weights for fusion
        self.facial_weight = self.add_weight(
            name='facial_weight',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.text_weight = self.add_weight(
            name='text_weight',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        super(WeightedFusion, self).build(input_shape)
        
    def call(self, inputs):
        facial_features, text_features = inputs
        
        # Normalize weights
        total_weight = self.facial_weight + self.text_weight
        facial_weight_norm = self.facial_weight / total_weight
        text_weight_norm = self.text_weight / total_weight
        
        # Weighted combination
        weighted_facial = facial_features * facial_weight_norm
        weighted_text = text_features * text_weight_norm
        
        # Concatenate weighted features
        fused_features = tf.concat([weighted_facial, weighted_text], axis=1)
        
        return fused_features

class MultimodalModel:
    """Multimodal model combining facial and text emotion analysis."""
    
    def __init__(self, config, facial_model, text_model):
        self.config = config
        self.facial_model = facial_model
        self.text_model = text_model
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the multimodal model architecture."""
        
        # Facial input
        facial_input = layers.Input(shape=(*self.facial_model.config["image_size"], 1))
        
        # Text input
        text_input = layers.Input(shape=(self.text_model.config["max_sequence_length"],))
        
        # Extract features from pre-trained models
        facial_features = self.facial_model.model(facial_input)
        text_features = self.text_model.model(text_input)
        
        # Fusion strategy
        if self.config["fusion_method"] == "concatenate":
            # Simple concatenation
            fused_features = layers.Concatenate()([facial_features, text_features])
            
        elif self.config["fusion_method"] == "attention":
            # Attention-based fusion
            fused_features = AttentionFusion(self.config["hidden_dim"])([facial_features, text_features])
            
        elif self.config["fusion_method"] == "weighted":
            # Weighted fusion
            fused_features = WeightedFusion()([facial_features, text_features])
            
        else:
            raise ValueError(f"Unknown fusion method: {self.config['fusion_method']}")
        
        # Dense layers for final classification
        x = layers.Dense(self.config["hidden_dim"], activation='relu')(fused_features)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        x = layers.Dense(self.config["hidden_dim"] // 2, activation='relu')(x)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        
        # Output layer
        output_layer = layers.Dense(
            units=self.facial_model.config["num_classes"],  # Use facial model's num_classes
            activation='softmax'
        )(x)
        
        # Create model
        self.model = Model(inputs=[facial_input, text_input], outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get("early_stopping_patience", 10),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.get("reduce_lr_factor", 0.5),
                patience=self.config.get("reduce_lr_patience", 5),
                min_lr=self.config.get("min_lr", 1e-7),
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.config["model_save_path"],
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_data, val_data):
        """Train the multimodal model."""
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.create_callbacks()
        
        # Prepare data
        train_facial = train_data['facial_images'].reshape(-1, *self.facial_model.config["image_size"], 1)
        train_text = train_data['text_sequences']
        train_labels = train_data['labels']
        
        val_facial = val_data['facial_images'].reshape(-1, *self.facial_model.config["image_size"], 1)
        val_text = val_data['text_sequences']
        val_labels = val_data['labels']
        
        # Train the model
        self.history = self.model.fit(
            [train_facial, train_text],
            train_labels,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=([val_facial, val_text], val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        
        # Prepare test data
        test_facial = test_data['facial_images'].reshape(-1, *self.facial_model.config["image_size"], 1)
        test_text = test_data['text_sequences']
        test_labels = test_data['labels']
        
        # Make predictions
        predictions = self.model.predict([test_facial, test_text])
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        
        # Classification report
        report = classification_report(
            test_labels, 
            predicted_labels, 
            target_names=self.facial_model.config["emotions"],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Multimodal Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Multimodal Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix, emotions, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions
        )
        plt.title('Confusion Matrix - Multimodal Emotion Recognition')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_training_history(self):
        """Save training history to JSON file."""
        if self.history is None:
            return
        
        history_dict = {
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']]
        }
        
        with open(self.config["history_save_path"], 'w') as f:
            json.dump(history_dict, f, indent=4)
    
    def load_model(self, model_path):
        """Load a trained model."""
        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                'AttentionFusion': AttentionFusion,
                'WeightedFusion': WeightedFusion
            }
        )
        return self.model
    
    def predict_multimodal(self, facial_image, text_sequence):
        """Predict emotion for multimodal input."""
        
        # Preprocess facial image
        if len(facial_image.shape) == 2:
            facial_image = facial_image.reshape(1, *self.facial_model.config["image_size"], 1)
        elif len(facial_image.shape) == 3 and facial_image.shape[-1] == 1:
            facial_image = facial_image.reshape(1, *facial_image.shape)
        else:
            facial_image = facial_image.reshape(1, *self.facial_model.config["image_size"], 1)
        
        # Normalize facial image
        facial_image = facial_image.astype(np.float32) / 255.0
        
        # Preprocess text sequence
        if len(text_sequence.shape) == 1:
            text_sequence = text_sequence.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict([facial_image, text_sequence])
        predicted_emotion = self.facial_model.config["emotions"][np.argmax(prediction)]
        confidence = np.max(prediction)
        
        return predicted_emotion, confidence, prediction[0]

class ModelComparison:
    """Class for comparing different models and fusion strategies."""
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.results = {}
        
    def compare_models(self, test_data):
        """Compare all models on the same test data."""
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            if model_name == "multimodal":
                results = model.evaluate(test_data)
            else:
                # For individual models, need to prepare data accordingly
                if model_name == "facial":
                    test_images = test_data['facial_images']
                    test_labels = test_data['labels']
                    results = model.evaluate(test_images, test_labels)
                elif model_name == "text":
                    test_sequences = test_data['text_sequences']
                    test_labels = test_data['labels']
                    results = model.evaluate(test_sequences, test_labels)
            
            self.results[model_name] = results
        
        return self.results
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of model accuracies."""
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """Generate detailed comparison report."""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MULTIMODAL EMOTION ANALYSIS - MODEL COMPARISON")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for model_name, results in self.results.items():
            report_lines.append(f"MODEL: {model_name.upper()}")
            report_lines.append("-" * 30)
            report_lines.append(f"Accuracy: {results['accuracy']:.4f}")
            report_lines.append("")
            
            # Add classification report
            report_lines.append("Classification Report:")
            for emotion, metrics in results['classification_report'].items():
                if isinstance(metrics, dict):
                    report_lines.append(f"  {emotion}:")
                    report_lines.append(f"    Precision: {metrics['precision']:.4f}")
                    report_lines.append(f"    Recall: {metrics['recall']:.4f}")
                    report_lines.append(f"    F1-Score: {metrics['f1-score']:.4f}")
                    report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text 