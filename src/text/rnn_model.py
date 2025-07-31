"""
RNN and Transformer models for text emotion analysis using EmoReact dataset.
Implements LSTM, BiLSTM, and Transformer architectures with attention mechanisms.
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

class MultiHeadAttention(layers.Layer):
    """Multi-head attention mechanism for Transformer."""
    
    def __init__(self, num_heads, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.att = MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class TextRNNModel:
    """RNN model for text emotion recognition."""
    
    def __init__(self, config, architecture_config):
        self.config = config
        self.architecture_config = architecture_config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the RNN model architecture."""
        
        # Input layer
        input_layer = layers.Input(shape=(self.config["max_sequence_length"],))
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self.config["vocab_size"],
            output_dim=self.config["embedding_dim"],
            input_length=self.config["max_sequence_length"]
        )(input_layer)
        
        # LSTM layers
        for i, units in enumerate(self.architecture_config["lstm_units"]):
            if self.architecture_config["bidirectional"]:
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=(i < len(self.architecture_config["lstm_units"]) - 1))
                )(x)
            else:
                x = layers.LSTM(
                    units, 
                    return_sequences=(i < len(self.architecture_config["lstm_units"]) - 1)
                )(x)
            
            x = layers.Dropout(0.2)(x)
        
        # Attention mechanism if enabled
        if self.architecture_config["attention"]:
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention_weights = layers.Activation('softmax')(attention)
            attention_weights = layers.RepeatVector(x.shape[-1])(attention_weights)
            attention_weights = layers.Permute([2, 1])(attention_weights)
            
            x = layers.Multiply()([x, attention_weights])
            x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Dense layers
        for dense_config in self.architecture_config["dense_layers"]:
            x = layers.Dense(
                units=dense_config["units"],
                activation=dense_config["activation"]
            )(x)
            if "dropout" in dense_config:
                x = layers.Dropout(dense_config["dropout"])(x)
        
        # Output layer
        output_layer = layers.Dense(
            units=self.config["num_classes"],
            activation='softmax'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
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
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the RNN model."""
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.create_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_labels)
        
        # Classification report
        report = classification_report(
            y_test, 
            predicted_labels, 
            target_names=self.config["emotions"],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_labels)
        
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
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
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
        plt.title('Confusion Matrix - Text Emotion Recognition')
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
        self.model = keras.models.load_model(model_path)
        return self.model
    
    def predict_single_text(self, text_sequence):
        """Predict emotion for a single text sequence."""
        # Ensure correct shape
        if len(text_sequence.shape) == 1:
            text_sequence = text_sequence.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(text_sequence)
        predicted_emotion = self.config["emotions"][np.argmax(prediction)]
        confidence = np.max(prediction)
        
        return predicted_emotion, confidence, prediction[0]

class TextTransformerModel:
    """Transformer model for text emotion recognition."""
    
    def __init__(self, config, architecture_config):
        self.config = config
        self.architecture_config = architecture_config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the Transformer model architecture."""
        
        # Input layer
        input_layer = layers.Input(shape=(self.config["max_sequence_length"],))
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self.config["vocab_size"],
            output_dim=self.config["embedding_dim"],
            input_length=self.config["max_sequence_length"]
        )(input_layer)
        
        # Positional encoding
        pos_encoding = self.positional_encoding(self.config["max_sequence_length"], self.config["embedding_dim"])
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.architecture_config["num_transformer_blocks"]):
            x = TransformerBlock(
                self.config["embedding_dim"],
                self.architecture_config["num_heads"],
                self.architecture_config["ff_dim"],
                self.architecture_config["dropout_rate"]
            )(x, True, None)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in self.architecture_config["mlp_units"]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.architecture_config["dropout_rate"])(x)
        
        # Output layer
        output_layer = layers.Dense(
            units=self.config["num_classes"],
            activation='softmax'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def positional_encoding(self, position, d_model):
        """Generate positional encoding for Transformer."""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        """Get angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
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
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the Transformer model."""
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.create_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_labels)
        
        # Classification report
        report = classification_report(
            y_test, 
            predicted_labels, 
            target_names=self.config["emotions"],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
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