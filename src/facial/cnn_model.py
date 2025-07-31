"""
CNN model for facial emotion analysis using FER2013 dataset.
Implements a deep convolutional neural network with attention mechanisms.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path

class AttentionLayer(layers.Layer):
    """Custom attention layer for CNN feature maps."""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs shape: (batch_size, height, width, channels)
        batch_size, height, width, channels = tf.shape(inputs)
        
        # Reshape to (batch_size, height*width, channels)
        inputs_reshaped = tf.reshape(inputs, (batch_size, height * width, channels))
        
        # Calculate attention weights
        attention_weights = tf.nn.softmax(
            tf.matmul(inputs_reshaped, self.W) + self.b, axis=1
        )
        
        # Apply attention
        attended_features = tf.reduce_sum(
            inputs_reshaped * attention_weights, axis=1
        )
        
        return attended_features, attention_weights

class FacialCNNModel:
    """CNN model for facial emotion recognition."""
    
    def __init__(self, config, architecture_config):
        self.config = config
        self.architecture_config = architecture_config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the CNN model architecture."""
        
        # Input layer
        input_layer = layers.Input(shape=(*self.config["image_size"], 1))
        
        # First convolutional block
        x = layers.Conv2D(
            filters=self.architecture_config["conv_layers"][0]["filters"],
            kernel_size=self.architecture_config["conv_layers"][0]["kernel_size"],
            activation=self.architecture_config["conv_layers"][0]["activation"],
            padding='same'
        )(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(
            pool_size=self.architecture_config["pooling_layers"][0]["pool_size"],
            strides=self.architecture_config["pooling_layers"][0]["strides"]
        )(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(
            filters=self.architecture_config["conv_layers"][1]["filters"],
            kernel_size=self.architecture_config["conv_layers"][1]["kernel_size"],
            activation=self.architecture_config["conv_layers"][1]["activation"],
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(
            pool_size=self.architecture_config["pooling_layers"][1]["pool_size"],
            strides=self.architecture_config["pooling_layers"][1]["strides"]
        )(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(
            filters=self.architecture_config["conv_layers"][2]["filters"],
            kernel_size=self.architecture_config["conv_layers"][2]["kernel_size"],
            activation=self.architecture_config["conv_layers"][2]["activation"],
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(
            pool_size=self.architecture_config["pooling_layers"][2]["pool_size"],
            strides=self.architecture_config["pooling_layers"][2]["strides"]
        )(x)
        x = layers.Dropout(0.25)(x)
        
        # Attention mechanism
        attended_features, attention_weights = AttentionLayer()(x)
        
        # Dense layers
        for dense_config in self.architecture_config["dense_layers"]:
            x = layers.Dense(
                units=dense_config["units"],
                activation=dense_config["activation"]
            )(attended_features)
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
    
    def train(self, train_generator, val_generator, steps_per_epoch=None):
        """Train the CNN model."""
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.create_callbacks()
        
        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=self.config["epochs"],
            validation_data=val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=len(val_generator),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def evaluate(self, test_images, test_labels):
        """Evaluate the model on test data."""
        
        # Reshape test images for grayscale
        test_images_reshaped = test_images.reshape(-1, *self.config["image_size"], 1)
        
        # Make predictions
        predictions = self.model.predict(test_images_reshaped)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        
        # Classification report
        report = classification_report(
            test_labels, 
            predicted_labels, 
            target_names=self.config["emotions"],
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
        plt.title('Confusion Matrix - Facial Emotion Recognition')
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
        self.model = keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        return self.model
    
    def predict_single_image(self, image):
        """Predict emotion for a single image."""
        # Preprocess image
        if len(image.shape) == 2:
            image = image.reshape(1, *self.config["image_size"], 1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.reshape(1, *image.shape)
        else:
            image = image.reshape(1, *self.config["image_size"], 1)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Predict
        prediction = self.model.predict(image)
        predicted_emotion = self.config["emotions"][np.argmax(prediction)]
        confidence = np.max(prediction)
        
        return predicted_emotion, confidence, prediction[0]

class ResNetFacialModel:
    """ResNet-based model for facial emotion recognition."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build ResNet-based model."""
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config["image_size"], 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        input_layer = layers.Input(shape=(*self.config["image_size"], 1))
        
        # Convert grayscale to RGB (repeat channel 3 times)
        x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(input_layer)
        
        # Pass through ResNet
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output_layer = layers.Dense(
            self.config["num_classes"],
            activation='softmax'
        )(x)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model 