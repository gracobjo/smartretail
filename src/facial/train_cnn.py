"""
Training script for facial emotion recognition CNN model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.utils.config import FACIAL_CONFIG, CNN_ARCHITECTURE, TRAINING_CONFIG
from src.utils.data_loader import FacialDataLoader
from src.facial.cnn_model import FacialCNNModel

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    """Main training function."""
    
    # Set random seed
    set_random_seed(TRAINING_CONFIG["random_seed"])
    
    print("=" * 60)
    print("FACIAL EMOTION RECOGNITION - CNN TRAINING")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = FacialDataLoader(FACIAL_CONFIG)
    
    # Load FER2013 dataset
    # Note: You need to download FER2013 dataset and place it in data/fer2013.csv
    data_path = "data/fer2013.csv"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please download FER2013 dataset and place it in the data directory.")
        print("You can download it from: https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    # Load and preprocess data
    print("Loading FER2013 dataset...")
    images, labels, label_encoder = data_loader.load_fer2013(data_path)
    
    # Split data into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=TRAINING_CONFIG["random_seed"], stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=TRAINING_CONFIG["random_seed"], stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create data generators
    train_generator, val_generator = data_loader.create_data_generators(
        X_train, y_train, X_val, y_val
    )
    
    # Initialize model
    print("Initializing CNN model...")
    model = FacialCNNModel(FACIAL_CONFIG, CNN_ARCHITECTURE)
    model.build_model()
    
    # Print model summary
    model.model.summary()
    
    # Train model
    print("Starting training...")
    history = model.train(train_generator, val_generator)
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Plot training history
    print("Plotting training history...")
    model.plot_training_history(save_path="results/facial_training_history.png")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    model.plot_confusion_matrix(
        results['confusion_matrix'], 
        FACIAL_CONFIG["emotions"],
        save_path="results/facial_confusion_matrix.png"
    )
    
    # Save results
    print("Saving results...")
    results_path = "results/facial_evaluation_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 