"""
Training script for text emotion recognition RNN/Transformer model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.utils.config import TEXT_CONFIG, RNN_ARCHITECTURE, TRANSFORMER_ARCHITECTURE, TRAINING_CONFIG
from src.utils.data_loader import TextDataLoader
from src.text.rnn_model import TextRNNModel, TextTransformerModel

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
    print("TEXT EMOTION RECOGNITION - RNN/TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = TextDataLoader(TEXT_CONFIG)
    
    # Load EmoReact dataset
    # Note: You need to download EmoReact dataset and place it in data/emoreact.csv
    data_path = "data/emoreact.csv"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please download EmoReact dataset and place it in the data directory.")
        print("You can create a sample dataset or download from appropriate source.")
        return
    
    # Load and preprocess data
    print("Loading EmoReact dataset...")
    sequences, labels, vocabulary, label_encoder = data_loader.load_emoreact(data_path)
    
    # Save tokenizer
    data_loader.save_tokenizer(TEXT_CONFIG["tokenizer_save_path"])
    
    # Split data into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=TRAINING_CONFIG["random_seed"], stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=TRAINING_CONFIG["random_seed"], stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Choose model type (RNN or Transformer)
    model_type = "rnn"  # Change to "transformer" for Transformer model
    
    if model_type == "rnn":
        # Initialize RNN model
        print("Initializing RNN model...")
        model = TextRNNModel(TEXT_CONFIG, RNN_ARCHITECTURE)
        architecture_config = RNN_ARCHITECTURE
    else:
        # Initialize Transformer model
        print("Initializing Transformer model...")
        model = TextTransformerModel(TEXT_CONFIG, TRANSFORMER_ARCHITECTURE)
        architecture_config = TRANSFORMER_ARCHITECTURE
    
    model.build_model()
    
    # Print model summary
    model.model.summary()
    
    # Train model
    print("Starting training...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Plot training history
    print("Plotting training history...")
    model.plot_training_history(save_path=f"results/text_{model_type}_training_history.png")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    model.plot_confusion_matrix(
        results['confusion_matrix'], 
        TEXT_CONFIG["emotions"],
        save_path=f"results/text_{model_type}_confusion_matrix.png"
    )
    
    # Save results
    print("Saving results...")
    results_path = f"results/text_{model_type}_evaluation_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'model_type': model_type
        }, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 