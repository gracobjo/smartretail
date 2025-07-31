"""
Training script for multimodal emotion recognition model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.utils.config import (
    FACIAL_CONFIG, TEXT_CONFIG, MULTIMODAL_CONFIG, 
    CNN_ARCHITECTURE, RNN_ARCHITECTURE, TRAINING_CONFIG
)
from src.utils.data_loader import MultimodalDataLoader
from src.facial.cnn_model import FacialCNNModel
from src.text.rnn_model import TextRNNModel
from src.multimodal.multimodal_model import MultimodalModel, ModelComparison

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
    print("MULTIMODAL EMOTION RECOGNITION - TRAINING")
    print("=" * 60)
    
    # Initialize multimodal data loader
    data_loader = MultimodalDataLoader(FACIAL_CONFIG, TEXT_CONFIG)
    
    # Load datasets
    facial_data_path = "data/fer2013.csv"
    text_data_path = "data/emoreact.csv"
    
    if not os.path.exists(facial_data_path):
        print(f"Facial dataset not found at {facial_data_path}")
        print("Please download FER2013 dataset and place it in the data directory.")
        return
    
    if not os.path.exists(text_data_path):
        print(f"Text dataset not found at {text_data_path}")
        print("Please download EmoReact dataset and place it in the data directory.")
        return
    
    # Load multimodal data
    print("Loading multimodal datasets...")
    data_dict = data_loader.load_multimodal_data(facial_data_path, text_data_path)
    
    # Create multimodal dataset
    multimodal_data = data_loader.create_multimodal_dataset(data_dict)
    
    train_data = multimodal_data['train']
    test_data = multimodal_data['test']
    
    # Split test data into validation and test
    indices = np.arange(len(test_data['labels']))
    val_indices, test_indices = train_test_split(
        indices, test_size=0.5, random_state=TRAINING_CONFIG["random_seed"], 
        stratify=test_data['labels']
    )
    
    val_data = {
        'facial_images': test_data['facial_images'][val_indices],
        'text_sequences': test_data['text_sequences'][val_indices],
        'labels': test_data['labels'][val_indices]
    }
    
    test_data = {
        'facial_images': test_data['facial_images'][test_indices],
        'text_sequences': test_data['text_sequences'][test_indices],
        'labels': test_data['labels'][test_indices]
    }
    
    print(f"Training samples: {len(train_data['labels'])}")
    print(f"Validation samples: {len(val_data['labels'])}")
    print(f"Test samples: {len(test_data['labels'])}")
    
    # Train individual models first (if not already trained)
    print("Training individual models...")
    
    # Train facial model
    print("Training facial CNN model...")
    facial_model = FacialCNNModel(FACIAL_CONFIG, CNN_ARCHITECTURE)
    facial_model.build_model()
    
    # For demonstration, we'll use a simplified training
    # In practice, you'd load pre-trained models or train them separately
    print("Note: Using pre-built models for demonstration")
    
    # Train text model
    print("Training text RNN model...")
    text_model = TextRNNModel(TEXT_CONFIG, RNN_ARCHITECTURE)
    text_model.build_model()
    
    # Initialize multimodal model
    print("Initializing multimodal model...")
    multimodal_model = MultimodalModel(MULTIMODAL_CONFIG, facial_model, text_model)
    multimodal_model.build_model()
    
    # Print model summary
    multimodal_model.model.summary()
    
    # Train multimodal model
    print("Training multimodal model...")
    history = multimodal_model.train(train_data, val_data)
    
    # Evaluate multimodal model
    print("Evaluating multimodal model...")
    multimodal_results = multimodal_model.evaluate(test_data)
    
    print(f"Multimodal Test Accuracy: {multimodal_results['accuracy']:.4f}")
    print(f"Multimodal Test F1-Score: {multimodal_results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Evaluate individual models for comparison
    print("Evaluating individual models...")
    
    # Facial model evaluation
    facial_results = facial_model.evaluate(test_data['facial_images'], test_data['labels'])
    
    # Text model evaluation
    text_results = text_model.evaluate(test_data['text_sequences'], test_data['labels'])
    
    # Compare models
    print("Comparing models...")
    models_dict = {
        'facial': facial_model,
        'text': text_model,
        'multimodal': multimodal_model
    }
    
    comparison = ModelComparison(models_dict)
    comparison_results = comparison.compare_models(test_data)
    
    # Plot comparison
    comparison.plot_comparison(save_path="results/model_comparison.png")
    
    # Generate detailed report
    comparison.generate_report(save_path="results/model_comparison_report.txt")
    
    # Plot training history
    print("Plotting multimodal training history...")
    multimodal_model.plot_training_history(save_path="results/multimodal_training_history.png")
    
    # Plot confusion matrix
    print("Plotting multimodal confusion matrix...")
    multimodal_model.plot_confusion_matrix(
        multimodal_results['confusion_matrix'], 
        FACIAL_CONFIG["emotions"],
        save_path="results/multimodal_confusion_matrix.png"
    )
    
    # Save results
    print("Saving results...")
    results_path = "results/multimodal_evaluation_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'multimodal_accuracy': float(multimodal_results['accuracy']),
            'facial_accuracy': float(facial_results['accuracy']),
            'text_accuracy': float(text_results['accuracy']),
            'multimodal_classification_report': multimodal_results['classification_report'],
            'multimodal_confusion_matrix': multimodal_results['confusion_matrix'].tolist(),
            'fusion_method': MULTIMODAL_CONFIG["fusion_method"]
        }, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print("Multimodal training completed successfully!")

if __name__ == "__main__":
    main() 