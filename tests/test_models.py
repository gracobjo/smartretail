"""
Unit tests for model classes in SmartRetail multimodal emotion analysis.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.utils.config import FACIAL_CONFIG, TEXT_CONFIG, CNN_ARCHITECTURE, RNN_ARCHITECTURE
from src.facial.cnn_model import FacialCNNModel
from src.text.rnn_model import TextRNNModel
from src.multimodal.multimodal_model import MultimodalModel

class TestFacialCNNModel(unittest.TestCase):
    """Test cases for FacialCNNModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = FacialCNNModel(FACIAL_CONFIG, CNN_ARCHITECTURE)
    
    def test_build_model(self):
        """Test model building."""
        model = self.model.build_model()
        
        # Check if model is created
        self.assertIsNotNone(model)
        
        # Check input shape
        expected_input_shape = (*FACIAL_CONFIG["image_size"], 1)
        actual_input_shape = model.input_shape[1:]
        self.assertEqual(actual_input_shape, expected_input_shape)
        
        # Check output shape
        expected_output_shape = (FACIAL_CONFIG["num_classes"],)
        actual_output_shape = model.output_shape[1:]
        self.assertEqual(actual_output_shape, expected_output_shape)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = self.model.build_model()
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertIsNotNone(model.metrics)
    
    def test_predict_single_image(self):
        """Test single image prediction."""
        # Create a dummy image
        dummy_image = np.random.rand(*FACIAL_CONFIG["image_size"]).astype(np.float32)
        
        # Build model
        self.model.build_model()
        
        # Test prediction
        emotion, confidence, probabilities = self.model.predict_single_image(dummy_image)
        
        # Check outputs
        self.assertIsInstance(emotion, str)
        self.assertIn(emotion, FACIAL_CONFIG["emotions"])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), FACIAL_CONFIG["num_classes"])

class TestTextRNNModel(unittest.TestCase):
    """Test cases for TextRNNModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TextRNNModel(TEXT_CONFIG, RNN_ARCHITECTURE)
    
    def test_build_model(self):
        """Test model building."""
        model = self.model.build_model()
        
        # Check if model is created
        self.assertIsNotNone(model)
        
        # Check input shape
        expected_input_shape = (TEXT_CONFIG["max_sequence_length"],)
        actual_input_shape = model.input_shape[1:]
        self.assertEqual(actual_input_shape, expected_input_shape)
        
        # Check output shape
        expected_output_shape = (TEXT_CONFIG["num_classes"],)
        actual_output_shape = model.output_shape[1:]
        self.assertEqual(actual_output_shape, expected_output_shape)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = self.model.build_model()
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertIsNotNone(model.metrics)
    
    def test_predict_single_text(self):
        """Test single text prediction."""
        # Create a dummy sequence
        dummy_sequence = np.random.randint(0, TEXT_CONFIG["vocab_size"], 
                                         TEXT_CONFIG["max_sequence_length"])
        
        # Build model
        self.model.build_model()
        
        # Test prediction
        emotion, confidence, probabilities = self.model.predict_single_text(dummy_sequence)
        
        # Check outputs
        self.assertIsInstance(emotion, str)
        self.assertIn(emotion, TEXT_CONFIG["emotions"])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), TEXT_CONFIG["num_classes"])

class TestMultimodalModel(unittest.TestCase):
    """Test cases for MultimodalModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.utils.config import MULTIMODAL_CONFIG
        
        # Create dummy models
        self.facial_model = FacialCNNModel(FACIAL_CONFIG, CNN_ARCHITECTURE)
        self.text_model = TextRNNModel(TEXT_CONFIG, RNN_ARCHITECTURE)
        
        # Build models
        self.facial_model.build_model()
        self.text_model.build_model()
        
        self.multimodal_model = MultimodalModel(MULTIMODAL_CONFIG, 
                                               self.facial_model, 
                                               self.text_model)
    
    def test_build_model(self):
        """Test multimodal model building."""
        model = self.multimodal_model.build_model()
        
        # Check if model is created
        self.assertIsNotNone(model)
        
        # Check input shapes
        facial_input_shape = model.input_shape[0][1:]
        text_input_shape = model.input_shape[1][1:]
        
        expected_facial_shape = (*FACIAL_CONFIG["image_size"], 1)
        expected_text_shape = (TEXT_CONFIG["max_sequence_length"],)
        
        self.assertEqual(facial_input_shape, expected_facial_shape)
        self.assertEqual(text_input_shape, expected_text_shape)
        
        # Check output shape
        expected_output_shape = (FACIAL_CONFIG["num_classes"],)
        actual_output_shape = model.output_shape[1:]
        self.assertEqual(actual_output_shape, expected_output_shape)
    
    def test_model_compilation(self):
        """Test multimodal model compilation."""
        model = self.multimodal_model.build_model()
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertIsNotNone(model.metrics)
    
    def test_predict_multimodal(self):
        """Test multimodal prediction."""
        # Create dummy inputs
        dummy_image = np.random.rand(*FACIAL_CONFIG["image_size"]).astype(np.float32)
        dummy_sequence = np.random.randint(0, TEXT_CONFIG["vocab_size"], 
                                         TEXT_CONFIG["max_sequence_length"])
        
        # Build model
        self.multimodal_model.build_model()
        
        # Test prediction
        emotion, confidence, probabilities = self.multimodal_model.predict_multimodal(
            dummy_image, dummy_sequence
        )
        
        # Check outputs
        self.assertIsInstance(emotion, str)
        self.assertIn(emotion, FACIAL_CONFIG["emotions"])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), FACIAL_CONFIG["num_classes"])

class TestDataLoaders(unittest.TestCase):
    """Test cases for data loaders."""
    
    def test_facial_data_loader_initialization(self):
        """Test FacialDataLoader initialization."""
        from src.utils.data_loader import FacialDataLoader
        
        loader = FacialDataLoader(FACIAL_CONFIG)
        
        self.assertEqual(loader.config, FACIAL_CONFIG)
        self.assertEqual(loader.image_size, FACIAL_CONFIG["image_size"])
        self.assertEqual(loader.num_classes, FACIAL_CONFIG["num_classes"])
    
    def test_text_data_loader_initialization(self):
        """Test TextDataLoader initialization."""
        from src.utils.data_loader import TextDataLoader
        
        loader = TextDataLoader(TEXT_CONFIG)
        
        self.assertEqual(loader.config, TEXT_CONFIG)
        self.assertEqual(loader.max_sequence_length, TEXT_CONFIG["max_sequence_length"])
        self.assertEqual(loader.vocab_size, TEXT_CONFIG["vocab_size"])
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        from src.utils.data_loader import TextDataLoader
        
        loader = TextDataLoader(TEXT_CONFIG)
        
        # Test text preprocessing
        test_text = "Hello, World! 123"
        processed_text = loader.preprocess_text(test_text)
        
        # Check that numbers and special characters are removed
        self.assertNotIn("123", processed_text)
        self.assertNotIn(",", processed_text)
        self.assertNotIn("!", processed_text)
        self.assertIn("hello", processed_text)
        self.assertIn("world", processed_text)

class TestConfigurations(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_facial_config(self):
        """Test facial configuration."""
        required_keys = ["image_size", "num_classes", "emotions", "batch_size", "epochs"]
        
        for key in required_keys:
            self.assertIn(key, FACIAL_CONFIG)
    
    def test_text_config(self):
        """Test text configuration."""
        required_keys = ["max_sequence_length", "vocab_size", "embedding_dim", 
                        "num_classes", "emotions", "batch_size", "epochs"]
        
        for key in required_keys:
            self.assertIn(key, TEXT_CONFIG)
    
    def test_cnn_architecture(self):
        """Test CNN architecture configuration."""
        required_keys = ["conv_layers", "pooling_layers", "dense_layers"]
        
        for key in required_keys:
            self.assertIn(key, CNN_ARCHITECTURE)
    
    def test_rnn_architecture(self):
        """Test RNN architecture configuration."""
        required_keys = ["lstm_units", "bidirectional", "attention", "dense_layers"]
        
        for key in required_keys:
            self.assertIn(key, RNN_ARCHITECTURE)

if __name__ == "__main__":
    unittest.main() 