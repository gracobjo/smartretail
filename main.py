"""
Main script for SmartRetail multimodal emotion analysis.
Runs the complete pipeline including training, evaluation, and visualization.
"""

import os
import sys
import argparse
import json
from pathlib import Path

def setup_environment():
    """Setup the environment and create necessary directories."""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = ["data", "models", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Environment setup completed!")

def run_training_pipeline():
    """Run the complete training pipeline."""
    print("\n" + "="*60)
    print("SMARTRETAIL - MULTIMODAL EMOTION ANALYSIS PIPELINE")
    print("="*60)
    
    # Check if datasets exist
    required_datasets = [
        "data/fer2013.csv",
        "data/emoreact.csv"
    ]
    
    missing_datasets = []
    for dataset in required_datasets:
        if not os.path.exists(dataset):
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print("Missing datasets:")
        for dataset in missing_datasets:
            print(f"  - {dataset}")
        print("\nPlease download the required datasets:")
        print("  - FER2013: https://www.kaggle.com/datasets/msambare/fer2013")
        print("  - EmoReact: Create or download appropriate text emotion dataset")
        return False
    
    print("All datasets found!")
    
    # Run individual model training
    print("\n1. Training Facial CNN Model...")
    try:
        os.system("python src/facial/train_cnn.py")
        print("✓ Facial CNN training completed!")
    except Exception as e:
        print(f"✗ Facial CNN training failed: {e}")
        return False
    
    print("\n2. Training Text RNN Model...")
    try:
        os.system("python src/text/train_rnn.py")
        print("✓ Text RNN training completed!")
    except Exception as e:
        print(f"✗ Text RNN training failed: {e}")
        return False
    
    print("\n3. Training Multimodal Model...")
    try:
        os.system("python src/multimodal/train_multimodal.py")
        print("✓ Multimodal training completed!")
    except Exception as e:
        print(f"✗ Multimodal training failed: {e}")
        return False
    
    return True

def run_evaluation():
    """Run comprehensive evaluation."""
    print("\n4. Running Comprehensive Evaluation...")
    try:
        os.system("python src/utils/evaluate.py")
        print("✓ Evaluation completed!")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False
    
    return True

def run_tests():
    """Run unit tests."""
    print("\n5. Running Unit Tests...")
    try:
        os.system("python -m pytest tests/ -v")
        print("✓ Tests completed!")
    except Exception as e:
        print(f"✗ Tests failed: {e}")
        return False
    
    return True

def generate_report():
    """Generate final report."""
    print("\n6. Generating Final Report...")
    
    # Check if results exist
    results_files = [
        "results/facial_evaluation_results.json",
        "results/text_rnn_evaluation_results.json",
        "results/multimodal_evaluation_results.json"
    ]
    
    available_results = []
    for file_path in results_files:
        if os.path.exists(file_path):
            available_results.append(file_path)
    
    if not available_results:
        print("No evaluation results found!")
        return False
    
    # Generate summary report
    report = {
        "project": "SmartRetail Multimodal Emotion Analysis",
        "status": "completed",
        "models_trained": len(available_results),
        "results_files": available_results
    }
    
    # Add model performance summary
    performance_summary = {}
    for result_file in available_results:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                model_name = result_file.split('/')[-1].split('_')[0]
                performance_summary[model_name] = {
                    "accuracy": data.get("accuracy", "N/A"),
                    "f1_score": data.get("f1_score", "N/A")
                }
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    report["performance_summary"] = performance_summary
    
    # Save report
    with open("results/final_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    print("✓ Final report generated!")
    print(f"Report saved to: results/final_report.json")
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    for model_name, metrics in performance_summary.items():
        print(f"{model_name.upper()}:")
        print(f"  - Accuracy: {metrics['accuracy']}")
        print(f"  - F1-Score: {metrics['f1_score']}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SmartRetail Multimodal Emotion Analysis")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    
    args = parser.parse_args()
    
    if args.all or (not args.setup and not args.train and not args.evaluate and not args.test and not args.report):
        # Run complete pipeline
        print("Running complete SmartRetail pipeline...")
        
        setup_environment()
        
        if run_training_pipeline():
            if run_evaluation():
                if run_tests():
                    generate_report()
                    print("\n🎉 Pipeline completed successfully!")
                    print("Check the 'results/' directory for outputs.")
                else:
                    print("\n❌ Pipeline failed at testing stage.")
            else:
                print("\n❌ Pipeline failed at evaluation stage.")
        else:
            print("\n❌ Pipeline failed at training stage.")
    
    else:
        # Run specific components
        if args.setup:
            setup_environment()
        
        if args.train:
            run_training_pipeline()
        
        if args.evaluate:
            run_evaluation()
        
        if args.test:
            run_tests()
        
        if args.report:
            generate_report()

if __name__ == "__main__":
    main() 