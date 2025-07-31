"""
Main script for SmartRetail hybrid recommendation system.
Orchestrates the entire pipeline from data loading to evaluation.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import *
from utils.data_loader import RecommendationDataLoader, SyntheticDataGenerator
from collaborative.collaborative_filtering import (
    UserBasedCF, ItemBasedCF, MatrixFactorization, CollaborativeFilteringEnsemble
)
from content_based.content_based_filtering import (
    TFIDFContentBased, WordEmbeddingsContentBased, ProductEmbeddingsModel, ContentBasedEnsemble
)
from hybrid.hybrid_recommender import HybridRecommender
from evaluation.evaluator import RecommendationEvaluator

def setup_environment():
    """Setup the environment and create necessary directories."""
    print("Setting up SmartRetail recommendation system environment...")
    
    # Create directories
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)
    
    print("Environment setup completed!")

def generate_synthetic_data():
    """Generate synthetic data for demonstration."""
    print("Generating synthetic data...")
    
    # Initialize data generator
    generator = SyntheticDataGenerator(n_users=1000, n_items=500, n_interactions=10000)
    
    # Generate interaction data
    interactions_df = generator.generate_interaction_data()
    interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)
    
    # Generate product features
    products_df = generator.generate_product_features()
    products_df.to_csv(DATA_DIR / "products.csv", index=False)
    
    print("Synthetic data generated successfully!")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Products: {len(products_df)}")

def train_models():
    """Train all recommendation models."""
    print("Training recommendation models...")
    
    # Load data
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    
    # Load interaction data
    user_item_matrix, user_encoder, item_encoder = data_loader.load_interaction_data(
        DATA_DIR / "interactions.csv"
    )
    
    # Load product features
    product_features = data_loader.load_product_features(DATA_DIR / "products.csv")
    
    # Create user profiles
    user_profiles = data_loader.create_user_profiles(user_item_matrix, product_features)
    
    # Split data
    train_matrix, test_matrix, train_interactions, test_interactions = data_loader.split_data(
        user_item_matrix, test_size=0.2
    )
    
    # Train collaborative filtering models
    print("\n" + "="*50)
    print("TRAINING COLLABORATIVE FILTERING MODELS")
    print("="*50)
    
    collaborative_models = {}
    
    # User-based CF
    user_based_cf = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    user_based_cf.fit(train_matrix)
    collaborative_models['user_based'] = user_based_cf
    
    # Item-based CF
    item_based_cf = ItemBasedCF(COLLABORATIVE_CONFIG['item_based'])
    item_based_cf.fit(train_matrix)
    collaborative_models['item_based'] = item_based_cf
    
    # SVD
    svd_model = MatrixFactorization(COLLABORATIVE_CONFIG['svd'], method='svd')
    svd_model.fit(train_matrix)
    collaborative_models['svd'] = svd_model
    
    # NMF
    nmf_model = MatrixFactorization(COLLABORATIVE_CONFIG['nmf'], method='nmf')
    nmf_model.fit(train_matrix)
    collaborative_models['nmf'] = nmf_model
    
    # Collaborative ensemble
    collaborative_ensemble = CollaborativeFilteringEnsemble(COLLABORATIVE_CONFIG)
    collaborative_ensemble.fit(train_matrix)
    collaborative_models['ensemble'] = collaborative_ensemble
    
    # Train content-based models
    print("\n" + "="*50)
    print("TRAINING CONTENT-BASED MODELS")
    print("="*50)
    
    content_models = {}
    
    # TF-IDF
    tfidf_model = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    tfidf_model.fit(product_features)
    content_models['tfidf'] = tfidf_model
    
    # Word embeddings
    word_embeddings_model = WordEmbeddingsContentBased(CONTENT_BASED_CONFIG['word_embeddings'])
    word_embeddings_model.fit(product_features)
    content_models['word_embeddings'] = word_embeddings_model
    
    # Product embeddings
    product_embeddings_model = ProductEmbeddingsModel(CONTENT_BASED_CONFIG['product_embeddings'])
    product_embeddings_model.fit(product_features)
    content_models['product_embeddings'] = product_embeddings_model
    
    # Content ensemble
    content_ensemble = ContentBasedEnsemble(CONTENT_BASED_CONFIG)
    content_ensemble.fit(product_features)
    content_models['ensemble'] = content_ensemble
    
    # Train hybrid models
    print("\n" + "="*50)
    print("TRAINING HYBRID MODELS")
    print("="*50)
    
    hybrid_models = {}
    
    # Create hybrid recommenders with different strategies
    for strategy in ['weighted', 'switching', 'cascade', 'feature_combination']:
        hybrid_recommender = HybridRecommender(
            collaborative_ensemble, content_ensemble, HYBRID_CONFIG
        )
        hybrid_recommender.set_strategy(strategy)
        hybrid_models[strategy] = hybrid_recommender
    
    # Save models and data
    if MODEL_CONFIG['save_models']:
        print("\nSaving models...")
        
        # Save collaborative models
        for name, model in collaborative_models.items():
            model_path = MODELS_DIR / f"collaborative_{name}.joblib"
            import joblib
            joblib.dump(model, model_path)
        
        # Save content models
        for name, model in content_models.items():
            model_path = MODELS_DIR / f"content_{name}.joblib"
            joblib.dump(model, model_path)
        
        # Save hybrid models
        for name, model in hybrid_models.items():
            model_path = MODELS_DIR / f"hybrid_{name}.joblib"
            joblib.dump(model, model_path)
        
        # Save data
        train_matrix.to_csv(MODELS_DIR / "train_matrix.csv")
        test_matrix.to_csv(MODELS_DIR / "test_matrix.csv")
        
        # Save user profiles
        with open(MODELS_DIR / "user_profiles.json", 'w') as f:
            json.dump(user_profiles, f, default=str)
    
    return {
        'collaborative_models': collaborative_models,
        'content_models': content_models,
        'hybrid_models': hybrid_models,
        'train_matrix': train_matrix,
        'test_matrix': test_matrix,
        'user_profiles': user_profiles,
        'product_features': product_features
    }

def evaluate_models(models_data):
    """Evaluate all trained models."""
    print("\n" + "="*50)
    print("EVALUATING MODELS")
    print("="*50)
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(EVALUATION_CONFIG)
    
    # Get test data
    test_matrix = models_data['test_matrix']
    user_profiles = models_data['user_profiles']
    
    # Generate recommendations for test users
    test_users = list(test_matrix.index)[:50]  # Evaluate first 50 users
    
    all_results = {}
    
    # Evaluate collaborative models
    print("\nEvaluating collaborative filtering models...")
    collaborative_results = {}
    
    for name, model in models_data['collaborative_models'].items():
        print(f"Evaluating {name}...")
        
        recommendations = {}
        for user_id in test_users:
            try:
                recs = model.recommend(user_id, 10)
                recommendations[user_id] = recs
            except Exception as e:
                print(f"Error with {name} for user {user_id}: {e}")
                continue
        
        # Create ground truth
        ground_truth = {}
        for user_id in test_users:
            user_ratings = test_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            ground_truth[user_id] = rated_items
        
        # Evaluate
        results = evaluator.evaluate(recommendations, ground_truth, test_users)
        collaborative_results[name] = results
    
    all_results['collaborative'] = collaborative_results
    
    # Evaluate content-based models
    print("\nEvaluating content-based models...")
    content_results = {}
    
    for name, model in models_data['content_models'].items():
        print(f"Evaluating {name}...")
        
        recommendations = {}
        for user_id in test_users:
            try:
                user_profile = user_profiles.get(user_id, {})
                recs = model.recommend(user_profile, 10)
                recommendations[user_id] = recs
            except Exception as e:
                print(f"Error with {name} for user {user_id}: {e}")
                continue
        
        # Create ground truth
        ground_truth = {}
        for user_id in test_users:
            user_ratings = test_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            ground_truth[user_id] = rated_items
        
        # Evaluate
        results = evaluator.evaluate(recommendations, ground_truth, test_users)
        content_results[name] = results
    
    all_results['content'] = content_results
    
    # Evaluate hybrid models
    print("\nEvaluating hybrid models...")
    hybrid_results = {}
    
    for name, model in models_data['hybrid_models'].items():
        print(f"Evaluating {name}...")
        
        recommendations = {}
        for user_id in test_users:
            try:
                user_profile = user_profiles.get(user_id, {})
                recs = model.recommend(user_id, user_profile, 10)
                recommendations[user_id] = recs
            except Exception as e:
                print(f"Error with {name} for user {user_id}: {e}")
                continue
        
        # Create ground truth
        ground_truth = {}
        for user_id in test_users:
            user_ratings = test_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            ground_truth[user_id] = rated_items
        
        # Evaluate
        results = evaluator.evaluate(recommendations, ground_truth, test_users)
        hybrid_results[name] = results
    
    all_results['hybrid'] = hybrid_results
    
    # Generate visualizations and reports
    print("\nGenerating visualizations and reports...")
    
    # Plot metrics comparison
    evaluator.plot_metrics_comparison(hybrid_results['weighted'], 
                                     save_path=RESULTS_DIR / "metrics_comparison.png")
    
    # Create interactive dashboard
    evaluator.create_interactive_dashboard(hybrid_results['weighted'])
    
    # Generate comprehensive report
    evaluator.generate_report(
        hybrid_results['weighted'],
        model_results={
            'User-Based CF': collaborative_results['user_based'],
            'Item-Based CF': collaborative_results['item_based'],
            'SVD': collaborative_results['svd'],
            'TF-IDF': content_results['tfidf'],
            'Hybrid Weighted': hybrid_results['weighted']
        },
        save_path=RESULTS_DIR / "evaluation_report.txt"
    )
    
    return all_results

def demonstrate_recommendations(models_data):
    """Demonstrate recommendations for different user profiles."""
    print("\n" + "="*50)
    print("DEMONSTRATING RECOMMENDATIONS")
    print("="*50)
    
    # Get models
    hybrid_model = models_data['hybrid_models']['weighted']
    user_profiles = models_data['user_profiles']
    
    # Demonstrate for different user types
    user_types = {
        'new_user': {'min_interactions': 0, 'max_interactions': 5},
        'active_user': {'min_interactions': 6, 'max_interactions': 50},
        'power_user': {'min_interactions': 51, 'max_interactions': float('inf')}
    }
    
    for user_type, criteria in user_types.items():
        print(f"\n{user_type.upper()} RECOMMENDATIONS:")
        print("-" * 30)
        
        # Find users matching this profile
        matching_users = []
        for user_id, profile in user_profiles.items():
            if (criteria['min_interactions'] <= profile['rating_count'] <= 
                criteria['max_interactions']):
                matching_users.append(user_id)
        
        if matching_users:
            # Show recommendations for first matching user
            demo_user = matching_users[0]
            user_profile = user_profiles[demo_user]
            
            print(f"User {demo_user} (interactions: {user_profile['rating_count']})")
            
            # Generate recommendations
            recommendations = hybrid_model.recommend(demo_user, user_profile, 5)
            
            print("Top 5 recommendations:")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"  {i}. Item {item_id} (score: {score:.3f})")
            
            # Get explanations
            explanations = hybrid_model.get_recommendation_explanation(
                demo_user, user_profile, recommendations
            )
            
            print("\nExplanations:")
            for item_id, explanation in explanations.items():
                print(f"  Item {item_id}: {', '.join(explanation['reasons'])}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SmartRetail Hybrid Recommendation System")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--demonstrate", action="store_true", help="Demonstrate recommendations")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    
    args = parser.parse_args()
    
    if args.all or (not args.setup and not args.generate_data and not args.train and 
                   not args.evaluate and not args.demonstrate):
        print("Running complete SmartRetail recommendation pipeline...")
        
        # Setup environment
        setup_environment()
        
        # Generate data if not exists
        if not (DATA_DIR / "interactions.csv").exists():
            generate_synthetic_data()
        
        # Train models
        models_data = train_models()
        
        # Evaluate models
        evaluation_results = evaluate_models(models_data)
        
        # Demonstrate recommendations
        demonstrate_recommendations(models_data)
        
        print("\n🎉 Pipeline completed successfully!")
        print("Check the 'results/' directory for outputs.")
        
    else:
        if args.setup:
            setup_environment()
        
        if args.generate_data:
            generate_synthetic_data()
        
        if args.train:
            models_data = train_models()
        
        if args.evaluate:
            # Load models if not already trained
            if 'models_data' not in locals():
                print("Loading trained models...")
                # In practice, you'd load the saved models here
                models_data = train_models()
            
            evaluation_results = evaluate_models(models_data)
        
        if args.demonstrate:
            # Load models if not already trained
            if 'models_data' not in locals():
                print("Loading trained models...")
                models_data = train_models()
            
            demonstrate_recommendations(models_data)

if __name__ == "__main__":
    main() 