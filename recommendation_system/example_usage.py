"""
Example usage of SmartRetail hybrid recommendation system.
Demonstrates how to train models, generate recommendations, and evaluate performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import *
from utils.data_loader import RecommendationDataLoader, SyntheticDataGenerator
from collaborative.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization
from content_based.content_based_filtering import TFIDFContentBased, WordEmbeddingsContentBased
from hybrid.hybrid_recommender import HybridRecommender
from evaluation.evaluator import RecommendationEvaluator

def example_1_basic_usage():
    """Example 1: Basic usage of the recommendation system."""
    print("=" * 60)
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 60)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(n_users=500, n_items=200, n_interactions=5000)
    interactions_df = generator.generate_interaction_data()
    products_df = generator.generate_product_features()
    
    # Save data
    interactions_df.to_csv("interactions.csv", index=False)
    products_df.to_csv("products.csv", index=False)
    
    # Load data
    print("Loading data...")
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    user_item_matrix, user_encoder, item_encoder = data_loader.load_interaction_data("interactions.csv")
    product_features = data_loader.load_product_features("products.csv")
    
    # Create user profiles
    user_profiles = data_loader.create_user_profiles(user_item_matrix, product_features)
    
    # Train collaborative filtering model
    print("Training collaborative filtering model...")
    user_based_cf = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    user_based_cf.fit(user_item_matrix)
    
    # Train content-based model
    print("Training content-based model...")
    tfidf_model = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    tfidf_model.fit(product_features)
    
    # Create hybrid recommender
    print("Creating hybrid recommender...")
    hybrid_recommender = HybridRecommender(user_based_cf, tfidf_model, HYBRID_CONFIG)
    hybrid_recommender.set_strategy('weighted')
    
    # Generate recommendations for a user
    user_id = 0
    user_profile = user_profiles.get(user_id, {})
    
    print(f"\nGenerating recommendations for user {user_id}...")
    recommendations = hybrid_recommender.recommend(user_id, user_profile, 5)
    
    print("Top 5 recommendations:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"  {i}. Item {item_id} (score: {score:.3f})")
    
    # Get explanations
    explanations = hybrid_recommender.get_recommendation_explanation(
        user_id, user_profile, recommendations
    )
    
    print("\nExplanations:")
    for item_id, explanation in explanations.items():
        print(f"  Item {item_id}: {', '.join(explanation['reasons'])}")

def example_2_model_comparison():
    """Example 2: Compare different recommendation models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MODEL COMPARISON")
    print("=" * 60)
    
    # Load data
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    user_item_matrix, _, _ = data_loader.load_interaction_data("interactions.csv")
    product_features = data_loader.load_product_features("products.csv")
    
    # Split data
    train_matrix, test_matrix, _, _ = data_loader.split_data(user_item_matrix, test_size=0.2)
    
    # Train different models
    models = {}
    
    # Collaborative filtering models
    print("Training collaborative filtering models...")
    models['user_based'] = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    models['user_based'].fit(train_matrix)
    
    models['item_based'] = ItemBasedCF(COLLABORATIVE_CONFIG['item_based'])
    models['item_based'].fit(train_matrix)
    
    models['svd'] = MatrixFactorization(COLLABORATIVE_CONFIG['svd'], method='svd')
    models['svd'].fit(train_matrix)
    
    # Content-based models
    print("Training content-based models...")
    models['tfidf'] = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    models['tfidf'].fit(product_features)
    
    models['word_embeddings'] = WordEmbeddingsContentBased(CONTENT_BASED_CONFIG['word_embeddings'])
    models['word_embeddings'].fit(product_features)
    
    # Evaluate models
    print("Evaluating models...")
    evaluator = RecommendationEvaluator(EVALUATION_CONFIG)
    test_users = list(test_matrix.index)[:20]  # Evaluate first 20 users
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        recommendations = {}
        for user_id in test_users:
            try:
                if model_name in ['tfidf', 'word_embeddings']:
                    # Content-based models need user profiles
                    user_profile = {'rated_items': []}  # Simplified profile
                    recs = model.recommend(user_profile, 10)
                else:
                    # Collaborative filtering models
                    recs = model.recommend(user_id, 10)
                
                recommendations[user_id] = recs
            except Exception as e:
                print(f"Error with {model_name} for user {user_id}: {e}")
                continue
        
        # Create ground truth
        ground_truth = {}
        for user_id in test_users:
            user_ratings = test_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            ground_truth[user_id] = rated_items
        
        # Evaluate
        model_results = evaluator.evaluate(recommendations, ground_truth, test_users)
        results[model_name] = model_results
    
    # Print comparison
    print("\nModel Performance Comparison:")
    print("-" * 50)
    for model_name, model_results in results.items():
        if 'overall' in model_results:
            precision = model_results['overall'].get('precision', 0)
            recall = model_results['overall'].get('recall', 0)
            print(f"{model_name:15s}: Precision={precision:.3f}, Recall={recall:.3f}")

def example_3_hybrid_strategies():
    """Example 3: Compare different hybrid strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: HYBRID STRATEGIES")
    print("=" * 60)
    
    # Load data
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    user_item_matrix, _, _ = data_loader.load_interaction_data("interactions.csv")
    product_features = data_loader.load_product_features("products.csv")
    
    # Create user profiles
    user_profiles = data_loader.create_user_profiles(user_item_matrix, product_features)
    
    # Train base models
    print("Training base models...")
    collaborative_model = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    collaborative_model.fit(user_item_matrix)
    
    content_model = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    content_model.fit(product_features)
    
    # Test different hybrid strategies
    strategies = ['weighted', 'switching', 'cascade', 'feature_combination']
    
    print("\nTesting different hybrid strategies...")
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 30)
        
        hybrid_recommender = HybridRecommender(collaborative_model, content_model, HYBRID_CONFIG)
        hybrid_recommender.set_strategy(strategy)
        
        # Test with different user types
        user_types = {
            'new_user': {'rating_count': 2},
            'active_user': {'rating_count': 15},
            'power_user': {'rating_count': 45}
        }
        
        for user_type, profile_data in user_types.items():
            # Find a user matching this profile
            matching_user = None
            for user_id, profile in user_profiles.items():
                if profile['rating_count'] == profile_data['rating_count']:
                    matching_user = user_id
                    break
            
            if matching_user:
                user_profile = user_profiles[matching_user]
                recommendations = hybrid_recommender.recommend(matching_user, user_profile, 3)
                
                print(f"  {user_type}: User {matching_user} ({user_profile['rating_count']} interactions)")
                for i, (item_id, score) in enumerate(recommendations, 1):
                    print(f"    {i}. Item {item_id} (score: {score:.3f})")

def example_4_user_profiles():
    """Example 4: Demonstrate different user profiles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: USER PROFILES")
    print("=" * 60)
    
    # Load data
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    user_item_matrix, _, _ = data_loader.load_interaction_data("interactions.csv")
    product_features = data_loader.load_product_features("products.csv")
    
    # Create user profiles
    user_profiles = data_loader.create_user_profiles(user_item_matrix, product_features)
    
    # Train models
    collaborative_model = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    collaborative_model.fit(user_item_matrix)
    
    content_model = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    content_model.fit(product_features)
    
    hybrid_recommender = HybridRecommender(collaborative_model, content_model, HYBRID_CONFIG)
    
    # Analyze different user types
    user_types = {
        'new_user': {'min_interactions': 0, 'max_interactions': 5},
        'active_user': {'min_interactions': 6, 'max_interactions': 20},
        'power_user': {'min_interactions': 21, 'max_interactions': float('inf')}
    }
    
    for user_type, criteria in user_types.items():
        print(f"\n{user_type.upper()} ANALYSIS:")
        print("-" * 40)
        
        # Find users matching this profile
        matching_users = []
        for user_id, profile in user_profiles.items():
            if (criteria['min_interactions'] <= profile['rating_count'] <= 
                criteria['max_interactions']):
                matching_users.append(user_id)
        
        print(f"Found {len(matching_users)} users matching {user_type} profile")
        
        if matching_users:
            # Analyze first matching user
            demo_user = matching_users[0]
            user_profile = user_profiles[demo_user]
            
            print(f"\nDemo user {demo_user}:")
            print(f"  Interactions: {user_profile['rating_count']}")
            print(f"  Average rating: {user_profile['avg_rating']:.2f}")
            print(f"  Rated items: {len(user_profile['rated_items'])}")
            
            if 'preferences' in user_profile:
                print("  Preferences:")
                for feature, value in user_profile['preferences'].items():
                    print(f"    {feature}: {value}")
            
            # Generate recommendations
            recommendations = hybrid_recommender.recommend(demo_user, user_profile, 5)
            
            print("\n  Top 5 recommendations:")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"    {i}. Item {item_id} (score: {score:.3f})")

def example_5_evaluation_metrics():
    """Example 5: Comprehensive evaluation with all metrics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: EVALUATION METRICS")
    print("=" * 60)
    
    # Load data
    data_loader = RecommendationDataLoader(DATA_CONFIG)
    user_item_matrix, _, _ = data_loader.load_interaction_data("interactions.csv")
    product_features = data_loader.load_product_features("products.csv")
    
    # Split data
    train_matrix, test_matrix, _, _ = data_loader.split_data(user_item_matrix, test_size=0.2)
    
    # Train hybrid model
    print("Training hybrid model...")
    collaborative_model = UserBasedCF(COLLABORATIVE_CONFIG['user_based'])
    collaborative_model.fit(train_matrix)
    
    content_model = TFIDFContentBased(CONTENT_BASED_CONFIG['tfidf'])
    content_model.fit(product_features)
    
    hybrid_recommender = HybridRecommender(collaborative_model, content_model, HYBRID_CONFIG)
    
    # Create user profiles
    user_profiles = data_loader.create_user_profiles(train_matrix, product_features)
    
    # Evaluate with different k values
    evaluator = RecommendationEvaluator(EVALUATION_CONFIG)
    test_users = list(test_matrix.index)[:30]
    
    print("\nGenerating recommendations for evaluation...")
    recommendations = {}
    for user_id in test_users:
        try:
            user_profile = user_profiles.get(user_id, {})
            recs = hybrid_recommender.recommend(user_id, user_profile, 20)
            recommendations[user_id] = recs
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
            continue
    
    # Create ground truth
    ground_truth = {}
    for user_id in test_users:
        user_ratings = test_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        ground_truth[user_id] = rated_items
    
    # Evaluate
    print("Evaluating with comprehensive metrics...")
    results = evaluator.evaluate(recommendations, ground_truth, test_users)
    
    # Print detailed results
    print("\nDetailed Evaluation Results:")
    print("=" * 50)
    
    for k_key, k_results in results.items():
        if k_key != 'overall':
            print(f"\n{k_key.upper()}:")
            for metric, value in k_results.items():
                print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nOVERALL METRICS:")
    print("-" * 20)
    for metric, value in results['overall'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_metrics_comparison(results)
    
    # Generate report
    print("\nGenerating evaluation report...")
    evaluator.generate_report(results)

if __name__ == "__main__":
    print("SmartRetail Hybrid Recommendation System - Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_model_comparison()
    example_3_hybrid_strategies()
    example_4_user_profiles()
    example_5_evaluation_metrics()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60) 