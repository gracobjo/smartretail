#!/usr/bin/env python3
"""
Script de demostración del Sistema de Detección de Fraude Financiero - SmartRetail

Este script ejecuta el pipeline completo del sistema de detección de fraude:
1. Generación de datos sintéticos
2. Análisis exploratorio
3. Preprocesamiento
4. Entrenamiento de modelos
5. Evaluación
6. Explicabilidad SHAP
7. Selección de variables
8. Tuning de hiperparámetros

Uso:
    python run_demo.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import MODEL_CONFIG
from data_processing.data_processor import DataProcessor
from models.fraud_detector import FraudDetector
from evaluation.evaluator import FraudEvaluator
from explainability.shap_explainer import SHAPExplainer

def create_synthetic_data(n_samples=10000, fraud_ratio=0.05):
    """
    Crear datos sintéticos de fraude financiero para demostración.
    
    Args:
        n_samples (int): Número de muestras
        fraud_ratio (float): Proporción de fraude
        
    Returns:
        pd.DataFrame: Dataset sintético
    """
    print("Creando datos sintéticos de fraude financiero...")
    
    np.random.seed(42)
    
    # Generar features básicas
    n_features = 20
    X_synthetic = np.random.randn(n_samples, n_features)
    
    # Crear target con desbalance
    y_synthetic = np.random.binomial(1, fraud_ratio, n_samples)
    
    # Crear DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X_synthetic, columns=feature_names)
    
    # Agregar features específicas de fraude financiero
    data['amount'] = np.random.exponential(100, n_samples)
    data['time'] = np.random.randint(0, 24, n_samples)
    data['location'] = np.random.randint(0, 10, n_samples)
    data['merchant_type'] = np.random.randint(0, 5, n_samples)
    data['fraud'] = y_synthetic
    
    # Introducir patrones de fraude
    fraud_indices = np.where(y_synthetic == 1)[0]
    for idx in fraud_indices:
        # Fraude: montos más altos
        data.loc[idx, 'amount'] = np.random.exponential(500)
        # Fraude: horarios sospechosos (noche)
        if np.random.random() > 0.5:
            data.loc[idx, 'time'] = np.random.randint(0, 6)
        # Fraude: ubicaciones inusuales
        if np.random.random() > 0.7:
            data.loc[idx, 'location'] = np.random.randint(8, 15)
    
    print(f"Dataset sintético creado: {data.shape}")
    print(f"Distribución de fraude: {data['fraud'].value_counts()}")
    print(f"Porcentaje de fraude: {data['fraud'].mean()*100:.2f}%")
    
    return data

def setup_directories():
    """Crear directorios necesarios."""
    directories = [
        'fraud_detection/data',
        'fraud_detection/results',
        'fraud_detection/models',
        'fraud_detection/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directorios creados correctamente!")

def run_complete_pipeline():
    """Ejecutar el pipeline completo del sistema de detección de fraude."""
    
    print("=" * 80)
    print("SISTEMA DE DETECCIÓN DE FRAUDE FINANCIERO - DEMOSTRACIÓN")
    print("=" * 80)
    
    # 1. Configurar directorios
    setup_directories()
    
    # 2. Crear datos sintéticos
    data = create_synthetic_data(n_samples=10000, fraud_ratio=0.05)
    
    # Guardar datos
    data.to_csv('fraud_detection/data/synthetic_fraud_data.csv', index=False)
    print("Datos guardados en fraud_detection/data/synthetic_fraud_data.csv")
    
    # 3. Inicializar componentes del sistema
    print("\nInicializando componentes del sistema...")
    data_processor = DataProcessor(MODEL_CONFIG)
    fraud_detector = FraudDetector(MODEL_CONFIG)
    evaluator = FraudEvaluator(MODEL_CONFIG)
    shap_explainer = SHAPExplainer(MODEL_CONFIG)
    
    # 4. Análisis exploratorio
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)
    
    data_processor.explore_data(data, target_column='fraud')
    feature_report = data_processor.create_feature_report(data, target_column='fraud')
    
    # 5. Preprocesamiento
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # Limpiar datos
    data_cleaned = data_processor.clean_data(data)
    
    # Crear features adicionales
    data_engineered = data_processor.engineer_features(data_cleaned)
    
    print(f"Datos originales: {data.shape}")
    print(f"Datos procesados: {data_engineered.shape}")
    print(f"Features adicionales creadas: {data_engineered.shape[1] - data.shape[1]}")
    
    # 6. Preparación para entrenamiento
    print("\n" + "=" * 60)
    print("PREPARACIÓN PARA ENTRENAMIENTO")
    print("=" * 60)
    
    # Separar features y target
    X = data_engineered.drop(columns=['fraud'])
    y = data_engineered['fraud']
    
    # Dividir datos
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"Conjunto de entrenamiento: {X_train_scaled.shape}")
    print(f"Conjunto de prueba: {X_test_scaled.shape}")
    print(f"Distribución de fraude en entrenamiento: {y_train.value_counts()}")
    print(f"Distribución de fraude en prueba: {y_test.value_counts()}")
    
    # 7. Entrenamiento de modelos
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELOS")
    print("=" * 60)
    
    models = fraud_detector.train_all_models(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        balance_data=True
    )
    
    print(f"Modelos entrenados: {list(models.keys())}")
    
    # Seleccionar mejor modelo
    best_model = fraud_detector.select_best_model(X_test_scaled, y_test, metric='f1')
    print(f"Mejor modelo: {best_model}")
    
    # 8. Evaluación de modelos
    print("\n" + "=" * 60)
    print("EVALUACIÓN DE MODELOS")
    print("=" * 60)
    
    # Generar predicciones
    predictions = fraud_detector.predict(X_test_scaled)
    
    # Evaluar todos los modelos
    evaluation_results = evaluator.evaluate_all_models(y_test, predictions)
    
    print("\nResultados de evaluación:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
    
    # Crear visualizaciones
    evaluator.plot_roc_curves(save_path='fraud_detection/results/roc_curves.png')
    evaluator.plot_precision_recall_curves(save_path='fraud_detection/results/precision_recall_curves.png')
    evaluator.plot_confusion_matrices(save_path='fraud_detection/results/confusion_matrices.png')
    evaluator.plot_metrics_comparison(save_path='fraud_detection/results/metrics_comparison.png')
    
    # Crear reporte de evaluación
    evaluator.create_evaluation_report(save_path='fraud_detection/results/evaluation_report.txt')
    
    # 9. Análisis de importancia de features
    print("\n" + "=" * 60)
    print("ANÁLISIS DE IMPORTANCIA DE FEATURES")
    print("=" * 60)
    
    feature_importance = fraud_detector.get_feature_importance()
    
    # Crear DataFrame con importancia de features
    importance_df = pd.DataFrame()
    
    for model_name, importance_data in feature_importance.items():
        model_importance = pd.DataFrame({
            'feature': importance_data['feature_names'],
            f'{model_name}_importance': importance_data['importance']
        })
        
        if importance_df.empty:
            importance_df = model_importance
        else:
            importance_df = importance_df.merge(model_importance, on='feature', how='outer')
    
    # Calcular importancia promedio
    importance_columns = [col for col in importance_df.columns if col.endswith('_importance')]
    importance_df['avg_importance'] = importance_df[importance_columns].mean(axis=1)
    
    # Ordenar por importancia promedio
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    print("Top 15 features por importancia promedio:")
    print(importance_df.head(15)[['feature', 'avg_importance']])
    
    # Guardar resultados
    importance_df.to_csv('fraud_detection/results/feature_importance.csv', index=False)
    
    # Visualizar importancia de features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['avg_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia Promedio')
    plt.title('Top 15 Features por Importancia Promedio')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('fraud_detection/results/feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. Explicabilidad SHAP
    print("\n" + "=" * 60)
    print("EXPLICABILIDAD SHAP")
    print("=" * 60)
    
    X_sample = X_test_scaled.head(100)  # Usar muestra para explicaciones
    
    for model_name in fraud_detector.models.keys():
        print(f"\nCreando explainer SHAP para {model_name}...")
        
        # Crear explainer
        shap_explainer.create_explainer(
            fraud_detector.models[model_name],
            X_train_scaled,
            model_name
        )
        
        # Calcular SHAP values
        shap_explainer.calculate_shap_values(X_sample, model_name)
        
        # Generar visualizaciones SHAP
        shap_explainer.plot_summary(
            X_sample, model_name,
            save_path=f'fraud_detection/results/shap_summary_{model_name}.png'
        )
        
        # Obtener importancia de features basada en SHAP
        importance_df = shap_explainer.get_feature_importance(model_name, top_n=10)
        print(f"Top 10 features por SHAP importance ({model_name}):")
        print(importance_df)
    
    # 11. Tuning de hiperparámetros (opcional)
    print("\n" + "=" * 60)
    print("TUNING DE HIPERPARÁMETROS")
    print("=" * 60)
    
    try:
        import optuna
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            
            import xgboost as xgb
            model = xgb.XGBClassifier(**params)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)  # Reducido para demostración
        
        print(f"Mejores parámetros: {study.best_params}")
        print(f"Mejor F1-score: {study.best_value:.4f}")
        
    except ImportError:
        print("Optuna no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'optuna'])
        print("Optuna instalado. Ejecuta el script nuevamente para tuning.")
    
    # 12. Guardar modelos y resultados
    print("\n" + "=" * 60)
    print("GUARDANDO MODELOS Y RESULTADOS")
    print("=" * 60)
    
    fraud_detector.save_models('fraud_detection/models/trained_models.pkl')
    evaluator.save_results('fraud_detection/results/evaluation_results.pkl')
    shap_explainer.save_explanations('fraud_detection/results/shap_explanations.pkl')
    
    # 13. Reporte final
    print("\n" + "=" * 80)
    print("REPORTE FINAL - SISTEMA DE DETECCIÓN DE FRAUDE FINANCIERO")
    print("=" * 80)
    
    print(f"\nDataset:")
    print(f"  - Muestras totales: {len(data)}")
    print(f"  - Features originales: {data.shape[1] - 1}")
    print(f"  - Features finales: {X_train_scaled.shape[1]}")
    print(f"  - Porcentaje de fraude: {data['fraud'].mean()*100:.2f}%")
    
    print(f"\nModelos entrenados: {list(fraud_detector.models.keys())}")
    print(f"Mejor modelo: {best_model}")
    
    print(f"\nResultados del mejor modelo ({best_model}):")
    best_results = evaluation_results[best_model]
    print(f"  - Accuracy: {best_results['accuracy']:.4f}")
    print(f"  - Precision: {best_results['precision']:.4f}")
    print(f"  - Recall: {best_results['recall']:.4f}")
    print(f"  - F1-Score: {best_results['f1_score']:.4f}")
    print(f"  - ROC AUC: {best_results['roc_auc']:.4f}")
    
    print(f"\nTop 5 features más importantes:")
    top_features = importance_df.head(5)
    for _, row in top_features.iterrows():
        print(f"  - {row['feature']}: {row['avg_importance']:.4f}")
    
    print(f"\nArchivos generados:")
    print(f"  - Modelos: fraud_detection/models/")
    print(f"  - Resultados: fraud_detection/results/")
    print(f"  - Visualizaciones: fraud_detection/results/")
    print(f"  - Datos: fraud_detection/data/")
    
    print(f"\nCaracterísticas del sistema:")
    print(f"  - Modelos supervisados: XGBoost, LightGBM, Random Forest, Logistic Regression")
    print(f"  - Explicabilidad: SHAP values")
    print(f"  - Balanceo de datos: SMOTE")
    print(f"  - Evaluación completa: Precision, Recall, F1, ROC AUC")
    print(f"  - Selección de variables: Basada en importancia")
    print(f"  - Tuning de hiperparámetros: Optuna")
    
    print(f"\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    
    return {
        'models': models,
        'best_model': best_model,
        'evaluation_results': evaluation_results,
        'feature_importance': importance_df
    }

if __name__ == "__main__":
    # Ejecutar pipeline completo
    results = run_complete_pipeline()
    
    print("\n¡Pipeline completado exitosamente!")
    print("Revisa los archivos en fraud_detection/results/ para ver los resultados detallados.") 