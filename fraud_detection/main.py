"""
Sistema de Detección de Fraude Financiero - SmartRetail
Script principal que integra análisis exploratorio, entrenamiento, evaluación y explicabilidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import MODEL_CONFIG
from data_processing.data_processor import DataProcessor
from models.fraud_detector import FraudDetector
from evaluation.evaluator import FraudEvaluator
from explainability.shap_explainer import SHAPExplainer

class FraudDetectionSystem:
    """Sistema completo de detección de fraude financiero."""
    
    def __init__(self, config=None):
        """Inicializar el sistema de detección de fraude."""
        self.config = config or MODEL_CONFIG
        self.data_processor = DataProcessor(self.config)
        self.fraud_detector = FraudDetector(self.config)
        self.evaluator = FraudEvaluator(self.config)
        self.shap_explainer = SHAPExplainer(self.config)
        
        # Datos y resultados
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self, data_path):
        """
        Cargar y explorar los datos de fraude financiero.
        
        Args:
            data_path (str): Ruta al archivo de datos
        """
        print("=" * 60)
        print("CARGANDO Y EXPLORANDO DATOS")
        print("=" * 60)
        
        # Cargar datos
        self.data = self.data_processor.load_data(data_path)
        print(f"Datos cargados: {self.data.shape}")
        
        # Análisis exploratorio básico
        print("\nInformación del dataset:")
        print(f"Dimensiones: {self.data.shape}")
        print(f"Columnas: {list(self.data.columns)}")
        print(f"Tipos de datos:\n{self.data.dtypes}")
        
        # Estadísticas descriptivas
        print("\nEstadísticas descriptivas:")
        print(self.data.describe())
        
        # Verificar valores faltantes
        print("\nValores faltantes:")
        missing_data = self.data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # Distribución de la variable objetivo
        if 'fraud' in self.data.columns:
            print("\nDistribución de fraude:")
            fraud_dist = self.data['fraud'].value_counts()
            print(fraud_dist)
            print(f"Porcentaje de fraude: {fraud_dist[1]/len(self.data)*100:.2f}%")
        
        return self.data
    
    def preprocess_data(self, target_column='fraud'):
        """
        Preprocesar los datos para el entrenamiento.
        
        Args:
            target_column (str): Nombre de la columna objetivo
        """
        print("\n" + "=" * 60)
        print("PREPROCESAMIENTO DE DATOS")
        print("=" * 60)
        
        # Separar features y target
        if target_column in self.data.columns:
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
        else:
            # Asumir que la última columna es el target
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
        
        print(f"Features: {X.shape}")
        print(f"Target: {y.shape}")
        
        # Limpiar datos
        X_cleaned = self.data_processor.clean_data(X)
        
        # Crear features adicionales
        X_engineered = self.data_processor.engineer_features(X_cleaned)
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=self.config['random_state'], stratify=y
        )
        
        # Escalar features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convertir de vuelta a DataFrame para mantener nombres de columnas
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        
        print(f"Conjunto de entrenamiento: {self.X_train_scaled.shape}")
        print(f"Conjunto de prueba: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self, balance_data=True):
        """
        Entrenar todos los modelos de detección de fraude.
        
        Args:
            balance_data (bool): Si balancear los datos
        """
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DE MODELOS")
        print("=" * 60)
        
        # Entrenar todos los modelos
        models = self.fraud_detector.train_all_models(
            self.X_train_scaled, self.y_train, 
            self.X_test_scaled, self.y_test,
            balance_data=balance_data
        )
        
        # Seleccionar mejor modelo
        best_model = self.fraud_detector.select_best_model(
            self.X_test_scaled, self.y_test, metric='f1'
        )
        
        print(f"\nMejor modelo seleccionado: {best_model}")
        
        return models, best_model
    
    def evaluate_models(self):
        """
        Evaluar todos los modelos entrenados.
        """
        print("\n" + "=" * 60)
        print("EVALUACIÓN DE MODELOS")
        print("=" * 60)
        
        # Generar predicciones
        predictions = self.fraud_detector.predict(self.X_test_scaled)
        
        # Evaluar todos los modelos
        evaluation_results = self.evaluator.evaluate_all_models(self.y_test, predictions)
        
        # Crear visualizaciones
        self.evaluator.plot_roc_curves(save_path='fraud_detection/results/roc_curves.png')
        self.evaluator.plot_precision_recall_curves(save_path='fraud_detection/results/precision_recall_curves.png')
        self.evaluator.plot_confusion_matrices(save_path='fraud_detection/results/confusion_matrices.png')
        self.evaluator.plot_metrics_comparison(save_path='fraud_detection/results/metrics_comparison.png')
        
        # Crear reporte de evaluación
        self.evaluator.create_evaluation_report(save_path='fraud_detection/results/evaluation_report.txt')
        
        return evaluation_results
    
    def explain_models(self, X_sample=None):
        """
        Generar explicaciones SHAP para los modelos.
        
        Args:
            X_sample (pd.DataFrame): Muestra de datos para explicar
        """
        print("\n" + "=" * 60)
        print("EXPLICABILIDAD DE MODELOS (SHAP)")
        print("=" * 60)
        
        if X_sample is None:
            X_sample = self.X_test_scaled.head(100)  # Usar primeros 100 ejemplos
        
        # Crear explainers para cada modelo
        for model_name in self.fraud_detector.models.keys():
            print(f"\nCreando explainer para {model_name}...")
            
            # Crear explainer
            self.shap_explainer.create_explainer(
                self.fraud_detector.models[model_name],
                self.X_train_scaled,
                model_name
            )
            
            # Calcular SHAP values
            self.shap_explainer.calculate_shap_values(X_sample, model_name)
            
            # Generar visualizaciones
            self.shap_explainer.plot_summary(
                X_sample, model_name, 
                save_path=f'fraud_detection/results/shap_summary_{model_name}.png'
            )
            
            # Plot de dependencia para las top 5 features
            importance_df = self.shap_explainer.get_feature_importance(model_name, top_n=5)
            for feature in importance_df['feature'][:3]:  # Top 3 features
                try:
                    self.shap_explainer.plot_dependence(
                        X_sample, model_name, feature,
                        save_path=f'fraud_detection/results/shap_dependence_{model_name}_{feature}.png'
                    )
                except Exception as e:
                    print(f"Error al crear dependence plot para {feature}: {e}")
        
        # Comparar importancia de features entre modelos
        model_names = list(self.fraud_detector.models.keys())
        self.shap_explainer.plot_feature_importance_comparison(
            model_names, save_path='fraud_detection/results/feature_importance_comparison.png'
        )
        
        # Crear reporte de explicaciones
        for model_name in model_names:
            self.shap_explainer.create_explanation_report(
                X_sample, model_name,
                save_path=f'fraud_detection/results/shap_explanation_{model_name}.txt'
            )
    
    def perform_feature_selection(self):
        """
        Realizar selección de variables basada en importancia.
        """
        print("\n" + "=" * 60)
        print("SELECCIÓN DE VARIABLES")
        print("=" * 60)
        
        # Obtener importancia de features para cada modelo
        feature_importance = self.fraud_detector.get_feature_importance()
        
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
        
        print("Top 20 features por importancia promedio:")
        print(importance_df.head(20)[['feature', 'avg_importance']])
        
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
        
        return importance_df
    
    def hyperparameter_tuning(self, model_name='xgboost'):
        """
        Realizar tuning de hiperparámetros usando Optuna.
        
        Args:
            model_name (str): Nombre del modelo a optimizar
        """
        print("\n" + "=" * 60)
        print(f"TUNING DE HIPERPARÁMETROS - {model_name.upper()}")
        print("=" * 60)
        
        try:
            import optuna
            
            def objective(trial):
                # Definir espacio de búsqueda
                if model_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': self.config['random_state']
                    }
                elif model_name == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': self.config['random_state']
                    }
                else:
                    return 0.0
                
                # Crear modelo temporal
                if model_name == 'xgboost':
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**params)
                elif model_name == 'lightgbm':
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**params)
                
                # Cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='f1')
                
                return scores.mean()
            
            # Crear estudio de optimización
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            print(f"Mejores parámetros: {study.best_params}")
            print(f"Mejor F1-score: {study.best_value:.4f}")
            
            # Actualizar configuración con mejores parámetros
            self.config[model_name].update(study.best_params)
            
            # Reentrenar modelo con mejores parámetros
            if model_name == 'xgboost':
                self.fraud_detector.train_xgboost(self.X_train_scaled, self.y_train)
            elif model_name == 'lightgbm':
                self.fraud_detector.train_lightgbm(self.X_train_scaled, self.y_train)
            
            return study.best_params, study.best_value
            
        except ImportError:
            print("Optuna no está instalado. Instalando...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'optuna'])
            return self.hyperparameter_tuning(model_name)
    
    def run_complete_pipeline(self, data_path, target_column='fraud'):
        """
        Ejecutar el pipeline completo de detección de fraude.
        
        Args:
            data_path (str): Ruta al archivo de datos
            target_column (str): Nombre de la columna objetivo
        """
        print("=" * 80)
        print("SISTEMA DE DETECCIÓN DE FRAUDE FINANCIERO - PIPELINE COMPLETO")
        print("=" * 80)
        
        # 1. Cargar y explorar datos
        self.load_and_explore_data(data_path)
        
        # 2. Preprocesar datos
        self.preprocess_data(target_column)
        
        # 3. Entrenar modelos
        models, best_model = self.train_models(balance_data=True)
        
        # 4. Evaluar modelos
        evaluation_results = self.evaluate_models()
        
        # 5. Selección de variables
        feature_importance = self.perform_feature_selection()
        
        # 6. Explicabilidad SHAP
        self.explain_models()
        
        # 7. Tuning de hiperparámetros (opcional)
        print("\n¿Desea realizar tuning de hiperparámetros? (s/n): ", end="")
        # En un entorno real, esto sería una entrada del usuario
        # Por ahora, asumimos que sí
        perform_tuning = True
        
        if perform_tuning:
            self.hyperparameter_tuning('xgboost')
            self.hyperparameter_tuning('lightgbm')
        
        # 8. Guardar modelos y resultados
        self.fraud_detector.save_models('fraud_detection/models/trained_models.pkl')
        self.evaluator.save_results('fraud_detection/results/evaluation_results.pkl')
        self.shap_explainer.save_explanations('fraud_detection/results/shap_explanations.pkl')
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print("Resultados guardados en:")
        print("- Modelos: fraud_detection/models/")
        print("- Resultados: fraud_detection/results/")
        print("- Visualizaciones: fraud_detection/results/")
        
        return {
            'models': models,
            'best_model': best_model,
            'evaluation_results': evaluation_results,
            'feature_importance': feature_importance
        }

def main():
    """Función principal para ejecutar el sistema de detección de fraude."""
    
    # Crear directorios necesarios
    import os
    os.makedirs('fraud_detection/results', exist_ok=True)
    os.makedirs('fraud_detection/models', exist_ok=True)
    
    # Inicializar sistema
    fraud_system = FraudDetectionSystem()
    
    # Ejecutar pipeline completo
    # Nota: Necesitas proporcionar la ruta a tu dataset de fraude
    # data_path = 'path/to/your/fraud_dataset.csv'
    
    # Para demostración, crearemos un dataset sintético
    print("Creando dataset sintético para demostración...")
    
    # Crear datos sintéticos de fraude financiero
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Generar features
    X_synthetic = np.random.randn(n_samples, n_features)
    
    # Crear target con desbalance (fraude es raro)
    fraud_prob = 0.05  # 5% de fraude
    y_synthetic = np.random.binomial(1, fraud_prob, n_samples)
    
    # Crear nombres de features
    feature_names = [f'feature_{i}' for i in range(n_features)]
    feature_names.extend(['amount', 'time', 'location', 'merchant_type', 'fraud'])
    
    # Crear DataFrame
    data = pd.DataFrame(X_synthetic, columns=feature_names[:-5])
    
    # Agregar features adicionales
    data['amount'] = np.random.exponential(100, n_samples)
    data['time'] = np.random.randint(0, 24, n_samples)
    data['location'] = np.random.randint(0, 10, n_samples)
    data['merchant_type'] = np.random.randint(0, 5, n_samples)
    data['fraud'] = y_synthetic
    
    # Guardar dataset sintético
    data.to_csv('fraud_detection/data/synthetic_fraud_data.csv', index=False)
    
    print("Dataset sintético creado y guardado en fraud_detection/data/synthetic_fraud_data.csv")
    
    # Ejecutar pipeline
    results = fraud_system.run_complete_pipeline(
        'fraud_detection/data/synthetic_fraud_data.csv',
        target_column='fraud'
    )
    
    print("\nPipeline completado!")
    print(f"Mejor modelo: {results['best_model']}")
    print("Revisa los archivos en fraud_detection/results/ para ver los resultados detallados.")

if __name__ == "__main__":
    main() 