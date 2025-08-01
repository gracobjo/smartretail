"""
Data processing module for fraud detection.
Handles data loading, cleaning, exploration, and feature engineering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data processor for fraud detection datasets."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """
        Load data from various file formats.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        print(f"Loading data from {file_path}...")
        
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Data loaded successfully: {data.shape}")
        return data
    
    def clean_data(self, data):
        """
        Clean the dataset by handling missing values, outliers, and data types.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("Cleaning data...")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Handle missing values
        missing_counts = cleaned_data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found: {missing_counts[missing_counts > 0]}")
            
            # For numerical columns, fill with median (exclude target column)
            numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            if 'fraud' in numerical_cols:
                numerical_cols = numerical_cols.drop('fraud')
            
            for col in numerical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    median_val = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    mode_val = cleaned_data[col].mode()[0]
                    cleaned_data[col].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Handle outliers for numerical columns (exclude target column)
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        if 'fraud' in numerical_cols:
            numerical_cols = numerical_cols.drop('fraud')
        
        for col in numerical_cols:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound))
            if outliers.sum() > 0:
                print(f"Found {outliers.sum()} outliers in {col}")
                # Cap outliers instead of removing them
                cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Convert data types
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    if cleaned_data[col].isnull().sum() == 0:
                        print(f"Converted {col} to numeric")
                except:
                    pass
        
        print("Data cleaning completed!")
        return cleaned_data
    
    def engineer_features(self, data):
        """
        Create new features for fraud detection.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        print("Engineering features...")
        
        engineered_data = data.copy()
        
        # Preserve the target column
        target_column = None
        if 'fraud' in engineered_data.columns:
            target_column = engineered_data['fraud'].copy()
        
        # Get numerical columns for feature engineering (exclude target column)
        numerical_cols = engineered_data.select_dtypes(include=[np.number]).columns
        if 'fraud' in numerical_cols:
            numerical_cols = numerical_cols.drop('fraud')
        
        # Create statistical features
        for col in numerical_cols:
            # Rolling statistics (if data has temporal structure)
            if len(engineered_data) > 10:
                engineered_data[f'{col}_rolling_mean'] = engineered_data[col].rolling(window=5, min_periods=1).mean()
                engineered_data[f'{col}_rolling_std'] = engineered_data[col].rolling(window=5, min_periods=1).std()
            
            # Polynomial features for important numerical columns
            if col in ['amount', 'time', 'location']:
                engineered_data[f'{col}_squared'] = engineered_data[col] ** 2
                engineered_data[f'{col}_cubed'] = engineered_data[col] ** 3
        
        # Create interaction features
        if 'amount' in engineered_data.columns and 'time' in engineered_data.columns:
            engineered_data['amount_time_interaction'] = engineered_data['amount'] * engineered_data['time']
        
        if 'amount' in engineered_data.columns and 'location' in engineered_data.columns:
            engineered_data['amount_location_interaction'] = engineered_data['amount'] * engineered_data['location']
        
        # Create ratio features
        if 'amount' in engineered_data.columns:
            # Amount ratios with other numerical features
            for col in numerical_cols:
                if col != 'amount':
                    engineered_data[f'amount_{col}_ratio'] = engineered_data['amount'] / (engineered_data[col] + 1e-8)
        
        # Create categorical features from numerical ones
        if 'amount' in engineered_data.columns:
            engineered_data['amount_category'] = pd.cut(
                engineered_data['amount'], 
                bins=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        if 'time' in engineered_data.columns:
            engineered_data['time_of_day'] = pd.cut(
                engineered_data['time'], 
                bins=[0, 6, 12, 18, 24], 
                labels=['night', 'morning', 'afternoon', 'evening']
            )
        
        # Encode categorical features
        categorical_cols = engineered_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                engineered_data[col] = self.label_encoders[col].fit_transform(engineered_data[col])
            else:
                # Handle new categories
                unique_values = engineered_data[col].unique()
                known_values = self.label_encoders[col].classes_
                unknown_values = set(unique_values) - set(known_values)
                
                if unknown_values:
                    # Add unknown category
                    engineered_data[col] = engineered_data[col].astype(str)
                    engineered_data[col] = engineered_data[col].map(lambda x: 'unknown' if x not in known_values else x)
                    self.label_encoders[col].fit(list(known_values) + ['unknown'])
                
                engineered_data[col] = self.label_encoders[col].transform(engineered_data[col])
        
        # Clean any NaN values created during feature engineering
        engineered_data = engineered_data.fillna(0)
        
        # Restore target column if it existed
        if target_column is not None:
            engineered_data['fraud'] = target_column
        
        print(f"Feature engineering completed! Original features: {len(data.columns)}, New features: {len(engineered_data.columns)}")
        return engineered_data
    
    def explore_data(self, data, target_column='fraud'):
        """
        Perform exploratory data analysis.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of the target column
        """
        print("Performing exploratory data analysis...")
        
        # Basic information
        print(f"\nDataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data types:\n{data.dtypes}")
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\nMissing values:\n{missing_data[missing_data > 0]}")
        
        # Target distribution
        if target_column in data.columns:
            target_dist = data[target_column].value_counts()
            print(f"\nTarget distribution:\n{target_dist}")
            print(f"Fraud percentage: {target_dist[1]/len(data)*100:.2f}%")
        
        # Numerical features analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\nNumerical features statistics:")
            print(data[numerical_cols].describe())
        
        # Correlation analysis
        if len(numerical_cols) > 1:
            correlation_matrix = data[numerical_cols].corr()
            print(f"\nCorrelation matrix shape: {correlation_matrix.shape}")
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('fraud_detection/results/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Distribution plots for numerical features
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(numerical_cols):
                row = i // n_cols
                col_idx = i % n_cols
                
                axes[row, col_idx].hist(data[col], bins=30, alpha=0.7)
                axes[row, col_idx].set_title(f'Distribution of {col}')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('fraud_detection/results/feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Target vs features analysis
        if target_column in data.columns and len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(numerical_cols):
                if col != target_column:
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    # Box plot by target
                    data.boxplot(column=col, by=target_column, ax=axes[row, col_idx])
                    axes[row, col_idx].set_title(f'{col} by {target_column}')
                    axes[row, col_idx].set_xlabel(target_column)
                    axes[row, col_idx].set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len([col for col in numerical_cols if col != target_column]), n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('fraud_detection/results/target_vs_features.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def perform_pca_analysis(self, data, n_components=0.95):
        """
        Perform Principal Component Analysis.
        
        Args:
            data (pd.DataFrame): Input data
            n_components (float): Fraction of variance to explain
            
        Returns:
            tuple: (pca, transformed_data, explained_variance)
        """
        print("Performing PCA analysis...")
        
        # Select numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("Not enough numerical features for PCA")
            return None, None, None
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numerical_cols])
        
        # Perform PCA
        if n_components < 1:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=int(n_components))
        
        transformed_data = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"Number of components: {pca.n_components_}")
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True, alpha=0.3)
        plt.savefig('fraud_detection/results/pca_explained_variance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca, transformed_data, explained_variance
    
    def create_feature_report(self, data, target_column='fraud'):
        """
        Create a comprehensive feature analysis report.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of the target column
        """
        print("Creating feature analysis report...")
        
        report = []
        report.append("=" * 60)
        report.append("FEATURE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic dataset information
        report.append("DATASET INFORMATION:")
        report.append("-" * 25)
        report.append(f"Shape: {data.shape}")
        report.append(f"Features: {len(data.columns)}")
        report.append(f"Samples: {len(data)}")
        report.append("")
        
        # Data types
        report.append("DATA TYPES:")
        report.append("-" * 15)
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            report.append(f"{dtype}: {count}")
        report.append("")
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            report.append("MISSING VALUES:")
            report.append("-" * 18)
            for col, missing_count in missing_data[missing_data > 0].items():
                percentage = (missing_count / len(data)) * 100
                report.append(f"{col}: {missing_count} ({percentage:.2f}%)")
            report.append("")
        
        # Target analysis
        if target_column in data.columns:
            target_dist = data[target_column].value_counts()
            report.append("TARGET ANALYSIS:")
            report.append("-" * 18)
            report.append(f"Class distribution:")
            for class_val, count in target_dist.items():
                percentage = (count / len(data)) * 100
                report.append(f"  Class {class_val}: {count} ({percentage:.2f}%)")
            report.append("")
        
        # Numerical features analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            report.append("NUMERICAL FEATURES:")
            report.append("-" * 22)
            for col in numerical_cols:
                stats = data[col].describe()
                report.append(f"{col}:")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Std: {stats['std']:.4f}")
                report.append(f"  Min: {stats['min']:.4f}")
                report.append(f"  Max: {stats['max']:.4f}")
                report.append(f"  Q1: {stats['25%']:.4f}")
                report.append(f"  Q3: {stats['75%']:.4f}")
                report.append("")
        
        # Correlation with target
        if target_column in data.columns and len(numerical_cols) > 0:
            correlations = data[numerical_cols].corrwith(data[target_column])
            report.append("CORRELATION WITH TARGET:")
            report.append("-" * 25)
            for col, corr in correlations.items():
                report.append(f"{col}: {corr:.4f}")
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('fraud_detection/results/feature_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Feature analysis report saved to fraud_detection/results/feature_analysis_report.txt")
        return report_text
    
    def save_processed_data(self, data, filepath):
        """
        Save processed data to disk.
        
        Args:
            data (pd.DataFrame): Processed data
            filepath (str): Path to save the data
        """
        print(f"Saving processed data to {filepath}...")
        
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=False)
        elif filepath.endswith('.parquet'):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print("Processed data saved successfully!")
    
    def load_processed_data(self, filepath):
        """
        Load processed data from disk.
        
        Args:
            filepath (str): Path to load the data from
            
        Returns:
            pd.DataFrame: Loaded data
        """
        print(f"Loading processed data from {filepath}...")
        
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print("Processed data loaded successfully!")
        return data 