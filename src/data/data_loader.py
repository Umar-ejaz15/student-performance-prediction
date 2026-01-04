"""
Data loading and preprocessing module
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Handle data loading and initial preprocessing"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_encoded = None
        
    def find_data_file(self) -> Optional[str]:
        """
        Try to find the CSV file in common locations
        
        Returns:
            Path to the CSV file if found, None otherwise
        """
        possible_paths = [
            'StudentPerformanceFactors.csv',
            'data/StudentPerformanceFactors.csv',
            '../StudentPerformanceFactors.csv',
            '../../StudentPerformanceFactors.csv',
            '/kaggle/input/student-performance-factors/StudentPerformanceFactors.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file (optional)
            
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = self.find_data_file()
            
        if file_path is None:
            raise FileNotFoundError(
                "CSV file not found! Please provide a valid path."
            )
        
        self.df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully from: {file_path}")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'exam_score_stats': {
                'mean': self.df['Exam_Score'].mean(),
                'median': self.df['Exam_Score'].median(),
                'std': self.df['Exam_Score'].std(),
                'min': self.df['Exam_Score'].min(),
                'max': self.df['Exam_Score'].max()
            }
        }
        return info
    
    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding
        
        Returns:
            Encoded DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        print(f"🏷️  Encoding {len(categorical_columns)} categorical columns...")
        
        # One-hot encoding
        self.df_encoded = pd.get_dummies(self.df, columns=categorical_columns)
        
        # Convert boolean to int
        self.df_encoded = self.df_encoded.apply(
            lambda col: col.map({True: 1, False: 0}) if col.dtype == 'bool' else col
        )
        
        print(f"✓ Encoding complete. New shape: {self.df_encoded.shape}")
        return self.df_encoded
    
    def prepare_features(self, include_exam_score: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable
        
        Args:
            include_exam_score: Whether to keep Exam_Score as a feature
            
        Returns:
            Tuple of (features, target)
        """
        if self.df_encoded is None:
            raise ValueError("Data not encoded. Call encode_categorical() first.")
        
        X = self.df_encoded.drop('Exam_Score', axis=1)
        y = self.df_encoded['Exam_Score']
        
        print(f"✓ Features prepared: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def create_pass_fail_labels(self, threshold: float = 60.0) -> pd.Series:
        """
        Create binary pass/fail labels based on exam scores
        
        Args:
            threshold: Score threshold for passing (default: 60)
            
        Returns:
            Series with binary labels (1=Pass, 0=Fail)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        y_binary = (self.df['Exam_Score'] >= threshold).astype(int)
        pass_count = y_binary.sum()
        fail_count = len(y_binary) - pass_count
        
        print(f"✓ Pass/Fail labels created (threshold: {threshold})")
        print(f"  Pass: {pass_count} ({pass_count/len(y_binary)*100:.1f}%)")
        print(f"  Fail: {fail_count} ({fail_count/len(y_binary)*100:.1f}%)")
        
        return y_binary
