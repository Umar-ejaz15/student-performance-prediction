"""
Student Performance Prediction Models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict
import pickle
import os


class ScorePredictor:
    """Predict student exam scores (regression)"""
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize Score Predictor
        
        Args:
            model_type: Type of model ('linear', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the regression model
        
        Args:
            X_train: Training features
            y_train: Training target (exam scores)
        """
        print(f"Training {self.model_type} model for score prediction...")
        self.model.fit(X_train, y_train)
        print("✓ Model training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True test scores
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return self.metrics
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model_type == 'linear':
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
        elif self.model_type == 'random_forest':
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            importance = pd.DataFrame()
        
        return importance
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")


class PassFailClassifier:
    """Classify students as Pass or Fail (classification)"""
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize Pass/Fail Classifier
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the classification model
        
        Args:
            X_train: Training features
            y_train: Training target (pass/fail labels)
        """
        print(f"Training {self.model_type} model for pass/fail classification...")
        self.model.fit(X_train, y_train)
        print("✓ Model training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted classes (0=Fail, 1=Pass)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.metrics
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model_type == 'logistic':
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': self.model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
        elif self.model_type in ['random_forest', 'gradient_boosting']:
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            importance = pd.DataFrame()
        
        return importance
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    @staticmethod
    def compare_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name: model_instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            result = {'Model': name}
            result.update(metrics)
            results.append(result)
        
        return pd.DataFrame(results)
