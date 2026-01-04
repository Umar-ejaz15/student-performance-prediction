"""
Enhanced Visualization Module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import os


class PerformanceVisualizer:
    """Create comprehensive visualizations for student performance analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = figsize
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_score_distribution(self, scores: pd.Series, save: bool = False) -> None:
        """
        Plot exam score distribution with histogram and box plot
        
        Args:
            scores: Exam scores
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Histogram
        axes[0].hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.2f}')
        axes[0].axvline(scores.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {scores.median():.2f}')
        axes[0].set_xlabel('Exam Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of Exam Scores', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        box = axes[1].boxplot(scores, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[1].set_ylabel('Exam Score', fontsize=12, fontweight='bold')
        axes[1].set_title('Exam Score Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Violin plot
        parts = axes[2].violinplot([scores], vert=True, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.7)
        axes[2].set_ylabel('Exam Score', fontsize=12, fontweight='bold')
        axes[2].set_title('Exam Score Violin Plot', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Mean: {scores.mean():.2f} | Median: {scores.median():.2f} | Std: {scores.std():.2f}")
    
    def plot_pass_fail_analysis(self, scores: pd.Series, threshold: float = 60.0, save: bool = False) -> None:
        """
        Plot pass/fail analysis
        
        Args:
            scores: Exam scores
            threshold: Passing threshold
            save: Whether to save the plot
        """
        pass_fail = (scores >= threshold).map({True: 'Pass', False: 'Fail'})
        counts = pass_fail.value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = ['#90EE90', '#FFB6C6']
        explode = (0.05, 0.05)
        axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                    colors=colors, explode=explode, shadow=True)
        axes[0].set_title('Pass vs Fail Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = axes[1].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
        axes[1].set_ylabel('Number of Students', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Pass/Fail Count (Threshold: {threshold})', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/pass_fail_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, top_n: int = 15, save: bool = False) -> None:
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame with numeric columns
            top_n: Number of top correlations to show
            save: Whether to save the plot
        """
        # Calculate correlations with Exam_Score
        correlations = df.corr()['Exam_Score'].drop('Exam_Score').sort_values(key=abs, ascending=False)
        
        # Select top features
        top_features = correlations.head(top_n).index.tolist() + ['Exam_Score']
        correlation_matrix = df[top_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap (Top {top_n} Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distributions(self, df: pd.DataFrame, features: list, save: bool = False) -> None:
        """
        Plot distributions of multiple features
        
        Args:
            df: DataFrame
            features: List of features to plot
            save: Whether to save the plot
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if df[feature].dtype in ['int64', 'float64']:
                axes[idx].hist(df[feature], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel(feature, fontsize=10, fontweight='bold')
                axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model", save: bool = False) -> None:
        """
        Plot actual vs predicted values with residuals
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolor='black', s=60)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Exam Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Exam Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='coral', edgecolor='black', s=60)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Exam Score', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Error distribution
        axes[2].hist(residuals, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[2].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[2].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/prediction_results_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15, save: bool = False) -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with Feature and Coefficient/Importance columns
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Determine if it's coefficients or importance
        value_col = 'Coefficient' if 'Coefficient' in top_features.columns else 'Importance'
        
        if value_col == 'Coefficient':
            colors = ['green' if x > 0 else 'red' for x in top_features[value_col]]
        else:
            colors = 'steelblue'
        
        plt.barh(top_features['Feature'], top_features[value_col], color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel(value_col, fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        
        if value_col == 'Coefficient':
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: list = ['Fail', 'Pass'],
                             save: bool = False) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            save: Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                    yticklabels=labels, cbar_kws={"shrink": 0.8}, linewidths=2, linecolor='black')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_metrics(self, metrics: dict, save: bool = False) -> None:
        """
        Plot classification metrics
        
        Args:
            metrics: Dictionary with accuracy, precision, recall, f1
            save: Whether to save the plot
        """
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'],
                        metrics['recall'], metrics['f1']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        bars = ax1.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                       edgecolor='black', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Classification Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        metric_values_radar = metric_values + [metric_values[0]]
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, metric_values_radar, 'o-', linewidth=2, color='#3498db')
        ax2.fill(angles, metric_values_radar, alpha=0.25, color='#3498db')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names)
        ax2.set_ylim(0, 1)
        ax2.set_title('Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_by_category(self, df: pd.DataFrame, category_col: str,
                               score_col: str = 'Exam_Score', save: bool = False) -> None:
        """
        Plot exam scores by category
        
        Args:
            df: DataFrame
            category_col: Column name for categories
            score_col: Column name for scores
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        df_grouped = df.groupby(category_col)[score_col].mean().sort_values(ascending=False)
        
        bars = plt.bar(range(len(df_grouped)), df_grouped.values,
                       color='teal', edgecolor='black', linewidth=2, alpha=0.7)
        plt.xticks(range(len(df_grouped)), df_grouped.index, rotation=45, ha='right')
        plt.xlabel(category_col, fontsize=12, fontweight='bold')
        plt.ylabel(f'Average {score_col}', fontsize=12, fontweight='bold')
        plt.title(f'Average {score_col} by {category_col}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, df_grouped.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/score_by_{category_col}.png', dpi=300, bbox_inches='tight')
        plt.show()
