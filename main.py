"""
Main Application - Student Performance Prediction System
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import DataLoader
from src.models.predictor import ScorePredictor, PassFailClassifier
from src.visualization.visualizer import PerformanceVisualizer


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def main():
    """Main application entry point"""
    
    print_header("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    
    # ============================
    # 1. LOAD AND PREPARE DATA
    # ============================
    print_header("1. DATA LOADING AND PREPROCESSING")
    
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Get data info
    info = data_loader.get_data_info()
    print(f"\n📊 Dataset Overview:")
    print(f"  • Shape: {info['shape']}")
    print(f"  • Numeric columns: {len(info['numeric_columns'])}")
    print(f"  • Categorical columns: {len(info['categorical_columns'])}")
    print(f"\n📈 Exam Score Statistics:")
    for key, value in info['exam_score_stats'].items():
        print(f"  • {key.capitalize()}: {value:.2f}")
    
    # Encode categorical variables
    df_encoded = data_loader.encode_categorical()
    
    # Prepare features
    X, y_scores = data_loader.prepare_features()
    y_binary = data_loader.create_pass_fail_labels(threshold=60.0)
    
    # Split data
    X_train, X_test, y_train_scores, y_test_scores = train_test_split(
        X, y_scores, test_size=0.2, random_state=42
    )
    _, _, y_train_binary, y_test_binary = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # ============================
    # 2. VISUALIZATION
    # ============================
    print_header("2. DATA VISUALIZATION")
    
    visualizer = PerformanceVisualizer()
    
    print("📊 Creating visualizations...")
    
    # Score distribution
    visualizer.plot_score_distribution(df['Exam_Score'], save=True)
    
    # Pass/Fail analysis
    visualizer.plot_pass_fail_analysis(df['Exam_Score'], threshold=60.0, save=True)
    
    # Correlation heatmap (only numeric features)
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    visualizer.plot_correlation_heatmap(numeric_df, top_n=15, save=True)
    
    # Feature distributions
    key_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                    'Tutoring_Sessions', 'Physical_Activity']
    visualizer.plot_feature_distributions(df, key_features, save=True)
    
    # ============================
    # 3. SCORE PREDICTION MODEL
    # ============================
    print_header("3. EXAM SCORE PREDICTION (REGRESSION)")
    
    # Train Linear Regression
    print("\n📚 Training Linear Regression model...")
    score_predictor_lr = ScorePredictor(model_type='linear')
    score_predictor_lr.train(X_train, y_train_scores)
    
    # Evaluate
    metrics_lr = score_predictor_lr.evaluate(X_test, y_test_scores)
    print(f"\n📊 Linear Regression Performance:")
    print(f"  • Mean Absolute Error (MAE): {metrics_lr['mae']:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {metrics_lr['rmse']:.4f}")
    print(f"  • R² Score: {metrics_lr['r2']:.4f} ({metrics_lr['r2']*100:.2f}%)")
    
    # Predictions
    y_pred_lr = score_predictor_lr.predict(X_test)
    visualizer.plot_prediction_results(y_test_scores.values, y_pred_lr,
                                       model_name="Linear Regression", save=True)
    
    # Feature importance
    importance_lr = score_predictor_lr.get_feature_importance(X.columns.tolist())
    visualizer.plot_feature_importance(importance_lr, top_n=15, save=True)
    
    # Save model
    score_predictor_lr.save_model('models/score_predictor_linear.pkl')
    
    # Train Random Forest
    print("\n🌲 Training Random Forest model...")
    score_predictor_rf = ScorePredictor(model_type='random_forest')
    score_predictor_rf.train(X_train, y_train_scores)
    
    # Evaluate
    metrics_rf = score_predictor_rf.evaluate(X_test, y_test_scores)
    print(f"\n📊 Random Forest Performance:")
    print(f"  • Mean Absolute Error (MAE): {metrics_rf['mae']:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {metrics_rf['rmse']:.4f}")
    print(f"  • R² Score: {metrics_rf['r2']:.4f} ({metrics_rf['r2']*100:.2f}%)")
    
    # Predictions
    y_pred_rf = score_predictor_rf.predict(X_test)
    visualizer.plot_prediction_results(y_test_scores.values, y_pred_rf,
                                       model_name="Random Forest", save=True)
    
    # Feature importance
    importance_rf = score_predictor_rf.get_feature_importance(X.columns.tolist())
    visualizer.plot_feature_importance(importance_rf, top_n=15, save=True)
    
    # Save model
    score_predictor_rf.save_model('models/score_predictor_rf.pkl')
    
    # ============================
    # 4. PASS/FAIL CLASSIFICATION
    # ============================
    print_header("4. PASS/FAIL CLASSIFICATION")
    
    # Train Logistic Regression
    print("\n📚 Training Logistic Regression classifier...")
    classifier_lr = PassFailClassifier(model_type='logistic')
    classifier_lr.train(X_train, y_train_binary)
    
    # Evaluate
    metrics_class_lr = classifier_lr.evaluate(X_test, y_test_binary)
    print(f"\n📊 Logistic Regression Performance:")
    print(f"  • Accuracy: {metrics_class_lr['accuracy']:.4f} ({metrics_class_lr['accuracy']*100:.2f}%)")
    print(f"  • Precision: {metrics_class_lr['precision']:.4f}")
    print(f"  • Recall: {metrics_class_lr['recall']:.4f}")
    print(f"  • F1-Score: {metrics_class_lr['f1']:.4f}")
    
    # Visualizations
    visualizer.plot_confusion_matrix(metrics_class_lr['confusion_matrix'], save=True)
    visualizer.plot_classification_metrics(metrics_class_lr, save=True)
    
    # Save model
    classifier_lr.save_model('models/classifier_logistic.pkl')
    
    # Train Random Forest Classifier
    print("\n🌲 Training Random Forest classifier...")
    classifier_rf = PassFailClassifier(model_type='random_forest')
    classifier_rf.train(X_train, y_train_binary)
    
    # Evaluate
    metrics_class_rf = classifier_rf.evaluate(X_test, y_test_binary)
    print(f"\n📊 Random Forest Classifier Performance:")
    print(f"  • Accuracy: {metrics_class_rf['accuracy']:.4f} ({metrics_class_rf['accuracy']*100:.2f}%)")
    print(f"  • Precision: {metrics_class_rf['precision']:.4f}")
    print(f"  • Recall: {metrics_class_rf['recall']:.4f}")
    print(f"  • F1-Score: {metrics_class_rf['f1']:.4f}")
    
    # Visualizations
    visualizer.plot_confusion_matrix(metrics_class_rf['confusion_matrix'], save=True)
    visualizer.plot_classification_metrics(metrics_class_rf, save=True)
    
    # Save model
    classifier_rf.save_model('models/classifier_rf.pkl')
    
    # Train Gradient Boosting Classifier
    print("\n🚀 Training Gradient Boosting classifier...")
    classifier_gb = PassFailClassifier(model_type='gradient_boosting')
    classifier_gb.train(X_train, y_train_binary)
    
    # Evaluate
    metrics_class_gb = classifier_gb.evaluate(X_test, y_test_binary)
    print(f"\n📊 Gradient Boosting Classifier Performance:")
    print(f"  • Accuracy: {metrics_class_gb['accuracy']:.4f} ({metrics_class_gb['accuracy']*100:.2f}%)")
    print(f"  • Precision: {metrics_class_gb['precision']:.4f}")
    print(f"  • Recall: {metrics_class_gb['recall']:.4f}")
    print(f"  • F1-Score: {metrics_class_gb['f1']:.4f}")
    
    # Visualizations
    visualizer.plot_confusion_matrix(metrics_class_gb['confusion_matrix'], save=True)
    visualizer.plot_classification_metrics(metrics_class_gb, save=True)
    
    # Save model
    classifier_gb.save_model('models/classifier_gb.pkl')
    
    # ============================
    # 5. MODEL COMPARISON
    # ============================
    print_header("5. MODEL COMPARISON SUMMARY")
    
    print("\n📊 Score Prediction Models:")
    comparison_scores = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'MAE': [metrics_lr['mae'], metrics_rf['mae']],
        'RMSE': [metrics_lr['rmse'], metrics_rf['rmse']],
        'R²': [metrics_lr['r2'], metrics_rf['r2']]
    })
    print(comparison_scores.to_string(index=False))
    
    print("\n📊 Pass/Fail Classification Models:")
    comparison_class = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [metrics_class_lr['accuracy'], metrics_class_rf['accuracy'], metrics_class_gb['accuracy']],
        'Precision': [metrics_class_lr['precision'], metrics_class_rf['precision'], metrics_class_gb['precision']],
        'Recall': [metrics_class_lr['recall'], metrics_class_rf['recall'], metrics_class_gb['recall']],
        'F1-Score': [metrics_class_lr['f1'], metrics_class_rf['f1'], metrics_class_gb['f1']]
    })
    print(comparison_class.to_string(index=False))
    
    # ============================
    # 6. PREDICTION EXAMPLE
    # ============================
    print_header("6. MAKING PREDICTIONS ON NEW DATA")
    
    # Example student
    print("\n📝 Example: Predicting for a new student...")
    example_idx = 0
    example_student = X_test.iloc[example_idx:example_idx+1]
    
    # Score prediction
    predicted_score_lr = score_predictor_lr.predict(example_student)[0]
    predicted_score_rf = score_predictor_rf.predict(example_student)[0]
    actual_score = y_test_scores.iloc[example_idx]
    
    print(f"\n🎓 Student Score Predictions:")
    print(f"  • Actual Score: {actual_score:.2f}")
    print(f"  • Linear Regression Prediction: {predicted_score_lr:.2f}")
    print(f"  • Random Forest Prediction: {predicted_score_rf:.2f}")
    
    # Pass/Fail prediction
    predicted_class_lr = classifier_lr.predict(example_student)[0]
    predicted_class_rf = classifier_rf.predict(example_student)[0]
    predicted_class_gb = classifier_gb.predict(example_student)[0]
    actual_class = y_test_binary.iloc[example_idx]
    
    # Get probabilities
    prob_lr = classifier_lr.predict_proba(example_student)[0]
    prob_rf = classifier_rf.predict_proba(example_student)[0]
    prob_gb = classifier_gb.predict_proba(example_student)[0]
    
    print(f"\n🎯 Pass/Fail Predictions:")
    print(f"  • Actual: {'Pass' if actual_class == 1 else 'Fail'}")
    print(f"  • Logistic Regression: {'Pass' if predicted_class_lr == 1 else 'Fail'} (Confidence: {prob_lr[predicted_class_lr]:.2%})")
    print(f"  • Random Forest: {'Pass' if predicted_class_rf == 1 else 'Fail'} (Confidence: {prob_rf[predicted_class_rf]:.2%})")
    print(f"  • Gradient Boosting: {'Pass' if predicted_class_gb == 1 else 'Fail'} (Confidence: {prob_gb[predicted_class_gb]:.2%})")
    
    # ============================
    # 7. FINAL SUMMARY
    # ============================
    print_header("✅ ANALYSIS COMPLETE")
    
    print("📁 All visualizations saved to 'outputs/' directory")
    print("💾 All trained models saved to 'models/' directory")
    print("\n🎉 Student Performance Prediction System is ready!")
    print("\nYou can now use the trained models to:")
    print("  1. Predict exam scores for new students")
    print("  2. Classify students as Pass/Fail")
    print("  3. Identify key factors affecting student performance")
    print("  4. Generate comprehensive performance reports")
    

if __name__ == "__main__":
    main()
