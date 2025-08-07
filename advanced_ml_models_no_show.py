#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED ML MODELS FOR RESPIRATORY DISEASE PREDICTION
Fixed version with AdaBoost compatibility - No interactive plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Configurazione per i grafici - NON INTERATTIVI
plt.style.use('default')
sns.set_palette('husl')
plt.ioff()  # Disabilita modalità interattiva

print('🚀 ADVANCED ML MODELS FOR RESPIRATORY DISEASE PREDICTION')
print('='*70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

try:
    # Load data with permission handling
    file_path = 'COLD 30.07.2025.xlsx'
    
    try:
        df = pd.read_excel(file_path)
    except PermissionError:
        print("⚠️  File in uso, creazione copia temporanea...")
        import shutil
        temp_file = 'temp_dataset.xlsx'
        shutil.copy2(file_path, temp_file)
        df = pd.read_excel(temp_file)
        print("✅ Dataset caricato da copia temporanea")
    
    print(f'📊 Dataset: {df.shape[0]} patients, {df.shape[1]} variables')
except Exception as e:
    print(f'❌ Errore nel caricamento del file: {e}')
    exit(1)

# Diagnosis mapping
diagnosis_mapping = {
    0: 'Altro',
    1: 'Asma bronchiale',
    2: 'BPCO', 
    3: 'Overlap asma/bpco'
}

print('\n🎯 Target Classes:')
for code, name in diagnosis_mapping.items():
    count = (df['Diagnosi'] == code).sum()
    print(f'  {code} - {name}: {count} patients')

# Preprocessing
df_ml = df.copy()

# Clean column names for compatibility
def clean_column_names(df):
    """Clean column names to remove special characters"""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)
    df_clean.columns = df_clean.columns.str.strip('_')
    return df_clean

df_ml = clean_column_names(df_ml)
print(f'✅ Column names cleaned for ML compatibility')

# Encode categorical variables
categorical_cols = df_ml.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    if col != 'Data_questionario':
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le

# Remove date column if present
if 'Data_questionario' in df_ml.columns:
    df_ml = df_ml.drop('Data_questionario', axis=1)

# Features and target
X = df_ml.drop(['Diagnosi', 'Identificativo'], axis=1)
y = df_ml['Diagnosi']

print(f'\n📈 Features: {X.shape[1]} variables')
print(f'🎯 Target: {len(y.unique())} classes')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\n✅ Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}')

# ============================================================================
# 2. ADVANCED ML MODELS
# ============================================================================

print('\n🤖 ADVANCED ML MODELS FOR MEDICAL DIAGNOSIS')
print('='*50)

# Define models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    ),
    
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True,
        class_weight='balanced'
    ),
    
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='minkowski'
    ),
    
    'Gaussian Naive Bayes': GaussianNB(
        var_smoothing=1e-9
    ),
    
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42
    )
}

print(f'🔧 {len(models)} models configured')

# ============================================================================
# 3. CROSS-VALIDATION EVALUATION
# ============================================================================

print('\n📊 CROSS-VALIDATION EVALUATION (5-FOLD)')
print('='*50)

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f'\n🔄 Evaluating {name}...')
    
    # Use scaled data for models that need it
    if name in ['SVM (RBF)', 'K-Nearest Neighbors', 'Neural Network']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    cv_results[name] = cv_scores
    print(f'   Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    print(f'   Range: [{cv_scores.min():.3f}, {cv_scores.max():.3f}]')

# Sort results
sorted_results = sorted(cv_results.items(), key=lambda x: x[1].mean(), reverse=True)

print('\n🏆 RANKING BY CROSS-VALIDATION ACCURACY:')
print('-' * 50)
for i, (name, scores) in enumerate(sorted_results, 1):
    print(f'{i:2d}. {name:<25}: {scores.mean():.3f} ± {scores.std():.3f}')

# ============================================================================
# 4. BEST MODEL EVALUATION
# ============================================================================

best_model_name = sorted_results[0][0]
best_model = models[best_model_name]

print(f'\n🥇 BEST MODEL: {best_model_name}')
print('='*50)

# Train and evaluate best model
if best_model_name in ['SVM (RBF)', 'K-Nearest Neighbors', 'Neural Network']:
    best_model.fit(X_train_scaled, y_train)
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'\n📈 PERFORMANCE METRICS:')
print(f'   Training Accuracy: {train_accuracy:.3f}')
print(f'   Test Accuracy: {test_accuracy:.3f}')
print(f'   Generalization Gap: {train_accuracy - test_accuracy:.3f}')

# Classification report
print('\n📋 DETAILED CLASSIFICATION REPORT:')
target_names = [diagnosis_mapping[i] for i in sorted(y.unique())]
print(classification_report(y_test, y_pred_test, target_names=target_names))

# Confusion Matrix - SALVATA COME FILE
print('\n🔍 CONFUSION MATRIX:')
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('Confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # Chiude la figura invece di mostrarla
print('💾 Confusion matrix salvata come: Confusion_matrix.png')

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print('\n🎯 FEATURE IMPORTANCE:')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Most Important Features - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Feature_importance_best_model.png', dpi=300, bbox_inches='tight')
    plt.close()  # Chiude la figura invece di mostrarla
    print('💾 Feature importance salvata come: Feature_importance_best_model.png')
    
    print('\nTop 10 Most Important Features:')
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f'{i:2d}. {row["feature"]:<30}: {row["importance"]:.4f}')

# ============================================================================
# 5. PREDICTION FUNCTION
# ============================================================================

def predict_new_patient(patient_data, model=best_model, scaler=scaler):
    """
    Predict diagnosis for a new patient
    
    Parameters:
    patient_data: dict or pandas Series with patient features
    model: trained model to use for prediction
    scaler: fitted scaler for feature normalization
    
    Returns:
    prediction: predicted class (0-3)
    probabilities: prediction probabilities for each class
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        missing_features = set(X.columns) - set(patient_df.columns)
        if missing_features:
            print(f"Warning: Missing features {missing_features}. Setting to 0.")
            for feature in missing_features:
                patient_df[feature] = 0
        
        # Reorder columns to match training data
        patient_df = patient_df[X.columns]
        
        # Scale if needed
        if best_model_name in ['SVM (RBF)', 'K-Nearest Neighbors', 'Neural Network']:
            patient_scaled = scaler.transform(patient_df)
            prediction = model.predict(patient_scaled)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(patient_scaled)[0]
            else:
                probabilities = None
        else:
            prediction = model.predict(patient_df)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(patient_df)[0]
            else:
                probabilities = None
        
        return prediction, probabilities
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

print('\n🔮 PREDICTION FUNCTION CREATED')
print('   Use predict_new_patient() to classify new patients')
print('   Returns: predicted class and probabilities')

print('\n📝 EXAMPLE USAGE:')
print('   prediction, probs = predict_new_patient(new_patient_data)')
print('   diagnosis = diagnosis_mapping[prediction]')
print('   confidence = max(probs) if probs is not None else "N/A"')

print('\n✅ ADVANCED ML PIPELINE COMPLETED!')
print('🎯 Ready for new patient diagnosis prediction')
print('='*70)

# Final comparison
print('\n📊 FINAL MODEL COMPARISON:')
results_df = pd.DataFrame({
    'Model': [name for name, _ in sorted_results],
    'CV_Accuracy': [scores.mean() for _, scores in sorted_results],
    'CV_Std': [scores.std() for _, scores in sorted_results]
})

print(results_df.to_string(index=False, float_format='%.3f'))

print(f'\n🏆 RECOMMENDED FOR PRODUCTION:')
print(f'   1. {best_model_name} (Best single model)')
print(f'   2. Ensemble of top 3 models (Higher robustness)')
print(f'   3. Consider hyperparameter tuning for optimization')

print('\n🎉 ANALISI COMPLETATA! Controlla i file grafici generati.')