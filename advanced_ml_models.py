# ============================================================================
# ADVANCED MACHINE LEARNING MODELS FOR RESPIRATORY DISEASE PREDICTION
# Excluding Logistic Regression - Focus on Tree-based and Ensemble Methods
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
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
# Optional imports - handle if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configurazione per i grafici
plt.style.use('default')
sns.set_palette('husl')

print('🚀 ADVANCED ML MODELS FOR RESPIRATORY DISEASE PREDICTION')
print('='*70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

# Load data
file_path = 'COLD 30.07.2025.xlsx'
df = pd.read_excel(file_path)

print(f'📊 Dataset: {df.shape[0]} patients, {df.shape[1]} variables')

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

# Clean column names for LightGBM compatibility
def clean_column_names(df):
    """Clean column names to remove special characters for LightGBM"""
    df_clean = df.copy()
    # Replace special characters with underscores
    df_clean.columns = df_clean.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    # Remove multiple consecutive underscores
    df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)
    # Remove leading/trailing underscores
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\n✅ Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}')

# ============================================================================
# 2. ADVANCED ML MODELS DEFINITION
# ============================================================================

print('\n🤖 ADVANCED ML MODELS FOR MEDICAL DIAGNOSIS')
print('='*50)

# Define advanced models (NO Logistic Regression)
models = {
    # Tree-based models (excellent for medical data)
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    ),
    
    # Advanced boosting algorithms (if available)
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

# Continue with other models
other_models = {
    
    # Support Vector Machine (good for medical data)
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True,
        class_weight='balanced'
    ),
    
    'SVM (Polynomial)': SVC(
        kernel='poly',
        degree=3,
        C=1.0,
        random_state=42,
        probability=True,
        class_weight='balanced'
    ),
    
    # K-Nearest Neighbors
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='minkowski'
    ),
    
    # Naive Bayes (good baseline for medical data)
    'Gaussian Naive Bayes': GaussianNB(
        var_smoothing=1e-9
    ),
    
    # Neural Network
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    ),
    
    # AdaBoost (fixed deprecated parameter)
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
}

# Merge all models
models.update(other_models)

print(f'🔧 {len(models)} advanced models configured')

# ============================================================================
# 3. MODEL EVALUATION WITH CROSS-VALIDATION
# ============================================================================

print('\n📊 CROSS-VALIDATION EVALUATION (5-FOLD)')
print('='*50)

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f'\n🔄 Evaluating {name}...')
    
    # Use scaled data for distance-based models, original for tree-based
    if name in ['SVM (RBF)', 'SVM (Polynomial)', 'K-Nearest Neighbors', 'Neural Network']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    cv_results[name] = cv_scores
    print(f'   Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    print(f'   Range: [{cv_scores.min():.3f}, {cv_scores.max():.3f}]')

# Sort results by mean accuracy
sorted_results = sorted(cv_results.items(), key=lambda x: x[1].mean(), reverse=True)

print('\n🏆 RANKING BY CROSS-VALIDATION ACCURACY:')
print('-' * 50)
for i, (name, scores) in enumerate(sorted_results, 1):
    print(f'{i:2d}. {name:<25}: {scores.mean():.3f} ± {scores.std():.3f}')

# ============================================================================
# 4. BEST MODEL TRAINING AND DETAILED EVALUATION
# ============================================================================

best_model_name = sorted_results[0][0]
best_model = models[best_model_name]

print(f'\n🥇 BEST MODEL: {best_model_name}')
print('='*50)

# Train best model
if best_model_name in ['SVM (RBF)', 'SVM (Polynomial)', 'K-Nearest Neighbors', 'Neural Network']:
    best_model.fit(X_train_scaled, y_train)
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    if hasattr(best_model, 'predict_proba'):
        y_pred_proba = best_model.predict_proba(X_test)

# Performance metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'\n📈 PERFORMANCE METRICS:')
print(f'   Training Accuracy: {train_accuracy:.3f}')
print(f'   Test Accuracy: {test_accuracy:.3f}')
print(f'   Generalization Gap: {train_accuracy - test_accuracy:.3f}')

# Detailed classification report
print('\n📋 DETAILED CLASSIFICATION REPORT:')
target_names = [diagnosis_mapping[i] for i in sorted(y.unique())]
print(classification_report(y_test, y_pred_test, target_names=target_names))

# Confusion Matrix
print('\n🔍 CONFUSION MATRIX:')
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test)

print('\n📊 PER-CLASS METRICS:')
for i, class_name in enumerate(target_names):
    print(f'   {class_name}:')
    print(f'     Precision: {precision[i]:.3f}')
    print(f'     Recall: {recall[i]:.3f}')
    print(f'     F1-Score: {f1[i]:.3f}')
    print(f'     Support: {support[i]}')

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print('\n🎯 FEATURE IMPORTANCE:')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 15 features
    top_features = feature_importance.head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Most Important Features - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print('\nTop 10 Most Important Features:')
    for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
        print(f'{i:2d}. {row["feature"]:<30}: {row["importance"]:.4f}')

# ============================================================================
# 5. ENSEMBLE MODEL (COMBINING BEST PERFORMERS)
# ============================================================================

print('\n🤝 ENSEMBLE MODEL CREATION')
print('='*40)

# Select top 3 models for ensemble
top_3_models = [name for name, _ in sorted_results[:3]]
print(f'Top 3 models for ensemble: {top_3_models}')

# Create ensemble
ensemble_estimators = []
for model_name in top_3_models:
    ensemble_estimators.append((model_name, models[model_name]))

ensemble_model = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft'  # Use probability-based voting
)

# Train ensemble
ensemble_model.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f'\n🎯 ENSEMBLE PERFORMANCE:')
print(f'   Ensemble Accuracy: {ensemble_accuracy:.3f}')
print(f'   Best Single Model: {test_accuracy:.3f}')
print(f'   Improvement: {ensemble_accuracy - test_accuracy:.3f}')

# ============================================================================
# 6. MODEL RECOMMENDATIONS FOR NEW PATIENT PREDICTION
# ============================================================================

print('\n💡 RECOMMENDATIONS FOR NEW PATIENT PREDICTION')
print('='*55)

print(f'\n🏆 BEST SINGLE MODEL: {best_model_name}')
print(f'   - Accuracy: {test_accuracy:.1%}')
print(f'   - Strengths: High performance, good interpretability')
print(f'   - Use case: Primary prediction model')

print(f'\n🤝 ENSEMBLE MODEL:')
print(f'   - Accuracy: {ensemble_accuracy:.1%}')
print(f'   - Strengths: Combines multiple algorithms, more robust')
print(f'   - Use case: Critical decisions requiring highest accuracy')

print('\n📋 MODEL CHARACTERISTICS FOR MEDICAL DATA:')
print('\n🌳 TREE-BASED MODELS (Recommended):')
print('   ✅ Handle mixed data types well')
print('   ✅ Provide feature importance')
print('   ✅ Robust to outliers')
print('   ✅ No need for feature scaling')
print('   ✅ Interpretable decision paths')

print('\n🚀 BOOSTING ALGORITHMS (Highly Recommended):')
print('   ✅ Excellent performance on tabular data')
print('   ✅ Handle class imbalance well')
print('   ✅ Feature importance available')
print('   ✅ Less prone to overfitting')

print('\n🎯 SUPPORT VECTOR MACHINES:')
print('   ✅ Good for high-dimensional data')
print('   ✅ Effective with limited samples')
print('   ⚠️  Requires feature scaling')
print('   ⚠️  Less interpretable')

print('\n🧠 NEURAL NETWORKS:')
print('   ✅ Can capture complex patterns')
print('   ✅ Good for large datasets')
print('   ⚠️  Requires more data')
print('   ⚠️  Less interpretable')
print('   ⚠️  Prone to overfitting')

# ============================================================================
# 7. PREDICTION FUNCTION FOR NEW PATIENTS
# ============================================================================

def predict_new_patient(patient_data, model=best_model, scaler=scaler, use_ensemble=False):
    """
    Predict diagnosis for a new patient
    
    Parameters:
    patient_data: dict or DataFrame with patient features
    model: trained model to use for prediction
    scaler: fitted StandardScaler
    use_ensemble: whether to use ensemble model
    
    Returns:
    prediction: predicted class
    probability: prediction probabilities
    """
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = patient_data.copy()
    
    # Apply same preprocessing
    for col in categorical_cols:
        if col in patient_df.columns and col in label_encoders:
            patient_df[col] = label_encoders[col].transform(patient_df[col].astype(str))
    
    # Remove unnecessary columns
    if 'Identificativo' in patient_df.columns:
        patient_df = patient_df.drop('Identificativo', axis=1)
    if 'Diagnosi' in patient_df.columns:
        patient_df = patient_df.drop('Diagnosi', axis=1)
    
    # Scale features if needed
    model_to_use = ensemble_model if use_ensemble else model
    
    if best_model_name in ['SVM (RBF)', 'SVM (Polynomial)', 'K-Nearest Neighbors', 'Neural Network'] or use_ensemble:
        patient_scaled = scaler.transform(patient_df)
        prediction = model_to_use.predict(patient_scaled)[0]
        probabilities = model_to_use.predict_proba(patient_scaled)[0]
    else:
        prediction = model_to_use.predict(patient_df)[0]
        if hasattr(model_to_use, 'predict_proba'):
            probabilities = model_to_use.predict_proba(patient_df)[0]
        else:
            probabilities = None
    
    return prediction, probabilities

print('\n🔮 PREDICTION FUNCTION CREATED')
print('   Use predict_new_patient() to classify new patients')
print('   Returns: predicted class and probabilities')

# Example usage
print('\n📝 EXAMPLE USAGE:')
print('   prediction, probs = predict_new_patient(new_patient_data)')
print('   diagnosis = diagnosis_mapping[prediction]')
print('   confidence = max(probs)')

print('\n✅ ADVANCED ML PIPELINE COMPLETED!')
print('🎯 Ready for new patient diagnosis prediction')
print('='*70)

# Summary of all models performance
print('\n📊 FINAL MODEL COMPARISON:')
results_df = pd.DataFrame({
    'Model': [name for name, _ in sorted_results],
    'CV_Accuracy': [scores.mean() for _, scores in sorted_results],
    'CV_Std': [scores.std() for _, scores in sorted_results]
})

print(results_df.to_string(index=False, float_format='%.3f'))

print(f'\n🏆 RECOMMENDED FOR PRODUCTION:')
print(f'   1. {best_model_name} (Single model)')
print(f'   2. Ensemble of top 3 models (Higher accuracy)')
print(f'   3. Consider XGBoost/LightGBM for large-scale deployment')