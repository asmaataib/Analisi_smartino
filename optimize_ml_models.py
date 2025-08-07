#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OTTIMIZZAZIONE AVANZATA DEI MODELLI ML
Tecniche per migliorare l'accuratezza oltre il 67%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier,
    BaggingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configurazione
plt.style.use('default')
sns.set_palette('husl')
np.random.seed(42)

print('🚀 OTTIMIZZAZIONE AVANZATA MODELLI ML')
print('='*60)
print('🎯 Obiettivo: Superare il 67% di accuratezza')
print('='*60)

# ============================================================================
# 1. CARICAMENTO E PREPROCESSING AVANZATO
# ============================================================================

def clean_column_names(df):
    """Pulisce i nomi delle colonne"""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)
    df_clean.columns = df_clean.columns.str.strip('_')
    return df_clean

def advanced_preprocessing(df):
    """Preprocessing avanzato con feature engineering"""
    print('🔧 PREPROCESSING AVANZATO')
    print('-'*40)
    
    df_processed = clean_column_names(df.copy())
    
    # Rimuovi colonne non necessarie
    if 'Data_questionario' in df_processed.columns:
        df_processed = df_processed.drop('Data_questionario', axis=1)
    
    # Encoding categorico
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f'✅ Encoded: {col}')
    
    # Feature Engineering
    print('\n🛠️  FEATURE ENGINEERING')
    
    # 1. Interazioni tra features importanti (se esistono)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Diagnosi', 'Identificativo']]
    
    # Aggiungi alcune interazioni chiave
    if len(numeric_cols) >= 2:
        # Esempio: interazione età con altri fattori
        age_col = None
        for col in numeric_cols:
            if 'eta' in col.lower() or 'age' in col.lower():
                age_col = col
                break
        
        if age_col and len(numeric_cols) > 1:
            other_cols = [col for col in numeric_cols[:3] if col != age_col]
            for other_col in other_cols:
                new_col = f'{age_col}_x_{other_col}'
                df_processed[new_col] = df_processed[age_col] * df_processed[other_col]
                print(f'➕ Aggiunta interazione: {new_col}')
    
    # 2. Binning di variabili continue
    for col in numeric_cols[:2]:  # Solo prime 2 per evitare esplosione features
        if df_processed[col].nunique() > 10:
            df_processed[f'{col}_binned'] = pd.cut(df_processed[col], bins=5, labels=False)
            print(f'📊 Binning: {col}_binned')
    
    # 3. Statistiche aggregate
    if len(numeric_cols) >= 3:
        df_processed['numeric_mean'] = df_processed[numeric_cols[:5]].mean(axis=1)
        df_processed['numeric_std'] = df_processed[numeric_cols[:5]].std(axis=1)
        print('📈 Aggiunte statistiche aggregate')
    
    return df_processed, label_encoders

# Carica dati
file_path = 'COLD 30.07.2025.xlsx'
df = pd.read_excel(file_path)

print(f'📊 Dataset originale: {df.shape[0]} pazienti, {df.shape[1]} variabili')

# Mapping diagnosi
diagnosis_mapping = {
    0: 'Altro',
    1: 'Asma bronchiale', 
    2: 'BPCO',
    3: 'Overlap asma/bpco'
}

# Analisi distribuzione classi
print('\n📊 DISTRIBUZIONE CLASSI ORIGINALE:')
class_counts = df['Diagnosi'].value_counts().sort_index()
for code, count in class_counts.items():
    name = diagnosis_mapping[code]
    perc = (count / len(df)) * 100
    print(f'  {code} - {name}: {count} ({perc:.1f}%)')

# Preprocessing avanzato
df_ml, label_encoders = advanced_preprocessing(df)

# Separa features e target
X = df_ml.drop(['Diagnosi', 'Identificativo'], axis=1)
y = df_ml['Diagnosi']

print(f'\n📈 Features finali: {X.shape[1]} variabili')
print(f'🎯 Target: {len(y.unique())} classi')

# ============================================================================
# 2. STRATEGIE DI BILANCIAMENTO CLASSI
# ============================================================================

print('\n⚖️  STRATEGIE DI BILANCIAMENTO CLASSI')
print('='*50)

def test_sampling_strategies(X, y):
    """Testa diverse strategie di bilanciamento"""
    strategies = {
        'Original': None,
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    results = {}
    
    for name, sampler in strategies.items():
        if sampler is None:
            X_resampled, y_resampled = X, y
        else:
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y)
            except Exception as e:
                print(f'❌ Errore con {name}: {e}')
                continue
        
        # Test rapido con Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        cv_scores = cross_val_score(rf, X_resampled, y_resampled, cv=3, scoring='accuracy')
        
        results[name] = {
            'accuracy': cv_scores.mean(),
            'std': cv_scores.std(),
            'shape': X_resampled.shape,
            'distribution': pd.Series(y_resampled).value_counts().sort_index().tolist()
        }
        
        print(f'{name:15}: Acc={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Shape={X_resampled.shape}')
    
    return results

# Testa strategie di bilanciamento
sampling_results = test_sampling_strategies(X, y)

# Seleziona la migliore strategia
best_sampling = max(sampling_results.items(), key=lambda x: x[1]['accuracy'])
print(f'\n🏆 Migliore strategia: {best_sampling[0]} (Acc: {best_sampling[1]["accuracy"]:.3f})')

# ============================================================================
# 3. FEATURE SELECTION AVANZATA
# ============================================================================

print('\n🎯 FEATURE SELECTION AVANZATA')
print('='*50)

def advanced_feature_selection(X, y):
    """Selezione avanzata delle features"""
    feature_scores = {}
    
    # 1. Univariate Selection
    selector_univariate = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]//2))
    X_univariate = selector_univariate.fit_transform(X, y)
    univariate_features = X.columns[selector_univariate.get_support()].tolist()
    
    # 2. Recursive Feature Elimination
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf_selector, n_features_to_select=min(15, X.shape[1]//3))
    X_rfe = rfe.fit_transform(X, y)
    rfe_features = X.columns[rfe.support_].tolist()
    
    # 3. Model-based Selection
    rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_importance.fit(X, y)
    selector_model = SelectFromModel(rf_importance, threshold='median')
    X_model = selector_model.fit_transform(X, y)
    model_features = X.columns[selector_model.get_support()].tolist()
    
    # Combina le selezioni
    all_selected = set(univariate_features + rfe_features + model_features)
    
    print(f'📊 Features selezionate:')
    print(f'  Univariate: {len(univariate_features)}')
    print(f'  RFE: {len(rfe_features)}')
    print(f'  Model-based: {len(model_features)}')
    print(f'  Totali uniche: {len(all_selected)}')
    
    return list(all_selected), rf_importance.feature_importances_

# Applica feature selection
selected_features, feature_importances = advanced_feature_selection(X, y)
X_selected = X[selected_features]

print(f'✅ Ridotte da {X.shape[1]} a {X_selected.shape[1]} features')

# ============================================================================
# 4. HYPERPARAMETER TUNING AVANZATO
# ============================================================================

print('\n🔧 HYPERPARAMETER TUNING AVANZATO')
print('='*50)

def optimize_models(X, y):
    """Ottimizzazione iperparametri per modelli chiave"""
    
    # Configurazioni per GridSearch
    param_grids = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
        }
    }
    
    optimized_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in param_grids.items():
        print(f'\n🔄 Ottimizzando {name}...')
        
        # Usa RandomizedSearchCV per velocità
        search = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=20,  # Ridotto per velocità
            cv=cv,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        # Scala i dati se necessario
        if name == 'SVM':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            search.fit(X_scaled, y)
        else:
            search.fit(X, y)
        
        optimized_models[name] = {
            'model': search.best_estimator_,
            'score': search.best_score_,
            'params': search.best_params_,
            'scaler': scaler if name == 'SVM' else None
        }
        
        print(f'✅ {name}: {search.best_score_:.3f}')
        print(f'   Migliori parametri: {search.best_params_}')
    
    return optimized_models

# Ottimizza modelli
optimized_models = optimize_models(X_selected, y)

# ============================================================================
# 5. ENSEMBLE METHODS AVANZATI
# ============================================================================

print('\n🤝 ENSEMBLE METHODS AVANZATI')
print('='*50)

def create_advanced_ensembles(optimized_models, X, y):
    """Crea ensemble avanzati"""
    
    # Prepara modelli base
    base_models = []
    for name, config in optimized_models.items():
        base_models.append((name, config['model']))
    
    # 1. Voting Classifier (Hard e Soft)
    voting_hard = VotingClassifier(estimators=base_models, voting='hard')
    voting_soft = VotingClassifier(estimators=base_models, voting='soft')
    
    # 2. Stacking Classifier
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=5
    )
    
    # 3. Bagging con migliore modello
    best_model_name = max(optimized_models.items(), key=lambda x: x[1]['score'])[0]
    best_base_model = optimized_models[best_model_name]['model']
    
    bagging = BaggingClassifier(
        estimator=best_base_model,
        n_estimators=10,
        random_state=42
    )
    
    ensembles = {
        'Voting_Hard': voting_hard,
        'Voting_Soft': voting_soft,
        'Stacking': stacking,
        'Bagging': bagging
    }
    
    return ensembles

# Crea ensemble
ensemble_models = create_advanced_ensembles(optimized_models, X_selected, y)

# ============================================================================
# 6. VALUTAZIONE COMPLETA
# ============================================================================

print('\n📊 VALUTAZIONE COMPLETA MODELLI')
print('='*50)

def comprehensive_evaluation(models, X, y, use_sampling=True):
    """Valutazione completa con cross-validation robusto"""
    
    # Applica la migliore strategia di sampling se richiesto
    if use_sampling and best_sampling[0] != 'Original':
        if best_sampling[0] == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif best_sampling[0] == 'ADASYN':
            sampler = ADASYN(random_state=42)
        else:
            sampler = SMOTE(random_state=42)  # Default
        
        X_eval, y_eval = sampler.fit_resample(X, y)
        print(f'✅ Applicato {best_sampling[0]}: {X.shape} → {X_eval.shape}')
    else:
        X_eval, y_eval = X, y
    
    # Cross-validation più robusto
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    results = {}
    
    # Valuta modelli ottimizzati
    for name, config in models.items():
        print(f'\n🔄 Valutando {name}...')
        
        model = config['model']
        scaler = config.get('scaler')
        
        if scaler:
            # Pipeline con scaling
            X_scaled = scaler.fit_transform(X_eval)
            scores = cross_val_score(model, X_scaled, y_eval, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X_eval, y_eval, cv=cv, scoring='accuracy')
        
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f'   Accuratezza: {scores.mean():.3f} ± {scores.std():.3f}')
        print(f'   Range: [{scores.min():.3f}, {scores.max():.3f}]')
    
    # Valuta ensemble
    print('\n🤝 Valutando Ensemble...')
    for name, model in ensemble_models.items():
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=cv, scoring='accuracy')
            results[f'Ensemble_{name}'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f'   {name}: {scores.mean():.3f} ± {scores.std():.3f}')
        except Exception as e:
            print(f'   ❌ Errore con {name}: {e}')
    
    return results, X_eval, y_eval

# Valutazione completa
final_results, X_final, y_final = comprehensive_evaluation(optimized_models, X_selected, y)

# ============================================================================
# 7. RISULTATI FINALI E RACCOMANDAZIONI
# ============================================================================

print('\n🏆 RISULTATI FINALI')
print('='*60)

# Ordina risultati
sorted_final = sorted(final_results.items(), key=lambda x: x[1]['mean'], reverse=True)

print('📊 RANKING FINALE (con ottimizzazioni):')
print('-' * 60)
for i, (name, metrics) in enumerate(sorted_final, 1):
    improvement = metrics['mean'] - 0.67  # Confronto con baseline 67%
    status = '🚀' if improvement > 0 else '📉'
    print(f'{i:2d}. {name:<25}: {metrics["mean"]:.3f} ± {metrics["std"]:.3f} {status}')
    if improvement > 0:
        print(f'    Miglioramento: +{improvement:.3f} ({improvement*100:.1f}%)')

# Migliore modello
best_final = sorted_final[0]
print(f'\n🥇 MIGLIORE MODELLO: {best_final[0]}')
print(f'   Accuratezza: {best_final[1]["mean"]:.3f} ± {best_final[1]["std"]:.3f}')
print(f'   Miglioramento vs baseline: +{best_final[1]["mean"] - 0.67:.3f}')

# Raccomandazioni
print('\n💡 RACCOMANDAZIONI PER ULTERIORI MIGLIORAMENTI:')
print('-' * 60)
print('1. 📊 Raccogliere più dati (specialmente per classi rare)')
print('2. 🔬 Feature engineering più sofisticato (domain knowledge)')
print('3. 🧠 Provare modelli più avanzati (XGBoost, LightGBM, CatBoost)')
print('4. 🔄 Ottimizzazione Bayesiana degli iperparametri')
print('5. 🎯 Ensemble più complessi (multi-level stacking)')
print('6. 📈 Validazione esterna su dataset indipendente')
print('7. 🏥 Incorporare expertise medica nella feature selection')

# Salva risultati
results_df = pd.DataFrame({
    'Modello': [name for name, _ in sorted_final],
    'Accuratezza_Media': [metrics['mean'] for _, metrics in sorted_final],
    'Deviazione_Standard': [metrics['std'] for _, metrics in sorted_final],
    'Miglioramento_vs_Baseline': [metrics['mean'] - 0.67 for _, metrics in sorted_final]
})

results_df.to_excel('risultati_ottimizzazione.xlsx', index=False)
print(f'\n💾 Risultati salvati in: risultati_ottimizzazione.xlsx')

print('\n✅ OTTIMIZZAZIONE COMPLETATA!')
print('🎯 Controlla i risultati per vedere i miglioramenti ottenuti')