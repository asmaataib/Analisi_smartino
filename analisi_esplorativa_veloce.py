#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALISI ESPLORATIVA VELOCE DEL DATASET
Versione ottimizzata per analisi rapida di distribuzione, correlazioni e feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurazione per evitare blocchi
plt.ioff()  # Disabilita modalità interattiva
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    """Carica il dataset"""
    print('📊 CARICAMENTO DATASET')
    print('='*40)
    
    try:
        # Prova con copia temporanea
        import shutil
        import os
        temp_file = 'temp_analysis.xlsx'
        shutil.copy2('COLD 30.07.2025.xlsx', temp_file)
        df = pd.read_excel(temp_file, engine='openpyxl')
        os.remove(temp_file)
        print(f'✅ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne')
        
        # Pulisci nomi colonne
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return df
        
    except Exception as e:
        print(f'❌ Errore: {e}')
        return None

def analyze_target(df):
    """Analizza la variabile target"""
    print('\n🎯 ANALISI TARGET')
    print('='*30)
    
    # Trova colonna target
    target_col = 'Diagnosi'
    if target_col not in df.columns:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            target_col = categorical_cols[0]
    
    print(f'Target identificato: {target_col}')
    
    # Distribuzione
    dist = df[target_col].value_counts()
    perc = df[target_col].value_counts(normalize=True) * 100
    
    print('\nDistribuzione classi:')
    for classe, count in dist.items():
        print(f'   {classe}: {count} ({perc[classe]:.1f}%)')
    
    # Salva grafico semplice
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    dist.plot(kind='bar', color='lightblue')
    plt.title('Distribuzione Target')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.pie(dist.values, labels=dist.index, autopct='%1.1f%%')
    plt.title('Percentuali Target')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()  # Chiudi per evitare blocchi
    print('✅ Salvato: target_distribution.png')
    
    return target_col

def analyze_correlations(df, target_col):
    """Analizza correlazioni"""
    print('\n🔥 ANALISI CORRELAZIONI')
    print('='*35)
    
    # Seleziona solo variabili numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    print(f'Variabili numeriche trovate: {len(numeric_cols)}')
    
    if len(numeric_cols) < 2:
        print('❌ Insufficienti variabili numeriche per correlazione')
        return None
    
    # Prepara dati
    corr_data = df[numeric_cols].copy()
    
    # Aggiungi target encodato
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        corr_data[target_col + '_encoded'] = le.fit_transform(df[target_col])
    else:
        corr_data[target_col] = df[target_col]
    
    # Calcola correlazioni
    corr_matrix = corr_data.corr()
    
    # Heatmap semplificata
    plt.figure(figsize=(12, 10))
    
    # Usa solo le prime 15 variabili per evitare sovraccarico
    if len(corr_matrix) > 15:
        # Seleziona le variabili più correlate con il target
        target_corr_col = target_col + '_encoded' if target_col + '_encoded' in corr_matrix.columns else target_col
        if target_corr_col in corr_matrix.columns:
            top_vars = corr_matrix[target_corr_col].abs().nlargest(15).index.tolist()
            corr_matrix = corr_matrix.loc[top_vars, top_vars]
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Matrice di Correlazione (Top 15 variabili)', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ Salvato: correlation_heatmap.png')
    
    # Stampa top correlazioni con target
    target_corr_col = target_col + '_encoded' if target_col + '_encoded' in corr_matrix.columns else target_col
    if target_corr_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_corr_col].abs().sort_values(ascending=False)
        print(f'\nTop 10 correlazioni con {target_col}:')
        for var, corr in target_corrs.head(11).items():
            if var != target_corr_col:
                print(f'   {var}: {corr:.3f}')
    
    return corr_matrix

def analyze_feature_importance(df, target_col):
    """Analizza feature importance"""
    print('\n🏆 FEATURE IMPORTANCE')
    print('='*30)
    
    # Prepara dati
    X = df.copy()
    y = df[target_col]
    
    # Rimuovi target
    if target_col in X.columns:
        X = X.drop(target_col, axis=1)
    
    # Encoding variabili categoriche
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].nunique() < 50:  # Solo se non troppe categorie
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X = X.drop(col, axis=1)  # Rimuovi se troppe categorie
    
    # Rimuovi colonne non numeriche rimanenti
    X = X.select_dtypes(include=[np.number])
    
    # Gestisci valori mancanti
    X = X.fillna(X.median())
    
    # Encoding target
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
    else:
        y_encoded = y
    
    print(f'Features finali per analisi: {X.shape[1]}')
    
    if X.shape[1] == 0:
        print('❌ Nessuna feature numerica disponibile')
        return None
    
    # Split dati
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Random Forest Importance
    print('\n🌲 Random Forest Feature Importance:')
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Mutual Information
    print('\n🔗 Mutual Information:')
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # F-score
    print('\n📊 F-Score:')
    f_scores, _ = f_classif(X_train, y_train)
    f_importance = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    # Stampa risultati
    print('\nTop 10 Features (Random Forest):')
    for i, (_, row) in enumerate(rf_importance.head(10).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<20}: {row["importance"]:.4f}')
    
    print('\nTop 10 Features (Mutual Information):')
    for i, (_, row) in enumerate(mi_importance.head(10).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<20}: {row["mi_score"]:.4f}')
    
    # Visualizzazione
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Random Forest
    top_rf = rf_importance.head(10)
    axes[0,0].barh(range(len(top_rf)), top_rf['importance'], color='forestgreen')
    axes[0,0].set_yticks(range(len(top_rf)))
    axes[0,0].set_yticklabels(top_rf['feature'])
    axes[0,0].set_title('Random Forest Importance')
    
    # Mutual Information
    top_mi = mi_importance.head(10)
    axes[0,1].barh(range(len(top_mi)), top_mi['mi_score'], color='orange')
    axes[0,1].set_yticks(range(len(top_mi)))
    axes[0,1].set_yticklabels(top_mi['feature'])
    axes[0,1].set_title('Mutual Information')
    
    # F-score
    top_f = f_importance.head(10)
    axes[1,0].barh(range(len(top_f)), top_f['f_score'], color='purple')
    axes[1,0].set_yticks(range(len(top_f)))
    axes[1,0].set_yticklabels(top_f['feature'])
    axes[1,0].set_title('F-Score')
    
    # Ranking combinato
    combined = pd.DataFrame({
        'feature': X.columns,
        'rf_rank': rf_importance.reset_index().index + 1,
        'mi_rank': mi_importance.reset_index().index + 1,
        'f_rank': f_importance.reset_index().index + 1
    })
    combined['avg_rank'] = combined[['rf_rank', 'mi_rank', 'f_rank']].mean(axis=1)
    combined = combined.sort_values('avg_rank')
    
    top_combined = combined.head(10)
    axes[1,1].barh(range(len(top_combined)), 1/top_combined['avg_rank'], color='red')
    axes[1,1].set_yticks(range(len(top_combined)))
    axes[1,1].set_yticklabels(top_combined['feature'])
    axes[1,1].set_title('Combined Ranking')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\n✅ Salvato: feature_importance.png')
    
    print('\n🏆 Top 10 Features (Ranking Combinato):')
    for i, (_, row) in enumerate(top_combined.head(10).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<20}: Rank {row["avg_rank"]:.1f}')
    
    return {
        'rf_importance': rf_importance,
        'mi_importance': mi_importance,
        'f_importance': f_importance,
        'combined_ranking': combined
    }

def analyze_distributions(df):
    """Analizza distribuzioni delle variabili"""
    print('\n📈 DISTRIBUZIONI VARIABILI')
    print('='*40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f'Variabili numeriche: {len(numeric_cols)}')
    
    if len(numeric_cols) > 0:
        # Statistiche descrittive
        print('\nStatistiche descrittive:')
        stats = df[numeric_cols].describe()
        print(stats.round(2))
        
        # Visualizza prime 8 distribuzioni
        n_plots = min(8, len(numeric_cols))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:n_plots]):
            df[col].hist(bins=20, ax=axes[i], alpha=0.7, color='skyblue')
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequenza')
        
        # Nascondi assi vuoti
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('✅ Salvato: distributions.png')

def main():
    """Funzione principale"""
    print('🔬 ANALISI ESPLORATIVA VELOCE - DATASET RESPIRATORIO')
    print('='*60)
    
    # Carica dati
    df = load_data()
    if df is None:
        return
    
    # Analisi target
    target_col = analyze_target(df)
    
    # Analisi distribuzioni
    analyze_distributions(df)
    
    # Analisi correlazioni
    corr_matrix = analyze_correlations(df, target_col)
    
    # Feature importance
    importance_results = analyze_feature_importance(df, target_col)
    
    print('\n🎉 ANALISI COMPLETATA!')
    print('📁 File generati:')
    print('   - target_distribution.png')
    print('   - distributions.png')
    print('   - correlation_heatmap.png')
    print('   - feature_importance.png')
    
    return df, importance_results

if __name__ == '__main__':
    df, results = main()