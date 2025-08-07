#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALISI ESPLORATIVA DEL DATASET
Analisi completa della distribuzione dei dati, correlazioni e feature importance
per la predizione di asma e BPCO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurazione visualizzazioni
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_clean_data(file_path):
    """Carica e pulisce il dataset"""
    print('📊 CARICAMENTO E PULIZIA DATASET')
    print('='*50)
    
    # Prova a caricare il dataset con diversi metodi
    try:
        # Prova prima con openpyxl
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f'✅ Dataset caricato con openpyxl: {df.shape[0]} righe, {df.shape[1]} colonne')
    except PermissionError:
        print('❌ Errore di permessi. Il file potrebbe essere aperto in Excel.')
        print('💡 Suggerimento: Chiudi il file Excel e riprova.')
        # Prova con un nome alternativo
        import shutil
        import os
        temp_file = 'temp_dataset.xlsx'
        try:
            shutil.copy2(file_path, temp_file)
            df = pd.read_excel(temp_file, engine='openpyxl')
            os.remove(temp_file)
            print(f'✅ Dataset caricato tramite copia temporanea: {df.shape[0]} righe, {df.shape[1]} colonne')
        except Exception as e:
            print(f'❌ Impossibile caricare il dataset: {e}')
            return None
    except Exception as e:
        print(f'❌ Errore nel caricamento: {e}')
        return None
    
    # Pulisci nomi colonne
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
    
    # Informazioni base
    print(f'\n📋 INFORMAZIONI DATASET:')
    print(f'   Dimensioni: {df.shape}')
    print(f'   Valori mancanti totali: {df.isnull().sum().sum()}')
    print(f'   Tipi di dati: {df.dtypes.value_counts().to_dict()}')
    
    return df

def analyze_target_distribution(df, target_col):
    """Analizza la distribuzione della variabile target"""
    print(f'\n🎯 DISTRIBUZIONE VARIABILE TARGET: {target_col}')
    print('='*60)
    
    if target_col not in df.columns:
        print(f'❌ Colonna {target_col} non trovata!')
        print(f'Colonne disponibili: {list(df.columns)}')
        return None
    
    # Distribuzione classi
    target_dist = df[target_col].value_counts()
    target_perc = df[target_col].value_counts(normalize=True) * 100
    
    print('📊 Distribuzione classi:')
    for classe, count in target_dist.items():
        perc = target_perc[classe]
        print(f'   {classe}: {count} ({perc:.1f}%)')
    
    # Visualizzazione
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grafico a barre
    target_dist.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_title('Distribuzione Classi - Conteggi')
    ax1.set_xlabel('Classi')
    ax1.set_ylabel('Frequenza')
    ax1.tick_params(axis='x', rotation=45)
    
    # Grafico a torta
    ax2.pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribuzione Classi - Percentuali')
    
    plt.tight_layout()
    plt.savefig('distribuzione_target.png', dpi=300, bbox_inches='tight')
    print('✅ Grafico salvato: distribuzione_target.png')
    plt.show()
    
    return target_dist

def analyze_feature_distributions(df, target_col):
    """Analizza la distribuzione delle features"""
    print(f'\n📈 DISTRIBUZIONE FEATURES')
    print('='*40)
    
    # Separa variabili numeriche e categoriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f'📊 Variabili numeriche: {len(numeric_cols)}')
    print(f'📊 Variabili categoriche: {len(categorical_cols)}')
    
    # Analisi variabili numeriche
    if numeric_cols:
        print(f'\n🔢 STATISTICHE VARIABILI NUMERICHE:')
        stats = df[numeric_cols].describe()
        print(stats.round(2))
        
        # Visualizza distribuzioni numeriche
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:16]):  # Limita a 16 variabili
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='lightblue')
                axes[i].set_title(f'Distribuzione: {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequenza')
        
        # Nascondi assi vuoti
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distribuzioni_numeriche.png', dpi=300, bbox_inches='tight')
        print('✅ Grafico salvato: distribuzioni_numeriche.png')
        plt.show()
    
    # Analisi variabili categoriche
    if categorical_cols:
        print(f'\n📝 VARIABILI CATEGORICHE:')
        for col in categorical_cols[:10]:  # Limita a 10 variabili
            unique_vals = df[col].nunique()
            print(f'   {col}: {unique_vals} valori unici')
            if unique_vals <= 10:
                print(f'      Valori: {df[col].value_counts().head().to_dict()}')
    
    return numeric_cols, categorical_cols

def create_correlation_heatmap(df, numeric_cols, target_col):
    """Crea heatmap delle correlazioni"""
    print(f'\n🔥 MATRICE DI CORRELAZIONE')
    print('='*40)
    
    if not numeric_cols:
        print('❌ Nessuna variabile numerica trovata per la correlazione')
        return None
    
    # Prepara dati per correlazione
    corr_data = df[numeric_cols].copy()
    
    # Aggiungi target se numerico
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            # Encode target categorico
            le = LabelEncoder()
            corr_data[target_col] = le.fit_transform(df[target_col])
        else:
            corr_data[target_col] = df[target_col]
    
    # Calcola matrice di correlazione
    correlation_matrix = corr_data.corr()
    
    # Crea heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Matrice di Correlazione - Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print('✅ Heatmap salvata: correlation_heatmap.png')
    plt.show()
    
    # Trova correlazioni più forti con target
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
        print(f'\n🎯 TOP 10 CORRELAZIONI CON {target_col}:')
        for var, corr in target_corr.head(11).items():  # 11 perché include se stesso
            if var != target_col:
                print(f'   {var}: {corr:.3f}')
    
    return correlation_matrix

def analyze_feature_importance(df, target_col, numeric_cols, categorical_cols):
    """Analizza l'importanza delle features per la predizione"""
    print(f'\n🏆 ANALISI FEATURE IMPORTANCE')
    print('='*50)
    
    if target_col not in df.columns:
        print(f'❌ Target {target_col} non trovato')
        return None
    
    # Prepara i dati
    X = df.copy()
    y = df[target_col]
    
    # Rimuovi target dalle features
    if target_col in X.columns:
        X = X.drop(target_col, axis=1)
    
    # Encoding delle variabili categoriche
    le_dict = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    # Encoding del target
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        target_classes = le_target.classes_
    else:
        y_encoded = y
        target_classes = None
    
    # Gestisci valori mancanti
    X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
    
    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 1. Random Forest Feature Importance
    print('\n🌲 RANDOM FOREST FEATURE IMPORTANCE:')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('Top 15 features (Random Forest):')
    for i, (_, row) in enumerate(rf_importance.head(15).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<25}: {row["importance"]:.4f}')
    
    # 2. Mutual Information
    print('\n🔗 MUTUAL INFORMATION:')
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print('Top 15 features (Mutual Information):')
    for i, (_, row) in enumerate(mi_importance.head(15).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<25}: {row["mi_score"]:.4f}')
    
    # 3. Univariate Feature Selection (F-score)
    print('\n📊 UNIVARIATE F-SCORE:')
    f_scores, _ = f_classif(X_train, y_train)
    f_importance = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    print('Top 15 features (F-score):')
    for i, (_, row) in enumerate(f_importance.head(15).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<25}: {row["f_score"]:.2f}')
    
    # Visualizzazione Feature Importance
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Random Forest
    top_rf = rf_importance.head(15)
    axes[0,0].barh(range(len(top_rf)), top_rf['importance'], color='forestgreen', alpha=0.7)
    axes[0,0].set_yticks(range(len(top_rf)))
    axes[0,0].set_yticklabels(top_rf['feature'])
    axes[0,0].set_title('Random Forest Feature Importance')
    axes[0,0].set_xlabel('Importance')
    
    # Mutual Information
    top_mi = mi_importance.head(15)
    axes[0,1].barh(range(len(top_mi)), top_mi['mi_score'], color='orange', alpha=0.7)
    axes[0,1].set_yticks(range(len(top_mi)))
    axes[0,1].set_yticklabels(top_mi['feature'])
    axes[0,1].set_title('Mutual Information Scores')
    axes[0,1].set_xlabel('MI Score')
    
    # F-score
    top_f = f_importance.head(15)
    axes[1,0].barh(range(len(top_f)), top_f['f_score'], color='purple', alpha=0.7)
    axes[1,0].set_yticks(range(len(top_f)))
    axes[1,0].set_yticklabels(top_f['feature'])
    axes[1,0].set_title('Univariate F-Score')
    axes[1,0].set_xlabel('F-Score')
    
    # Combinazione dei ranking
    combined_ranking = pd.DataFrame({
        'feature': X.columns,
        'rf_rank': rf_importance.reset_index().index + 1,
        'mi_rank': mi_importance.reset_index().index + 1,
        'f_rank': f_importance.reset_index().index + 1
    })
    combined_ranking['avg_rank'] = combined_ranking[['rf_rank', 'mi_rank', 'f_rank']].mean(axis=1)
    combined_ranking = combined_ranking.sort_values('avg_rank')
    
    top_combined = combined_ranking.head(15)
    axes[1,1].barh(range(len(top_combined)), 1/top_combined['avg_rank'], color='red', alpha=0.7)
    axes[1,1].set_yticks(range(len(top_combined)))
    axes[1,1].set_yticklabels(top_combined['feature'])
    axes[1,1].set_title('Combined Ranking (1/avg_rank)')
    axes[1,1].set_xlabel('Score (1/avg_rank)')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print('\n✅ Grafico salvato: feature_importance_analysis.png')
    plt.show()
    
    # Stampa ranking combinato
    print('\n🏆 TOP 15 FEATURES (RANKING COMBINATO):')
    for i, (_, row) in enumerate(top_combined.head(15).iterrows()):
        print(f'   {i+1:2d}. {row["feature"]:<25}: Rank medio {row["avg_rank"]:.1f}')
    
    return {
        'rf_importance': rf_importance,
        'mi_importance': mi_importance,
        'f_importance': f_importance,
        'combined_ranking': combined_ranking,
        'target_classes': target_classes
    }

def analyze_class_specific_features(df, target_col, feature_importance_results):
    """Analizza features specifiche per asma e BPCO"""
    print(f'\n🫁 ANALISI SPECIFICA ASMA vs BPCO')
    print('='*50)
    
    if target_col not in df.columns:
        return None
    
    # Filtra solo asma e BPCO
    asma_bpco_data = df[df[target_col].isin(['Asma bronchiale', 'COPD'])].copy()
    
    if asma_bpco_data.empty:
        print('❌ Nessun dato trovato per Asma bronchiale o COPD')
        print(f'Classi disponibili: {df[target_col].unique()}')
        return None
    
    print(f'📊 Dati Asma vs BPCO: {asma_bpco_data.shape[0]} campioni')
    print(f'   Asma bronchiale: {(asma_bpco_data[target_col] == "Asma bronchiale").sum()}')
    print(f'   COPD: {(asma_bpco_data[target_col] == "COPD").sum()}')
    
    # Analizza differenze nelle top features
    top_features = feature_importance_results['combined_ranking'].head(10)['feature'].tolist()
    
    numeric_features = [f for f in top_features if f in df.select_dtypes(include=[np.number]).columns]
    
    if numeric_features:
        # Confronto statistico
        print(f'\n📈 CONFRONTO STATISTICO TOP FEATURES:')
        asma_data = asma_bpco_data[asma_bpco_data[target_col] == 'Asma bronchiale']
        copd_data = asma_bpco_data[asma_bpco_data[target_col] == 'COPD']
        
        comparison_stats = []
        for feature in numeric_features[:8]:  # Limita a 8 features
            if feature in asma_data.columns and feature in copd_data.columns:
                asma_mean = asma_data[feature].mean()
                copd_mean = copd_data[feature].mean()
                asma_std = asma_data[feature].std()
                copd_std = copd_data[feature].std()
                
                print(f'\n   {feature}:')
                print(f'      Asma: {asma_mean:.3f} ± {asma_std:.3f}')
                print(f'      COPD: {copd_mean:.3f} ± {copd_std:.3f}')
                print(f'      Differenza: {abs(asma_mean - copd_mean):.3f}')
                
                comparison_stats.append({
                    'feature': feature,
                    'asma_mean': asma_mean,
                    'copd_mean': copd_mean,
                    'difference': abs(asma_mean - copd_mean)
                })
        
        # Visualizzazione confronto
        if comparison_stats:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, stat in enumerate(comparison_stats[:4]):
                feature = stat['feature']
                
                # Box plot
                asma_bpco_data.boxplot(column=feature, by=target_col, ax=axes[i])
                axes[i].set_title(f'Distribuzione: {feature}')
                axes[i].set_xlabel('Diagnosi')
                axes[i].set_ylabel(feature)
            
            plt.suptitle('Confronto Asma vs COPD - Top Features', fontsize=16)
            plt.tight_layout()
            plt.savefig('asma_vs_copd_comparison.png', dpi=300, bbox_inches='tight')
            print('\n✅ Grafico salvato: asma_vs_copd_comparison.png')
            plt.show()
    
    return asma_bpco_data

def main():
    """Funzione principale"""
    print('🔬 ANALISI ESPLORATIVA DATASET MALATTIE RESPIRATORIE')
    print('='*70)
    
    # Carica dataset
    df = load_and_clean_data('COLD 30.07.2025.xlsx')
    
    if df is None:
        print('\n❌ Impossibile procedere senza dataset. Uscita.')
        return None, None
    
    # Identifica colonna target (prova nomi comuni)
    possible_targets = ['Diagnosi', 'diagnosis', 'target', 'class', 'label', 'Diagnosi_finale']
    target_col = None
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print('\n❓ Colonna target non identificata automaticamente.')
        print('Colonne disponibili:')
        for i, col in enumerate(df.columns):
            print(f'   {i+1:2d}. {col}')
        
        # Usa la prima colonna che sembra categorica
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            target_col = categorical_cols[0]
            print(f'\n🎯 Usando come target: {target_col}')
    
    # Analisi distribuzione target
    target_dist = analyze_target_distribution(df, target_col)
    
    # Analisi distribuzione features
    numeric_cols, categorical_cols = analyze_feature_distributions(df, target_col)
    
    # Matrice di correlazione
    correlation_matrix = create_correlation_heatmap(df, numeric_cols, target_col)
    
    # Feature importance
    importance_results = analyze_feature_importance(df, target_col, numeric_cols, categorical_cols)
    
    # Analisi specifica Asma vs BPCO
    if importance_results:
        analyze_class_specific_features(df, target_col, importance_results)
    
    print('\n🎉 ANALISI COMPLETATA!')
    print('📁 File generati:')
    print('   - distribuzione_target.png')
    print('   - distribuzioni_numeriche.png')
    print('   - correlation_heatmap.png')
    print('   - feature_importance_analysis.png')
    print('   - asma_vs_copd_comparison.png')
    
    return df, importance_results

if __name__ == '__main__':
    df, results = main()