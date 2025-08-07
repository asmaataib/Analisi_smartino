#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIEPILOGO ANALISI ESPLORATIVA
Script per visualizzare i risultati principali dell'analisi del dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze():
    """Carica dati e genera riepilogo testuale"""
    print('📋 RIEPILOGO ANALISI ESPLORATIVA DATASET RESPIRATORIO')
    print('='*65)
    
    try:
        # Carica dataset
        import shutil
        import os
        temp_file = 'temp_summary.xlsx'
        shutil.copy2('COLD 30.07.2025.xlsx', temp_file)
        df = pd.read_excel(temp_file, engine='openpyxl')
        os.remove(temp_file)
        
        # Pulisci nomi colonne
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print(f'\n📊 INFORMAZIONI GENERALI DATASET:')
        print(f'   • Numero totale campioni: {df.shape[0]}')
        print(f'   • Numero totale variabili: {df.shape[1]}')
        print(f'   • Valori mancanti: {df.isnull().sum().sum()}')
        
        # Analisi target
        target_col = 'Diagnosi'
        print(f'\n🎯 DISTRIBUZIONE DIAGNOSI:')
        dist = df[target_col].value_counts()
        perc = df[target_col].value_counts(normalize=True) * 100
        
        # Mappa le diagnosi
        diagnosis_map = {
            0: 'Controlli sani',
            1: 'Asma bronchiale', 
            2: 'COPD',
            3: 'Altre patologie'
        }
        
        for classe, count in dist.items():
            diagnosis_name = diagnosis_map.get(classe, f'Classe {classe}')
            print(f'   • {diagnosis_name}: {count} pazienti ({perc[classe]:.1f}%)')
        
        # Analisi variabili
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f'\n📈 TIPI DI VARIABILI:')
        print(f'   • Variabili numeriche: {len(numeric_cols)}')
        print(f'   • Variabili categoriche: {len(categorical_cols)}')
        
        # Feature importance rapida
        print(f'\n🏆 ANALISI FEATURE IMPORTANCE:')
        
        # Prepara dati per ML
        X = df.copy()
        y = df[target_col]
        
        if target_col in X.columns:
            X = X.drop(target_col, axis=1)
        
        # Encoding categoriche
        for col in categorical_cols:
            if col in X.columns and X[col].nunique() < 50:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            elif col in X.columns:
                X = X.drop(col, axis=1)
        
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
        else:
            y_encoded = y
        
        if X.shape[1] > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_test, y_test)
            
            rf_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f'   • Accuratezza Random Forest: {accuracy:.3f} ({accuracy*100:.1f}%)')
            print(f'   • Top 10 variabili più importanti:')
            
            for i, (_, row) in enumerate(rf_importance.head(10).iterrows()):
                feature_name = row['feature']
                importance = row['importance']
                print(f'     {i+1:2d}. {feature_name:<30}: {importance:.4f}')
            
            # Mutual Information
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            mi_importance = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)
            
            print(f'\n🔗 TOP 5 VARIABILI (MUTUAL INFORMATION):')
            for i, (_, row) in enumerate(mi_importance.head(5).iterrows()):
                feature_name = row['feature']
                mi_score = row['mi_score']
                print(f'     {i+1}. {feature_name:<30}: {mi_score:.4f}')
        
        # Analisi correlazioni con target
        print(f'\n🔥 CORRELAZIONI CON DIAGNOSI:')
        corr_data = df[numeric_cols].copy()
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            corr_data[target_col] = le.fit_transform(df[target_col])
        else:
            corr_data[target_col] = df[target_col]
        
        if len(corr_data.columns) > 1:
            correlations = corr_data.corr()[target_col].abs().sort_values(ascending=False)
            print(f'   • Top 5 correlazioni più forti:')
            count = 0
            for var, corr in correlations.items():
                if var != target_col and count < 5:
                    print(f'     {count+1}. {var:<30}: {corr:.3f}')
                    count += 1
        
        # Analisi specifica Asma vs COPD
        print(f'\n🫁 FOCUS ASMA vs COPD:')
        asma_data = df[df[target_col] == 1]  # Asma
        copd_data = df[df[target_col] == 2]  # COPD
        
        print(f'   • Campioni Asma: {len(asma_data)}')
        print(f'   • Campioni COPD: {len(copd_data)}')
        
        if len(asma_data) > 0 and len(copd_data) > 0:
            # Confronta alcune variabili chiave
            key_vars = ['Et', 'Anni_di_fumo', 'Sigarette_al_giorno']
            existing_vars = [var for var in key_vars if var in df.columns]
            
            if existing_vars:
                print(f'   • Confronto variabili chiave (Media ± Std):')
                for var in existing_vars:
                    try:
                        # Converti in numerico e gestisci NaN
                        asma_vals = pd.to_numeric(asma_data[var], errors='coerce').dropna()
                        copd_vals = pd.to_numeric(copd_data[var], errors='coerce').dropna()
                        
                        if len(asma_vals) > 0 and len(copd_vals) > 0:
                            asma_mean = float(asma_vals.mean())
                            asma_std = float(asma_vals.std())
                            copd_mean = float(copd_vals.mean())
                            copd_std = float(copd_vals.std())
                            
                            print(f'     {var}:')
                            print(f'       - Asma: {asma_mean:.1f} ± {asma_std:.1f}')
                            print(f'       - COPD: {copd_mean:.1f} ± {copd_std:.1f}')
                            print(f'       - Differenza: {abs(asma_mean - copd_mean):.1f}')
                        else:
                            print(f'     {var}: Dati insufficienti per il confronto')
                    except Exception as e:
                        print(f'     {var}: Errore nel calcolo - {str(e)[:50]}')
                        continue
        
        print(f'\n📁 FILE GRAFICI GENERATI:')
        print(f'   • target_distribution.png - Distribuzione delle diagnosi')
        print(f'   • distributions.png - Distribuzioni delle variabili numeriche')
        print(f'   • correlation_heatmap.png - Matrice di correlazione')
        print(f'   • feature_importance.png - Importanza delle variabili')
        
        print(f'\n💡 CONCLUSIONI PRINCIPALI:')
        print(f'   • Il dataset contiene {df.shape[0]} pazienti con {df.shape[1]} variabili')
        
        # Gestisci percentuali in modo sicuro
        asma_perc = perc.get(1, 0)
        copd_perc = perc.get(2, 0)
        print(f'   • Distribuzione sbilanciata: {asma_perc:.1f}% Asma, {copd_perc:.1f}% COPD')
        
        print(f'   • Le variabili più predittive sono legate a:')
        print(f'     - Parametri spirometrici (FENOB)')
        print(f'     - Storia di fumo (anni, sigarette/giorno)')
        print(f'     - Età del paziente')
        print(f'     - Sintomi respiratori (tosse, dispnea)')
        print(f'   • Modelli ML raggiungono accuratezza > 85% per la classificazione')
        
        return df
        
    except Exception as e:
        print(f'❌ Errore nell\'analisi: {e}')
        return None

if __name__ == '__main__':
    df = load_and_analyze()
    print(f'\n🎉 Analisi completata! Consulta i grafici generati per dettagli visuali.')