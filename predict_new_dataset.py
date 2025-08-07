#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTION SCRIPT FOR NEW DATASET
Applica il modello addestrato a un nuovo dataset
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Mapping delle diagnosi
diagnosis_mapping = {
    0: 'Altro',
    1: 'Asma bronchiale',
    2: 'BPCO', 
    3: 'Overlap asma/bpco'
}

# ============================================================================
# FUNZIONI DI PREPROCESSING
# ============================================================================

def clean_column_names(df):
    """Pulisce i nomi delle colonne per compatibilità ML"""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)
    df_clean.columns = df_clean.columns.str.strip('_')
    return df_clean

def preprocess_new_dataset(new_df, label_encoders, feature_columns):
    """
    Preprocessa un nuovo dataset usando gli encoders del training
    
    Parameters:
    new_df: DataFrame con i nuovi dati
    label_encoders: dict con gli encoders addestrati
    feature_columns: list delle colonne feature del training
    
    Returns:
    DataFrame preprocessato
    """
    
    print(f"📊 Nuovo dataset: {new_df.shape[0]} pazienti, {new_df.shape[1]} variabili")
    
    # Pulisci i nomi delle colonne
    df_processed = clean_column_names(new_df)
    
    # Rimuovi colonne non necessarie
    columns_to_remove = ['Data_questionario', 'Identificativo']
    for col in columns_to_remove:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
            print(f"✅ Rimossa colonna: {col}")
    
    # Encode variabili categoriche usando gli encoders del training
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in label_encoders:
            try:
                # Gestisci valori non visti durante il training
                unique_values = df_processed[col].unique()
                known_values = label_encoders[col].classes_
                
                # Sostituisci valori sconosciuti con il più frequente del training
                unknown_values = set(unique_values) - set(known_values)
                if unknown_values:
                    most_frequent = known_values[0]  # Primo valore delle classi
                    print(f"⚠️  Colonna {col}: sostituiti {len(unknown_values)} valori sconosciuti con '{most_frequent}'")
                    df_processed[col] = df_processed[col].replace(list(unknown_values), most_frequent)
                
                # Applica l'encoder
                df_processed[col] = label_encoders[col].transform(df_processed[col].astype(str))
                print(f"✅ Encoded colonna categorica: {col}")
                
            except Exception as e:
                print(f"❌ Errore nell'encoding di {col}: {e}")
                # Rimuovi la colonna se non può essere processata
                df_processed = df_processed.drop(col, axis=1)
    
    # Assicurati che tutte le feature del training siano presenti
    missing_features = set(feature_columns) - set(df_processed.columns)
    if missing_features:
        print(f"⚠️  Feature mancanti nel nuovo dataset: {missing_features}")
        # Aggiungi colonne mancanti con valore 0
        for feature in missing_features:
            df_processed[feature] = 0
            print(f"➕ Aggiunta feature mancante: {feature} (valore=0)")
    
    # Riordina le colonne per matchare il training
    df_processed = df_processed[feature_columns]
    
    print(f"✅ Preprocessing completato: {df_processed.shape}")
    return df_processed

# ============================================================================
# FUNZIONE PRINCIPALE DI PREDIZIONE
# ============================================================================

def predict_new_dataset(new_dataset_path, model=None, scaler=None, label_encoders=None, feature_columns=None):
    """
    Predice le diagnosi per un nuovo dataset
    
    Parameters:
    new_dataset_path: path del file Excel/CSV del nuovo dataset
    model: modello addestrato (se None, usa RandomForest di default)
    scaler: scaler addestrato (se None, ne crea uno nuovo)
    label_encoders: encoders addestrati (se None, ne crea di nuovi)
    feature_columns: colonne feature del training (se None, usa tutte)
    
    Returns:
    DataFrame con predizioni e probabilità
    """
    
    print('🔮 PREDIZIONE SU NUOVO DATASET')
    print('='*50)
    
    # Carica il nuovo dataset
    try:
        if new_dataset_path.endswith('.xlsx'):
            new_df = pd.read_excel(new_dataset_path)
        elif new_dataset_path.endswith('.csv'):
            new_df = pd.read_csv(new_dataset_path)
        else:
            raise ValueError("Formato file non supportato. Usa .xlsx o .csv")
        
        print(f"📁 Dataset caricato: {new_dataset_path}")
        
    except Exception as e:
        print(f"❌ Errore nel caricamento del dataset: {e}")
        return None
    
    # Se non sono forniti i parametri del training, addestra un modello veloce
    if model is None or scaler is None or label_encoders is None:
        print("⚠️  Parametri di training non forniti. Addestramento rapido su dataset originale...")
        
        # Carica dataset originale per training rapido
        try:
            original_df = pd.read_excel('COLD 30.07.2025.xlsx')
            model, scaler, label_encoders, feature_columns = quick_train_model(original_df)
        except Exception as e:
            print(f"❌ Errore nel training rapido: {e}")
            return None
    
    # Preprocessa il nuovo dataset
    try:
        X_new = preprocess_new_dataset(new_df, label_encoders, feature_columns)
    except Exception as e:
        print(f"❌ Errore nel preprocessing: {e}")
        return None
    
    # Scala le feature se necessario
    if scaler is not None:
        try:
            X_new_scaled = scaler.transform(X_new)
            print("✅ Feature scalate")
        except Exception as e:
            print(f"⚠️  Errore nello scaling: {e}. Uso dati non scalati.")
            X_new_scaled = X_new
    else:
        X_new_scaled = X_new
    
    # Fai le predizioni
    try:
        predictions = model.predict(X_new_scaled)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new_scaled)
            max_probs = np.max(probabilities, axis=1)
        else:
            probabilities = None
            max_probs = None
        
        print(f"✅ Predizioni completate per {len(predictions)} pazienti")
        
    except Exception as e:
        print(f"❌ Errore nelle predizioni: {e}")
        return None
    
    # Crea DataFrame con risultati
    results_df = new_df.copy()
    results_df['Predicted_Class'] = predictions
    results_df['Predicted_Diagnosis'] = [diagnosis_mapping[pred] for pred in predictions]
    
    if max_probs is not None:
        results_df['Confidence'] = max_probs
        results_df['Confidence_Percent'] = (max_probs * 100).round(1)
    
    # Aggiungi probabilità per ogni classe se disponibili
    if probabilities is not None:
        for i, diagnosis in diagnosis_mapping.items():
            results_df[f'Prob_{diagnosis.replace(" ", "_")}'] = probabilities[:, i]
    
    # Statistiche delle predizioni
    print('\n📊 STATISTICHE PREDIZIONI:')
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for class_id, count in pred_counts.items():
        diagnosis = diagnosis_mapping[class_id]
        percentage = (count / len(predictions)) * 100
        print(f"   {diagnosis}: {count} pazienti ({percentage:.1f}%)")
    
    if max_probs is not None:
        print(f"\n🎯 CONFIDENZA MEDIA: {np.mean(max_probs):.3f} ({np.mean(max_probs)*100:.1f}%)")
        print(f"   Range confidenza: [{np.min(max_probs):.3f}, {np.max(max_probs):.3f}]")
    
    return results_df

def quick_train_model(df):
    """
    Addestra rapidamente un modello RandomForest per le predizioni
    """
    from sklearn.model_selection import train_test_split
    
    print("🚀 Training rapido del modello...")
    
    # Preprocessing
    df_ml = clean_column_names(df.copy())
    
    # Encode categorical variables
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col != 'Data_questionario':
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
    
    # Remove unnecessary columns
    if 'Data_questionario' in df_ml.columns:
        df_ml = df_ml.drop('Data_questionario', axis=1)
    
    # Features and target
    X = df_ml.drop(['Diagnosi', 'Identificativo'], axis=1)
    y = df_ml['Diagnosi']
    feature_columns = X.columns.tolist()
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Train scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print("✅ Modello addestrato")
    
    return model, scaler, label_encoders, feature_columns

# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    print("📋 ESEMPIO DI UTILIZZO:")
    print("")
    print("# 1. Predizione su nuovo dataset (training automatico)")
    print("results = predict_new_dataset('nuovo_dataset.xlsx')")
    print("")
    print("# 2. Salva i risultati")
    print("results.to_excel('predizioni_risultati.xlsx', index=False)")
    print("")
    print("# 3. Visualizza statistiche")
    print("print(results['Predicted_Diagnosis'].value_counts())")
    print("print(f'Confidenza media: {results[\"Confidence\"].mean():.3f}')")
    print("")
    print("🔧 Per usare questo script:")
    print("   1. Assicurati che il nuovo dataset abbia le stesse colonne del training")
    print("   2. Chiama predict_new_dataset('path_del_tuo_file.xlsx')")
    print("   3. I risultati includeranno predizioni e probabilità")
    print("")
    print("💡 NOTA: Il nuovo dataset deve avere colonne simili al dataset originale")
    print("   per ottenere predizioni accurate.")