#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESEMPIO PRATICO - PREDIZIONE SU NUOVO DATASET
Script di esempio per utilizzare il modello addestrato
"""

import pandas as pd
import numpy as np
from predict_new_dataset import predict_new_dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ESEMPIO 1: PREDIZIONE SEMPLICE
# ============================================================================

def esempio_predizione_semplice():
    """
    Esempio base di predizione su nuovo dataset
    """
    print("🔮 ESEMPIO 1: PREDIZIONE SEMPLICE")
    print("="*50)
    
    # Sostituisci con il path del tuo nuovo dataset
    nuovo_dataset_path = "nuovo_dataset.xlsx"  # CAMBIA QUESTO PATH
    
    try:
        # Fai le predizioni
        risultati = predict_new_dataset(nuovo_dataset_path)
        
        if risultati is not None:
            print("\n✅ PREDIZIONI COMPLETATE!")
            
            # Mostra statistiche
            print("\n📊 DISTRIBUZIONE PREDIZIONI:")
            print(risultati['Predicted_Diagnosis'].value_counts())
            
            # Mostra confidenza media
            if 'Confidence' in risultati.columns:
                confidenza_media = risultati['Confidence'].mean()
                print(f"\n🎯 Confidenza media: {confidenza_media:.3f} ({confidenza_media*100:.1f}%)")
            
            # Salva i risultati
            output_file = "predizioni_risultati.xlsx"
            risultati.to_excel(output_file, index=False)
            print(f"\n💾 Risultati salvati in: {output_file}")
            
            # Mostra primi 5 risultati
            print("\n👀 PRIMI 5 RISULTATI:")
            cols_to_show = ['Predicted_Diagnosis', 'Confidence_Percent']
            if all(col in risultati.columns for col in cols_to_show):
                print(risultati[cols_to_show].head())
            
            return risultati
        
    except FileNotFoundError:
        print(f"❌ File non trovato: {nuovo_dataset_path}")
        print("💡 Assicurati che il file esista e il path sia corretto")
        return None
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

# ============================================================================
# ESEMPIO 2: ANALISI DETTAGLIATA DEI RISULTATI
# ============================================================================

def analizza_risultati_dettagliati(risultati_df):
    """
    Analisi dettagliata dei risultati delle predizioni
    """
    print("\n🔍 ESEMPIO 2: ANALISI DETTAGLIATA")
    print("="*50)
    
    if risultati_df is None:
        print("❌ Nessun risultato da analizzare")
        return
    
    # Statistiche generali
    print(f"📊 STATISTICHE GENERALI:")
    print(f"   Totale pazienti: {len(risultati_df)}")
    
    # Distribuzione per diagnosi
    print("\n🏥 DISTRIBUZIONE DIAGNOSI:")
    distr = risultati_df['Predicted_Diagnosis'].value_counts()
    for diagnosi, count in distr.items():
        perc = (count / len(risultati_df)) * 100
        print(f"   {diagnosi}: {count} pazienti ({perc:.1f}%)")
    
    # Analisi confidenza
    if 'Confidence' in risultati_df.columns:
        print("\n🎯 ANALISI CONFIDENZA:")
        conf = risultati_df['Confidence']
        print(f"   Media: {conf.mean():.3f} ({conf.mean()*100:.1f}%)")
        print(f"   Mediana: {conf.median():.3f} ({conf.median()*100:.1f}%)")
        print(f"   Min: {conf.min():.3f} ({conf.min()*100:.1f}%)")
        print(f"   Max: {conf.max():.3f} ({conf.max()*100:.1f}%)")
        
        # Pazienti con bassa confidenza
        bassa_conf = risultati_df[conf < 0.6]
        if len(bassa_conf) > 0:
            print(f"\n⚠️  ATTENZIONE: {len(bassa_conf)} pazienti con confidenza < 60%")
            print("   Questi casi potrebbero richiedere revisione clinica")
    
    # Top 10 pazienti per confidenza
    if 'Confidence' in risultati_df.columns:
        print("\n🏆 TOP 10 PREDIZIONI PIÙ SICURE:")
        top_conf = risultati_df.nlargest(10, 'Confidence')
        for idx, row in top_conf.iterrows():
            print(f"   Paziente {idx+1}: {row['Predicted_Diagnosis']} (confidenza: {row['Confidence_Percent']:.1f}%)")
    
    # Crea report dettagliato
    report_file = "report_predizioni_dettagliato.xlsx"
    with pd.ExcelWriter(report_file) as writer:
        # Foglio principale con tutti i risultati
        risultati_df.to_excel(writer, sheet_name='Tutti_Risultati', index=False)
        
        # Foglio con statistiche
        stats_df = pd.DataFrame({
            'Diagnosi': distr.index,
            'Numero_Pazienti': distr.values,
            'Percentuale': (distr.values / len(risultati_df) * 100).round(1)
        })
        stats_df.to_excel(writer, sheet_name='Statistiche', index=False)
        
        # Foglio con casi a bassa confidenza
        if 'Confidence' in risultati_df.columns:
            bassa_conf = risultati_df[risultati_df['Confidence'] < 0.6]
            if len(bassa_conf) > 0:
                bassa_conf.to_excel(writer, sheet_name='Bassa_Confidenza', index=False)
    
    print(f"\n📋 Report dettagliato salvato in: {report_file}")

# ============================================================================
# ESEMPIO 3: CONFRONTO CON DIAGNOSI REALI (se disponibili)
# ============================================================================

def confronta_con_diagnosi_reali(risultati_df, colonna_diagnosi_reale='Diagnosi_Reale'):
    """
    Confronta le predizioni con diagnosi reali se disponibili
    """
    print("\n⚖️  ESEMPIO 3: CONFRONTO CON DIAGNOSI REALI")
    print("="*50)
    
    if colonna_diagnosi_reale not in risultati_df.columns:
        print(f"❌ Colonna '{colonna_diagnosi_reale}' non trovata")
        print("💡 Se hai le diagnosi reali, rinomina la colonna o specifica il nome corretto")
        return
    
    # Calcola accuratezza
    diagnosi_reali = risultati_df[colonna_diagnosi_reale]
    predizioni = risultati_df['Predicted_Diagnosis']
    
    accuratezza = (diagnosi_reali == predizioni).mean()
    print(f"🎯 ACCURATEZZA COMPLESSIVA: {accuratezza:.3f} ({accuratezza*100:.1f}%)")
    
    # Matrice di confusione semplificata
    print("\n📊 MATRICE DI CONFUSIONE:")
    confusion_df = pd.crosstab(diagnosi_reali, predizioni, margins=True)
    print(confusion_df)
    
    # Accuratezza per classe
    print("\n🏥 ACCURATEZZA PER DIAGNOSI:")
    for diagnosi in diagnosi_reali.unique():
        if diagnosi != 'All':
            mask = diagnosi_reali == diagnosi
            acc_classe = (diagnosi_reali[mask] == predizioni[mask]).mean()
            n_casi = mask.sum()
            print(f"   {diagnosi}: {acc_classe:.3f} ({acc_classe*100:.1f}%) - {n_casi} casi")

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale che esegue tutti gli esempi
    """
    print("🚀 AVVIO ESEMPI DI PREDIZIONE")
    print("="*60)
    
    # IMPORTANTE: Modifica questo path con il tuo file
    nuovo_dataset_path = "nuovo_dataset.xlsx"  # 👈 CAMBIA QUESTO!
    
    print(f"📁 Dataset da analizzare: {nuovo_dataset_path}")
    print("\n💡 NOTA: Assicurati che il file esista e abbia colonne simili al dataset di training")
    
    # Esempio 1: Predizione semplice
    risultati = esempio_predizione_semplice()
    
    if risultati is not None:
        # Esempio 2: Analisi dettagliata
        analizza_risultati_dettagliati(risultati)
        
        # Esempio 3: Confronto (solo se hai diagnosi reali)
        # Decommentare se hai una colonna con diagnosi reali
        # confronta_con_diagnosi_reali(risultati, 'Nome_Colonna_Diagnosi_Reale')
        
        print("\n✅ TUTTI GLI ESEMPI COMPLETATI!")
        print("\n📋 FILE CREATI:")
        print("   - predizioni_risultati.xlsx (risultati principali)")
        print("   - report_predizioni_dettagliato.xlsx (analisi completa)")
        
    else:
        print("\n❌ Impossibile completare gli esempi")
        print("\n🔧 RISOLUZIONE PROBLEMI:")
        print("   1. Verifica che il file del nuovo dataset esista")
        print("   2. Controlla che il formato sia .xlsx o .csv")
        print("   3. Assicurati che le colonne siano simili al dataset originale")
        print("   4. Verifica che il dataset originale 'COLD 30.07.2025.xlsx' sia presente")

# ============================================================================
# ISTRUZIONI PER L'UTENTE
# ============================================================================

if __name__ == "__main__":
    print("📖 ISTRUZIONI PER L'USO:")
    print("="*50)
    print("")
    print("1️⃣  PREPARAZIONE:")
    print("   - Assicurati di avere il tuo nuovo dataset in formato .xlsx o .csv")
    print("   - Il dataset deve avere colonne simili a quello di training")
    print("   - Modifica la variabile 'nuovo_dataset_path' con il path corretto")
    print("")
    print("2️⃣  ESECUZIONE:")
    print("   - Esegui questo script: python esempio_predizione.py")
    print("   - Oppure importa le funzioni: from esempio_predizione import *")
    print("")
    print("3️⃣  RISULTATI:")
    print("   - predizioni_risultati.xlsx: risultati principali")
    print("   - report_predizioni_dettagliato.xlsx: analisi completa")
    print("")
    print("🚀 Vuoi eseguire gli esempi ora? Decommentare la riga seguente:")
    print("# main()")
    print("")
    print("💡 ESEMPIO RAPIDO:")
    print("")
    print("from predict_new_dataset import predict_new_dataset")
    print("risultati = predict_new_dataset('mio_dataset.xlsx')")
    print("risultati.to_excel('risultati.xlsx', index=False)")
    print("print(risultati['Predicted_Diagnosis'].value_counts())")