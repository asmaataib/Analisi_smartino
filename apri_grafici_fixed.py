#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per aprire i grafici generati da advanced_ml_models_fixed.py
"""

import os
import platform

def open_file(filepath):
    """Apre un file con l'applicazione predefinita del sistema"""
    if os.path.exists(filepath):
        print(f"📊 Aprendo: {filepath}")
        try:
            if platform.system() == 'Windows':
                os.startfile(filepath)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{filepath}"')
            else:  # Linux
                os.system(f'xdg-open "{filepath}"')
            return True
        except Exception as e:
            print(f"❌ Errore nell'apertura di {filepath}: {e}")
            return False
    else:
        print(f"❌ File non trovato: {filepath}")
        return False

print("🎯 APERTURA GRAFICI ADVANCED ML MODELS (FIXED)")
print("=" * 50)

# Lista dei grafici da aprire
grafici = [
    'Confusion_matrix_fixed.png',
    'Feature_importance_best_model_fixed.png'
]

# Apri ogni grafico
for grafico in grafici:
    if os.path.exists(grafico):
        size = os.path.getsize(grafico)
        print(f"✅ {grafico} ({size} bytes)")
        open_file(grafico)
    else:
        print(f"❌ {grafico} - File non trovato")

print("\n🎉 Grafici aperti! Chiudi manualmente le finestre quando hai finito.")
print("📝 Descrizione grafici:")
print("   • Confusion_matrix_fixed.png: Matrice di confusione del modello migliore")
print("   • Feature_importance_best_model_fixed.png: Importanza delle variabili")