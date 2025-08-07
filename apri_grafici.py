import os
import subprocess
import sys
from pathlib import Path

# Lista dei grafici generati
grafici = [
    'target_distribution.png',
    'distribuzione_target.png', 
    'distributions.png',
    'correlation_heatmap.png',
    'feature_importance.png',
    'tutti_i_grafici.png'  # Il file combinato appena creato
]

print("🔍 Verifica grafici disponibili...\n")

# Verifica quali grafici esistono
grafici_esistenti = []
for grafico in grafici:
    if os.path.exists(grafico):
        grafici_esistenti.append(grafico)
        size = os.path.getsize(grafico) / 1024  # KB
        print(f"✅ {grafico} ({size:.1f} KB)")
    else:
        print(f"❌ {grafico} - Non trovato")

if grafici_esistenti:
    print(f"\n📊 Trovati {len(grafici_esistenti)} grafici!")
    print("\n🖼️  Apertura grafici con il visualizzatore predefinito...\n")
    
    # Apri ogni grafico con il programma predefinito del sistema
    for i, grafico in enumerate(grafici_esistenti, 1):
        try:
            print(f"{i}. Apertura {grafico}...")
            
            # Su Windows usa 'start' per aprire con il programma predefinito
            if sys.platform.startswith('win'):
                os.startfile(grafico)
            elif sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', grafico])
            else:  # Linux
                subprocess.run(['xdg-open', grafico])
                
        except Exception as e:
            print(f"❌ Errore nell'aprire {grafico}: {e}")
    
    print("\n🎉 Tutti i grafici sono stati aperti!")
    print("\n📋 Descrizione dei grafici:")
    print("   • target_distribution.png - Distribuzione delle diagnosi")
    print("   • distributions.png - Distribuzioni delle variabili numeriche")
    print("   • correlation_heatmap.png - Matrice di correlazione")
    print("   • feature_importance.png - Importanza delle variabili")
    print("   • tutti_i_grafici.png - Visualizzazione combinata")
    
else:
    print("\n❌ Nessun grafico trovato!")
    print("\n🔄 Per generare i grafici, esegui:")
    print("   python analisi_esplorativa_veloce.py")

print("\n✨ Script completato!")