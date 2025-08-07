import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path
import time

# Configura matplotlib per mantenere le finestre aperte
plt.ion()  # Modalità interattiva

# Lista dei grafici generati
grafici = [
    'target_distribution.png',
    'distribuzione_target.png', 
    'distributions.png',
    'correlation_heatmap.png',
    'feature_importance.png'
]

# Titoli descrittivi per i grafici
titoli = {
    'target_distribution.png': '📊 Distribuzione delle Diagnosi',
    'distribuzione_target.png': '📈 Distribuzione Target (Alternativa)',
    'distributions.png': '📉 Distribuzioni Variabili Numeriche',
    'correlation_heatmap.png': '🔥 Heatmap delle Correlazioni',
    'feature_importance.png': '⭐ Importanza delle Variabili'
}

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
    print(f"\n📊 Apertura di {len(grafici_esistenti)} grafici in finestre separate...\n")
    
    figure_handles = []
    
    # Crea una finestra separata per ogni grafico
    for i, grafico in enumerate(grafici_esistenti):
        try:
            print(f"{i+1}. Caricamento {grafico}...")
            
            # Crea una nuova figura
            fig = plt.figure(figsize=(12, 8))
            figure_handles.append(fig)
            
            # Carica e mostra l'immagine
            img = mpimg.imread(grafico)
            plt.imshow(img)
            plt.title(titoli.get(grafico, grafico), fontsize=14, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Posiziona la finestra (opzionale)
            mngr = fig.canvas.manager
            if hasattr(mngr, 'window'):
                try:
                    # Posiziona le finestre in modo ordinato
                    x_offset = (i % 3) * 400
                    y_offset = (i // 3) * 300
                    mngr.window.wm_geometry(f"+{x_offset}+{y_offset}")
                except:
                    pass
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"❌ Errore nel caricare {grafico}: {e}")
    
    print(f"\n🎉 {len(figure_handles)} grafici aperti in finestre separate!")
    print("\n📋 Descrizione dei grafici:")
    for grafico in grafici_esistenti:
        print(f"   • {titoli.get(grafico, grafico)}")
    
    print("\n⚠️  IMPORTANTE:")
    print("   • Le finestre rimarranno aperte fino alla chiusura manuale")
    print("   • Chiudi questo script per chiudere tutte le finestre")
    print("   • Usa Ctrl+C nel terminale per terminare")
    
    # Mantieni le finestre aperte
    try:
        print("\n🔄 Mantenimento finestre attivo... (Premi Ctrl+C per uscire)")
        while True:
            plt.pause(1)  # Pausa di 1 secondo
            
            # Verifica se tutte le finestre sono ancora aperte
            open_figures = [fig for fig in figure_handles if plt.fignum_exists(fig.number)]
            if not open_figures:
                print("\n📝 Tutte le finestre sono state chiuse dall'utente.")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Interruzione da tastiera ricevuta.")
        print("🔒 Chiusura di tutte le finestre...")
        plt.close('all')
        
else:
    print("\n❌ Nessun grafico trovato!")
    print("\n🔄 Per generare i grafici, esegui:")
    print("   python analisi_esplorativa_veloce.py")

print("\n✨ Script completato!")