import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

# Lista dei grafici generati
grafici = [
    'target_distribution.png',
    'distribuzione_target.png', 
    'distributions.png',
    'correlation_heatmap.png',
    'feature_importance.png'
]

# Verifica quali grafici esistono
grafici_esistenti = []
for grafico in grafici:
    if os.path.exists(grafico):
        grafici_esistenti.append(grafico)
        print(f"✅ Trovato: {grafico}")
    else:
        print(f"❌ Non trovato: {grafico}")

print(f"\n📊 Visualizzazione di {len(grafici_esistenti)} grafici...\n")

# Crea una figura con subplot per tutti i grafici
if grafici_esistenti:
    n_grafici = len(grafici_esistenti)
    
    # Calcola il layout ottimale (righe x colonne)
    if n_grafici <= 2:
        rows, cols = 1, n_grafici
        figsize = (12, 6)
    elif n_grafici <= 4:
        rows, cols = 2, 2
        figsize = (15, 12)
    else:
        rows, cols = 3, 2
        figsize = (15, 18)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Se c'è solo un grafico, axes non è un array
    if n_grafici == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Visualizza ogni grafico
    for i, grafico in enumerate(grafici_esistenti):
        try:
            img = mpimg.imread(grafico)
            axes[i].imshow(img)
            axes[i].set_title(grafico.replace('.png', '').replace('_', ' ').title(), 
                            fontsize=12, fontweight='bold')
            axes[i].axis('off')
            print(f"📈 Caricato: {grafico}")
        except Exception as e:
            print(f"❌ Errore nel caricare {grafico}: {e}")
            axes[i].text(0.5, 0.5, f'Errore\n{grafico}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Nascondi gli assi non utilizzati
    for i in range(len(grafici_esistenti), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('📊 ANALISI ESPLORATIVA - GRAFICI GENERATI', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Salva la visualizzazione combinata
    plt.savefig('tutti_i_grafici.png', dpi=300, bbox_inches='tight')
    print("\n💾 Salvato: tutti_i_grafici.png")
    
    # Mostra i grafici
    plt.show()
    
    print("\n🎉 Visualizzazione completata!")
    print("\n📁 File grafici disponibili:")
    for grafico in grafici_esistenti:
        size = os.path.getsize(grafico) / 1024  # KB
        print(f"   • {grafico} ({size:.1f} KB)")
else:
    print("❌ Nessun grafico trovato!")
    print("\n🔄 Esegui prima 'analisi_esplorativa_veloce.py' per generare i grafici.")