#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZZAZIONE RISULTATI OTTIMIZZAZIONE
Analisi dettagliata dei miglioramenti ottenuti
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurazione grafici
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print('📊 ANALISI RISULTATI OTTIMIZZAZIONE ML')
print('='*60)

# ============================================================================
# 1. CARICAMENTO E ANALISI RISULTATI
# ============================================================================

try:
    # Carica risultati ottimizzazione
    results_df = pd.read_excel('risultati_ottimizzazione.xlsx')
    print('✅ Risultati caricati con successo')
    
    print(f'\n📈 RISULTATI OTTIMIZZAZIONE ({len(results_df)} modelli testati)')
    print('='*70)
    
    # Mostra tabella completa
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(results_df.to_string(index=False, float_format='%.3f'))
    
except FileNotFoundError:
    print('❌ File risultati_ottimizzazione.xlsx non trovato')
    print('💡 Esegui prima optimize_ml_models.py')
    exit()

# ============================================================================
# 2. ANALISI STATISTICHE
# ============================================================================

print('\n\n📊 ANALISI STATISTICHE DETTAGLIATE')
print('='*60)

# Statistiche generali
accuracy_stats = results_df['Accuratezza_Media'].describe()
print('\n🎯 STATISTICHE ACCURATEZZA:')
print(f'   Media generale: {accuracy_stats["mean"]:.3f}')
print(f'   Mediana: {accuracy_stats["50%"]:.3f}')
print(f'   Migliore: {accuracy_stats["max"]:.3f}')
print(f'   Peggiore: {accuracy_stats["min"]:.3f}')
print(f'   Range: {accuracy_stats["max"] - accuracy_stats["min"]:.3f}')

# Modelli che superano baseline
baseline = 0.67
modelli_migliorati = results_df[results_df['Miglioramento_vs_Baseline'] > 0]
print(f'\n🚀 MODELLI CHE SUPERANO BASELINE ({baseline:.1%}):')
print(f'   Totali: {len(modelli_migliorati)}/{len(results_df)}')
print(f'   Percentuale: {len(modelli_migliorati)/len(results_df)*100:.1f}%')

if len(modelli_migliorati) > 0:
    miglioramento_medio = modelli_migliorati['Miglioramento_vs_Baseline'].mean()
    print(f'   Miglioramento medio: +{miglioramento_medio:.3f} ({miglioramento_medio*100:.1f}%)')
    
    print('\n🏆 TOP 3 MIGLIORAMENTI:')
    top_3 = modelli_migliorati.nlargest(3, 'Miglioramento_vs_Baseline')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f'   {i}. {row["Modello"]}: +{row["Miglioramento_vs_Baseline"]:.3f} ({row["Miglioramento_vs_Baseline"]*100:.1f}%)')

# ============================================================================
# 3. VISUALIZZAZIONI
# ============================================================================

print('\n\n📈 CREAZIONE VISUALIZZAZIONI')
print('='*40)

# Figura 1: Confronto Accuratezza
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analisi Risultati Ottimizzazione ML', fontsize=16, fontweight='bold')

# Grafico 1: Bar plot accuratezza
colors = ['#2E8B57' if x > 0 else '#DC143C' for x in results_df['Miglioramento_vs_Baseline']]
bars = ax1.barh(range(len(results_df)), results_df['Accuratezza_Media'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(results_df)))
ax1.set_yticklabels(results_df['Modello'], fontsize=9)
ax1.set_xlabel('Accuratezza')
ax1.set_title('Accuratezza per Modello')
ax1.axvline(x=baseline, color='red', linestyle='--', alpha=0.8, label=f'Baseline ({baseline:.1%})')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Aggiungi valori sulle barre
for i, (bar, acc) in enumerate(zip(bars, results_df['Accuratezza_Media'])):
    ax1.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=8)

# Grafico 2: Miglioramento vs Baseline
colors_improvement = ['#2E8B57' if x > 0 else '#DC143C' for x in results_df['Miglioramento_vs_Baseline']]
bars2 = ax2.barh(range(len(results_df)), results_df['Miglioramento_vs_Baseline'], color=colors_improvement, alpha=0.7)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['Modello'], fontsize=9)
ax2.set_xlabel('Miglioramento vs Baseline')
ax2.set_title('Miglioramento rispetto al Baseline (67%)')
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
ax2.grid(axis='x', alpha=0.3)

# Aggiungi valori sulle barre
for i, (bar, improvement) in enumerate(zip(bars2, results_df['Miglioramento_vs_Baseline'])):
    sign = '+' if improvement >= 0 else ''
    ax2.text(improvement + (0.005 if improvement >= 0 else -0.005), i, 
             f'{sign}{improvement:.3f}', va='center', fontsize=8,
             ha='left' if improvement >= 0 else 'right')

# Grafico 3: Accuratezza vs Deviazione Standard
scatter = ax3.scatter(results_df['Accuratezza_Media'], results_df['Deviazione_Standard'], 
                     c=results_df['Miglioramento_vs_Baseline'], cmap='RdYlGn', 
                     s=100, alpha=0.7, edgecolors='black')
ax3.set_xlabel('Accuratezza Media')
ax3.set_ylabel('Deviazione Standard')
ax3.set_title('Accuratezza vs Stabilità (Deviazione Standard)')
ax3.grid(alpha=0.3)

# Aggiungi etichette ai punti
for i, row in results_df.iterrows():
    ax3.annotate(row['Modello'], 
                (row['Accuratezza_Media'], row['Deviazione_Standard']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

# Colorbar per il scatter plot
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Miglioramento vs Baseline')

# Grafico 4: Distribuzione miglioramenti
miglioramenti = results_df['Miglioramento_vs_Baseline']
ax4.hist(miglioramenti, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Baseline')
ax4.axvline(x=miglioramenti.mean(), color='green', linestyle='-', alpha=0.8, label=f'Media ({miglioramenti.mean():.3f})')
ax4.set_xlabel('Miglioramento vs Baseline')
ax4.set_ylabel('Frequenza')
ax4.set_title('Distribuzione Miglioramenti')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analisi_risultati_ottimizzazione.png', dpi=300, bbox_inches='tight')
print('✅ Grafico principale salvato: analisi_risultati_ottimizzazione.png')
plt.show()

# ============================================================================
# 4. GRAFICO DETTAGLIATO MIGLIORI MODELLI
# ============================================================================

# Figura 2: Focus sui migliori modelli
top_5 = results_df.nlargest(5, 'Accuratezza_Media')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Focus sui Top 5 Modelli', fontsize=14, fontweight='bold')

# Grafico accuratezza con error bars
y_pos = np.arange(len(top_5))
accuracies = top_5['Accuratezza_Media']
errors = top_5['Deviazione_Standard']

bars = ax1.barh(y_pos, accuracies, xerr=errors, capsize=5, 
                color='lightgreen', alpha=0.7, edgecolor='darkgreen')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_5['Modello'])
ax1.set_xlabel('Accuratezza (con deviazione standard)')
ax1.set_title('Top 5 Modelli - Accuratezza')
ax1.axvline(x=baseline, color='red', linestyle='--', alpha=0.8, label=f'Baseline ({baseline:.1%})')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Aggiungi valori
for i, (acc, err) in enumerate(zip(accuracies, errors)):
    ax1.text(acc + err + 0.005, i, f'{acc:.3f}±{err:.3f}', va='center', fontsize=9)

# Grafico miglioramento percentuale
miglioramenti_perc = top_5['Miglioramento_vs_Baseline'] * 100
bars2 = ax2.barh(y_pos, miglioramenti_perc, color='gold', alpha=0.7, edgecolor='orange')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_5['Modello'])
ax2.set_xlabel('Miglioramento (%)')
ax2.set_title('Top 5 Modelli - Miglioramento Percentuale')
ax2.grid(axis='x', alpha=0.3)

# Aggiungi valori
for i, improvement in enumerate(miglioramenti_perc):
    ax2.text(improvement + 0.5, i, f'+{improvement:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('top_5_modelli_dettaglio.png', dpi=300, bbox_inches='tight')
print('✅ Grafico top 5 salvato: top_5_modelli_dettaglio.png')
plt.show()

# ============================================================================
# 5. REPORT FINALE
# ============================================================================

print('\n\n📋 REPORT FINALE OTTIMIZZAZIONE')
print('='*60)

# Migliore modello
best_model = results_df.iloc[0]
print(f'🥇 MIGLIORE MODELLO: {best_model["Modello"]}')
print(f'   Accuratezza: {best_model["Accuratezza_Media"]:.3f} ± {best_model["Deviazione_Standard"]:.3f}')
print(f'   Miglioramento: +{best_model["Miglioramento_vs_Baseline"]:.3f} ({best_model["Miglioramento_vs_Baseline"]*100:.1f}%)')

# Confronto con baseline
print(f'\n📊 CONFRONTO CON BASELINE:')
print(f'   Baseline originale: {baseline:.1%}')
print(f'   Nuovo best: {best_model["Accuratezza_Media"]:.1%}')
print(f'   Incremento assoluto: +{best_model["Miglioramento_vs_Baseline"]:.1%}')
print(f'   Incremento relativo: +{(best_model["Miglioramento_vs_Baseline"]/baseline)*100:.1f}%')

# Statistiche generali
print(f'\n🎯 STATISTICHE GENERALI:')
print(f'   Modelli testati: {len(results_df)}')
print(f'   Modelli migliorati: {len(modelli_migliorati)}')
print(f'   Tasso di successo: {len(modelli_migliorati)/len(results_df)*100:.1f}%')
print(f'   Miglioramento medio: +{results_df["Miglioramento_vs_Baseline"].mean():.3f}')

# Raccomandazioni
print(f'\n💡 RACCOMANDAZIONI:')
print(f'   1. 🏆 Usa {best_model["Modello"]} per produzione')
print(f'   2. 🔄 Considera ensemble dei top 3 modelli per robustezza')
print(f'   3. 📊 Valida su dataset esterno indipendente')
print(f'   4. 🔧 Continua ottimizzazione con più dati')

# Salva report testuale
with open('report_ottimizzazione.txt', 'w', encoding='utf-8') as f:
    f.write('REPORT OTTIMIZZAZIONE MODELLI ML\n')
    f.write('='*50 + '\n\n')
    f.write(f'Migliore modello: {best_model["Modello"]}\n')
    f.write(f'Accuratezza: {best_model["Accuratezza_Media"]:.3f} ± {best_model["Deviazione_Standard"]:.3f}\n')
    f.write(f'Miglioramento vs baseline: +{best_model["Miglioramento_vs_Baseline"]*100:.1f}%\n\n')
    
    f.write('RANKING COMPLETO:\n')
    f.write('-'*30 + '\n')
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        f.write(f'{i:2d}. {row["Modello"]:<25}: {row["Accuratezza_Media"]:.3f} (+{row["Miglioramento_vs_Baseline"]*100:+.1f}%)\n')

print('\n💾 Report salvato in: report_ottimizzazione.txt')
print('\n✅ ANALISI COMPLETATA!')
print('🎯 Controlla i grafici generati per visualizzazioni dettagliate')