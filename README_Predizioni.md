# 🔮 Sistema di Predizione per Malattie Respiratorie

## 📋 Panoramica

Questo sistema ti permette di utilizzare il modello di machine learning addestrato per fare predizioni su nuovi pazienti. Il modello può classificare pazienti in 4 categorie:

- **Altro** - Altre condizioni respiratorie
- **Asma bronchiale** - Asma
- **BPCO** - Broncopneumopatia cronica ostruttiva
- **Overlap asma/bpco** - Sovrapposizione asma-BPCO

## 📁 File del Sistema

### File Principali
- `predict_new_dataset.py` - Script principale per le predizioni
- `esempio_predizione.py` - Esempi pratici di utilizzo
- `advanced_ml_models_fixed.py` - Modello addestrato (versione corretta)

### File di Output
- `predizioni_risultati.xlsx` - Risultati delle predizioni
- `report_predizioni_dettagliato.xlsx` - Analisi completa

## 🚀 Come Utilizzare il Sistema

### Metodo 1: Utilizzo Semplice

```python
from predict_new_dataset import predict_new_dataset

# Fai predizioni su un nuovo dataset
risultati = predict_new_dataset('mio_nuovo_dataset.xlsx')

# Salva i risultati
risultati.to_excel('risultati_predizioni.xlsx', index=False)

# Visualizza statistiche
print(risultati['Predicted_Diagnosis'].value_counts())
print(f"Confidenza media: {risultati['Confidence'].mean():.3f}")
```

### Metodo 2: Utilizzo con Esempi Completi

```python
from esempio_predizione import esempio_predizione_semplice, analizza_risultati_dettagliati

# Esegui predizione con analisi
risultati = esempio_predizione_semplice()
analizza_risultati_dettagliati(risultati)
```

## 📊 Formato del Dataset

### Requisiti del Nuovo Dataset

Il tuo nuovo dataset deve avere colonne simili a quelle del dataset di training. Le colonne principali includono:

- **Dati demografici**: Età, sesso
- **Storia clinica**: Allergie, familiarità
- **Sintomi**: Tosse, dispnea, wheezing
- **Esami**: Spirometria, test allergologici
- **Abitudini**: Fumo, esposizioni

### Formati Supportati
- `.xlsx` (Excel)
- `.csv` (Comma-separated values)

### Esempio di Struttura
```
Età | Sesso | Allergie | Tosse | Dispnea | FEV1 | ...
45  | M     | Sì       | Sì    | No      | 2.1  | ...
62  | F     | No       | No    | Sì      | 1.8  | ...
```

## 🔧 Installazione e Setup

### Prerequisiti
```bash
pip install pandas numpy scikit-learn openpyxl
```

### Verifica File
Assicurati di avere questi file nella stessa cartella:
- `predict_new_dataset.py`
- `esempio_predizione.py`
- `COLD 30.07.2025.xlsx` (dataset originale per training)

## 📈 Interpretazione dei Risultati

### Colonne di Output

| Colonna | Descrizione |
|---------|-------------|
| `Predicted_Class` | Numero della classe (0-3) |
| `Predicted_Diagnosis` | Nome della diagnosi |
| `Confidence` | Confidenza della predizione (0-1) |
| `Confidence_Percent` | Confidenza in percentuale |
| `Prob_Altro` | Probabilità di "Altro" |
| `Prob_Asma_bronchiale` | Probabilità di "Asma bronchiale" |
| `Prob_BPCO` | Probabilità di "BPCO" |
| `Prob_Overlap_asma_bpco` | Probabilità di "Overlap" |

### Livelli di Confidenza

- **Alta confidenza (>80%)**: Predizione molto affidabile
- **Media confidenza (60-80%)**: Predizione buona
- **Bassa confidenza (<60%)**: Richiede revisione clinica

## ⚠️ Considerazioni Cliniche

### Limitazioni del Modello
1. **Accuratezza**: ~67% sul dataset di test
2. **Bias**: Migliore per asma bronchiale, meno accurato per overlap
3. **Dimensione**: Addestrato su dataset limitato

### Raccomandazioni
1. **Non sostituire il giudizio clinico**
2. **Usare come supporto decisionale**
3. **Verificare casi a bassa confidenza**
4. **Considerare il contesto clinico completo**

## 🔍 Risoluzione Problemi

### Errori Comuni

#### "File non trovato"
```python
# Verifica il path del file
import os
print(os.path.exists('mio_dataset.xlsx'))  # Deve essere True
```

#### "Colonne mancanti"
- Il sistema aggiunge automaticamente colonne mancanti con valore 0
- Verifica che le colonne principali siano presenti

#### "Errore di encoding"
- Il sistema gestisce automaticamente valori sconosciuti
- Sostituisce con valori più frequenti del training

### Debug

```python
# Verifica il dataset
import pandas as pd
df = pd.read_excel('mio_dataset.xlsx')
print(f"Forma: {df.shape}")
print(f"Colonne: {df.columns.tolist()}")
print(f"Tipi: {df.dtypes}")
```

## 📞 Supporto

### Controlli Pre-Predizione
1. ✅ File del dataset esiste
2. ✅ Formato corretto (.xlsx o .csv)
3. ✅ Colonne simili al training
4. ✅ Dati puliti (no valori strani)

### Validazione Post-Predizione
1. ✅ Confidenza media ragionevole (>50%)
2. ✅ Distribuzione diagnosi sensata
3. ✅ Nessun errore durante l'esecuzione

## 📚 Esempi Pratici

### Esempio 1: Dataset Piccolo
```python
# Per 10-50 pazienti
risultati = predict_new_dataset('pazienti_ambulatorio.xlsx')
print(risultati[['Predicted_Diagnosis', 'Confidence_Percent']].head(10))
```

### Esempio 2: Dataset Grande
```python
# Per centinaia di pazienti
risultati = predict_new_dataset('database_ospedale.xlsx')

# Analisi per reparto/medico
if 'Reparto' in risultati.columns:
    per_reparto = risultati.groupby('Reparto')['Predicted_Diagnosis'].value_counts()
    print(per_reparto)
```

### Esempio 3: Monitoraggio Qualità
```python
# Identifica casi che richiedono attenzione
bassa_confidenza = risultati[risultati['Confidence'] < 0.6]
print(f"Casi da rivedere: {len(bassa_confidenza)}")

# Salva per revisione
bassa_confidenza.to_excel('casi_da_rivedere.xlsx', index=False)
```

## 🎯 Best Practices

1. **Sempre verificare la confidenza** delle predizioni
2. **Combinare con valutazione clinica** esperta
3. **Monitorare la performance** nel tempo
4. **Aggiornare il modello** con nuovi dati quando possibile
5. **Documentare le decisioni** basate sulle predizioni

---

*Questo sistema è uno strumento di supporto decisionale. Le decisioni cliniche finali devono sempre essere prese da professionisti sanitari qualificati.*