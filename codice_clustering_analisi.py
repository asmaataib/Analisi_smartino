# ============================================================================
# CODICE COMPLETO PER CLUSTERING E ANALISI DEI DATI RESPIRATORI
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Configurazione per i grafici
plt.style.use('default')
sns.set_palette('husl')

# ============================================================================
# 1. CARICAMENTO E ESPLORAZIONE INIZIALE DEI DATI
# ============================================================================

# Caricamento del file Excel
file_path = 'COLD 30.07.2025.xlsx'  # Assicurati che il file sia nella directory corrente
df = pd.read_excel(file_path)

print('=== INFORMAZIONI GENERALI SUL DATASET ===')
print(f'Dimensioni del dataset: {df.shape}')
print(f'Numero di pazienti: {df.shape[0]}')
print(f'Numero di variabili: {df.shape[1]}')

# Informazioni sul dataset
df.info()

# Statistiche descrittive
print('\n=== STATISTICHE DESCRITTIVE ===')
print(df.describe())

# ============================================================================
# 2. ESPLORAZIONE APPROFONDITA DEI DATI
# ============================================================================

print('\n=== ESPLORAZIONE APPROFONDITA DEI DATI ===')

# Distribuzione delle variabili numeriche principali
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f'\nVariabili numeriche: {len(numeric_cols)}')
print(list(numeric_cols))

# Distribuzione delle variabili categoriche
categorical_cols = df.select_dtypes(include=['object']).columns
print(f'\nVariabili categoriche: {len(categorical_cols)}')
print(list(categorical_cols))

# Visualizzazione distribuzione età
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['Età'].hist(bins=20, alpha=0.7)
plt.title('Distribuzione Età')
plt.xlabel('Età')
plt.ylabel('Frequenza')

# Distribuzione diagnosi
plt.subplot(1, 3, 2)
df['Diagnosi'].value_counts().plot(kind='bar')
plt.title('Distribuzione Diagnosi')
plt.xlabel('Codice Diagnosi')
plt.ylabel('Numero Pazienti')
plt.xticks(rotation=0)

# Distribuzione probabilità COLD
plt.subplot(1, 3, 3)
df['Probabilità di COLD'].hist(bins=10, alpha=0.7)
plt.title('Distribuzione Probabilità COLD')
plt.xlabel('Probabilità COLD')
plt.ylabel('Frequenza')

plt.tight_layout()
plt.show()

# ============================================================================
# 3. ANALISI DEI VALORI MANCANTI
# ============================================================================

print('\n=== ANALISI DEI VALORI MANCANTI ===')

# Conteggio valori mancanti
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonna': missing_data.index,
    'Valori_Mancanti': missing_data.values,
    'Percentuale': missing_percent.values
}).sort_values('Valori_Mancanti', ascending=False)

print(missing_df[missing_df['Valori_Mancanti'] > 0])

if missing_df['Valori_Mancanti'].sum() == 0:
    print('✅ Nessun valore mancante trovato!')

# ============================================================================
# 4. ANALISI DELLA VARIABILE TARGET (DIAGNOSI)
# ============================================================================

print('\n=== ANALISI DELLA VARIABILE TARGET ===') 

# Mappatura delle diagnosi
diagnosis_mapping = {
    0: 'Altro',
    1: 'Asma bronchiale',
    2: 'BPCO',
    3: 'Overlap asma/bpco'
}

# Distribuzione delle diagnosi
diagnosis_counts = df['Diagnosi'].value_counts().sort_index()
print('\nDistribuzione delle diagnosi:')
for diag_code, count in diagnosis_counts.items():
    diag_name = diagnosis_mapping.get(diag_code, f'Codice {diag_code}')
    percentage = (count / len(df)) * 100
    print(f'{diag_code} - {diag_name}: {count} pazienti ({percentage:.1f}%)')

# Visualizzazione
plt.figure(figsize=(12, 5))

# Grafico a barre
plt.subplot(1, 2, 1)
diagnosis_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.title('Distribuzione delle Diagnosi')
plt.xlabel('Codice Diagnosi')
plt.ylabel('Numero di Pazienti')
plt.xticks(rotation=0)

# Grafico a torta
plt.subplot(1, 2, 2)
labels = [f'{diagnosis_mapping[i]}\n({count})' for i, count in diagnosis_counts.items()]
plt.pie(diagnosis_counts.values, labels=labels, autopct='%1.1f%%', 
        colors=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.title('Proporzione delle Diagnosi')

plt.tight_layout()
plt.show()

# ============================================================================
# 5. PREPROCESSING PER MACHINE LEARNING
# ============================================================================

print('\n=== PREPROCESSING PER MACHINE LEARNING ===')

# Creazione di una copia per il preprocessing
df_ml = df.copy()

# Encoding delle variabili categoriche
label_encoders = {}
for col in categorical_cols:
    if col != 'Data questionario':  # Escludiamo le date
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le

# Rimozione della colonna data
if 'Data questionario' in df_ml.columns:
    df_ml = df_ml.drop('Data questionario', axis=1)

# Separazione features e target
X = df_ml.drop(['Diagnosi', 'Identificativo'], axis=1)
y = df_ml['Diagnosi']

print(f'Dimensioni features: {X.shape}')
print(f'Dimensioni target: {y.shape}')
print(f'Features utilizzate: {list(X.columns)}')

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\nTrain set: {X_train_scaled.shape}')
print(f'Test set: {X_test_scaled.shape}')

# ============================================================================
# 6. CLUSTERING NON SUPERVISIONATO
# ============================================================================

print('\n=== ANALISI DI CLUSTERING ===')

# Utilizziamo tutti i dati standardizzati per il clustering
X_clustering = scaler.fit_transform(X)

# 1. Determinare il numero ottimale di cluster con il metodo Elbow
print('1. Determinazione numero ottimale di cluster...')

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clustering)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_clustering, kmeans.labels_))

# Grafici per determinare k ottimale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Metodo Elbow
ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Numero di Cluster (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Metodo Elbow per Determinare k Ottimale')
ax1.grid(True)

# Silhouette Score
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Numero di Cluster (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score per Diversi k')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Trova k ottimale basato su silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f'\nK ottimale basato su Silhouette Score: {optimal_k}')
print(f'Silhouette Score massimo: {max(silhouette_scores):.3f}')

# 2. Applicare K-means con 3 cluster (Asma, BPCO, Altro)
print('\n2. Applicazione K-means con 3 cluster...')

# Forziamo 3 cluster come richiesto (Asma, BPCO, Altro)
n_clusters = 3
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_clustering)

# Aggiungiamo le etichette al dataset originale
df_clustered = df.copy()
df_clustered['Cluster'] = cluster_labels

# Statistiche dei cluster
print(f'Silhouette Score con {n_clusters} cluster: {silhouette_score(X_clustering, cluster_labels):.3f}')
print(f'\nDistribuzione dei pazienti nei cluster:')
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for i, count in enumerate(cluster_counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f'Cluster {i}: {count} pazienti ({percentage:.1f}%)')

# Visualizzazione dei cluster con PCA
print('\n3. Visualizzazione dei cluster...')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']

for i in range(n_clusters):
    mask = cluster_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i], label=cluster_names[i], alpha=0.7, s=50)

# Centroidi
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroidi')

plt.xlabel(f'Prima Componente Principale ({pca.explained_variance_ratio_[0]:.1%} varianza)')
plt.ylabel(f'Seconda Componente Principale ({pca.explained_variance_ratio_[1]:.1%} varianza)')
plt.title('Visualizzazione Cluster con PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Varianza spiegata dalle prime 2 componenti: {sum(pca.explained_variance_ratio_):.1%}')

# 4. Caratterizzazione dei cluster
print('\n4. Caratterizzazione dei cluster...')

# Analisi delle caratteristiche principali per ogni cluster
print('\n=== CARATTERISTICHE MEDIE PER CLUSTER ===')

# Variabili numeriche chiave
key_numeric_vars = ['Età', 'Diagnosi', 'Probabilità di COLD',
                   'Il paziente ha avuto tosse persistente o ricorrente?',
                   'Il paziente è sensibilizzato ad allergeni inalatori?',
                   'Il paziente è o è stato fumatore?']

cluster_summary = df_clustered.groupby('Cluster')[key_numeric_vars].mean()
print(cluster_summary.round(2))

# Visualizzazione delle caratteristiche principali
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, var in enumerate(key_numeric_vars):
    if var in df_clustered.columns:
        df_clustered.boxplot(column=var, by='Cluster', ax=axes[i])
        axes[i].set_title(f'{var} per Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(var)

plt.suptitle('Distribuzione delle Variabili Chiave per Cluster', y=1.02)
plt.tight_layout()
plt.show()

# 5. Etichettatura dei cluster basata sulle caratteristiche
print('\n5. Interpretazione e etichettatura dei cluster...')

# Analisi dettagliata per interpretare i cluster
cluster_interpretation = {}

for cluster_id in range(n_clusters):
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    
    # Caratteristiche principali
    avg_age = cluster_data['Età'].mean()
    avg_diagnosis = cluster_data['Diagnosi'].mean()
    avg_cold_prob = cluster_data['Probabilità di COLD'].mean()
    smoker_pct = cluster_data['Il paziente è o è stato fumatore?'].mean() * 100
    allergy_pct = cluster_data['Il paziente è sensibilizzato ad allergeni inalatori?'].mean() * 100
    cough_pct = cluster_data['Il paziente ha avuto tosse persistente o ricorrente?'].mean() * 100
    
    cluster_interpretation[cluster_id] = {
        'size': len(cluster_data),
        'avg_age': avg_age,
        'avg_diagnosis': avg_diagnosis,
        'avg_cold_prob': avg_cold_prob,
        'smoker_pct': smoker_pct,
        'allergy_pct': allergy_pct,
        'cough_pct': cough_pct
    }
    
    print(f'\n--- CLUSTER {cluster_id} ---')
    print(f'Dimensione: {len(cluster_data)} pazienti')
    print(f'Età media: {avg_age:.1f}')
    print(f'Diagnosi media: {avg_diagnosis:.2f}')
    print(f'Probabilità COLD media: {avg_cold_prob:.2f}')
    print(f'Fumatori: {smoker_pct:.1f}%')
    print(f'Allergici: {allergy_pct:.1f}%')
    print(f'Tosse persistente: {cough_pct:.1f}%')

# Proposta di etichettatura
print('\n=== PROPOSTA DI ETICHETTATURA ===')

# Logica per assegnare etichette basata sulle caratteristiche
cluster_labels_proposed = {}

for cluster_id, stats in cluster_interpretation.items():
    if stats['allergy_pct'] > 60 and stats['avg_cold_prob'] < 2.5:
        cluster_labels_proposed[cluster_id] = 'ASMA'
    elif stats['smoker_pct'] > 60 and stats['avg_cold_prob'] > 2.5:
        cluster_labels_proposed[cluster_id] = 'BPCO'
    else:
        cluster_labels_proposed[cluster_id] = 'MISTO/ALTRO'

# Aggiungiamo le etichette interpretate
df_clustered['Etichetta_Cluster'] = df_clustered['Cluster'].map(cluster_labels_proposed)

for cluster_id, label in cluster_labels_proposed.items():
    count = len(df_clustered[df_clustered['Cluster'] == cluster_id])
    print(f'Cluster {cluster_id} → {label} ({count} pazienti)')

# Distribuzione finale
print('\n=== DISTRIBUZIONE FINALE ===')
final_distribution = df_clustered['Etichetta_Cluster'].value_counts()
print(final_distribution)

# Grafico finale
plt.figure(figsize=(12, 8))
colors_final = {'ASMA': 'lightblue', 'BPCO': 'lightcoral', 'MISTO/ALTRO': 'lightgreen'}

for label in df_clustered['Etichetta_Cluster'].unique():
    mask = df_clustered['Etichetta_Cluster'] == label
    cluster_data_pca = X_pca[mask]
    plt.scatter(cluster_data_pca[:, 0], cluster_data_pca[:, 1], 
               c=colors_final[label], label=label, alpha=0.7, s=50)

plt.xlabel(f'Prima Componente Principale ({pca.explained_variance_ratio_[0]:.1%} varianza)')
plt.ylabel(f'Seconda Componente Principale ({pca.explained_variance_ratio_[1]:.1%} varianza)')
plt.title('Clustering Finale: Asma, BPCO e Misto/Altro')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('\n✅ Clustering completato! I pazienti sono stati classificati in:')
print('- ASMA: Pazienti con alta sensibilizzazione allergica')
print('- BPCO: Pazienti fumatori con alta probabilità COLD')
print('- MISTO/ALTRO: Pazienti con caratteristiche intermedie')

# ============================================================================
# 7. CLASSIFICAZIONE SUPERVISIONATA
# ============================================================================

print('\n=== CLASSIFICAZIONE SUPERVISIONATA ===')

# Preparazione dei dati per la classificazione
print('\n1. Preparazione dei dati per classificazione supervisionata...')

# Target variable
y_classification = df['Diagnosi']

# Features (escludiamo la diagnosi)
X_classification = X.copy()  # Usiamo le stesse features del clustering

print(f'Dimensioni dataset: {X_classification.shape}')
print(f'Numero di classi: {len(y_classification.unique())}')
print(f'Classi presenti: {sorted(y_classification.unique())}')

# Split train/test
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_classification, y_classification, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_classification
)

# Standardizzazione
scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

print(f'\nTrain set: {X_train_cls_scaled.shape}')
print(f'Test set: {X_test_cls_scaled.shape}')

# Verifica bilanciamento classi nel train set
train_distribution = pd.Series(y_train_cls).value_counts().sort_index()
print('\nDistribuzione classi nel training set:')
for diag_code, count in train_distribution.items():
    percentage = (count / len(y_train_cls)) * 100
    print(f'Classe {diag_code}: {count} campioni ({percentage:.1f}%)')

# 2. Addestramento di diversi modelli
print('\n2. Addestramento modelli di classificazione...')

# Definizione dei modelli
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Cross-validation per valutare i modelli
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Valutazione con Cross-Validation (5-fold):')
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_cls_scaled, y_train_cls, 
                               cv=cv, scoring='accuracy')
    cv_results[name] = cv_scores
    print(f'{name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')

# Selezione del miglior modello
best_model_name = max(cv_results.keys(), 
                     key=lambda x: cv_results[x].mean())
best_model = models[best_model_name]

print(f'\n🏆 Miglior modello: {best_model_name}')

# Addestramento del miglior modello
best_model.fit(X_train_cls_scaled, y_train_cls)

# Predizioni
y_pred_train = best_model.predict(X_train_cls_scaled)
y_pred_test = best_model.predict(X_test_cls_scaled)

# Accuratezza
train_accuracy = accuracy_score(y_train_cls, y_pred_train)
test_accuracy = accuracy_score(y_test_cls, y_pred_test)

print(f'\nAccuratezza Training: {train_accuracy:.3f}')
print(f'Accuratezza Test: {test_accuracy:.3f}')

# 3. Valutazione dettagliata del modello
print('\n3. Valutazione dettagliata del modello...')

# Classification Report
print('\n=== CLASSIFICATION REPORT ===')
target_names = [diagnosis_mapping[i] for i in sorted(y_classification.unique())]
print(classification_report(y_test_cls, y_pred_test, target_names=target_names))

# Confusion Matrix
print('\n=== CONFUSION MATRIX ===')
cm = confusion_matrix(y_test_cls, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.tight_layout()
plt.show()

# Feature Importance (se disponibile)
if hasattr(best_model, 'feature_importances_'):
    print('\n=== FEATURE IMPORTANCE ===')
    feature_importance = pd.DataFrame({
        'feature': X_classification.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 15 features più importanti
    top_features = feature_importance.head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importanza')
    plt.title(f'Top 15 Feature più Importanti - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print('Top 10 feature più importanti:')
    for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
        print(f'{i:2d}. {row["feature"]}: {row["importance"]:.4f}')

print(f'\n✅ Classificazione completata con {best_model_name}!')
print(f'Accuratezza finale: {test_accuracy:.1%}')

# ============================================================================
# 8. CONFRONTO TRA CLUSTERING E CLASSIFICAZIONE SUPERVISIONATA
# ============================================================================

print('\n=== CONFRONTO CLUSTERING VS CLASSIFICAZIONE SUPERVISIONATA ===')

# Predizioni del modello supervisionato su tutto il dataset
X_all_scaled = scaler_cls.transform(X_classification)
y_pred_supervised = best_model.predict(X_all_scaled)

# Aggiungiamo le predizioni al dataframe
df_comparison = df_clustered.copy()
df_comparison['Predizione_Supervisionata'] = y_pred_supervised

# Mappatura per confronto
df_comparison['Diagnosi_Reale'] = df_comparison['Diagnosi'].map(diagnosis_mapping)
df_comparison['Predizione_Supervisionata_Nome'] = df_comparison['Predizione_Supervisionata'].map(diagnosis_mapping)

print('=== CONFRONTO METODI ===')

# Tabella di confronto
comparison_table = pd.crosstab(
    df_comparison['Etichetta_Cluster'], 
    df_comparison['Diagnosi_Reale'],
    margins=True
)

print('Clustering (righe) vs Diagnosi Reale (colonne):')
print(comparison_table)

# Accuratezza del clustering rispetto alla diagnosi reale
# Mappiamo i cluster alle diagnosi più frequenti
cluster_to_diagnosis = {}
for cluster_label in df_comparison['Etichetta_Cluster'].unique():
    cluster_data = df_comparison[df_comparison['Etichetta_Cluster'] == cluster_label]
    most_common_diagnosis = cluster_data['Diagnosi_Reale'].mode()[0]
    cluster_to_diagnosis[cluster_label] = most_common_diagnosis

print('\nMappatura cluster → diagnosi più frequente:')
for cluster, diagnosis in cluster_to_diagnosis.items():
    print(f'{cluster} → {diagnosis}')

# Calcolo accuratezza clustering
df_comparison['Cluster_Mapped'] = df_comparison['Etichetta_Cluster'].map(cluster_to_diagnosis)
clustering_accuracy = (df_comparison['Cluster_Mapped'] == df_comparison['Diagnosi_Reale']).mean()

# Accuratezza classificazione supervisionata
supervised_accuracy = (df_comparison['Predizione_Supervisionata_Nome'] == df_comparison['Diagnosi_Reale']).mean()

print(f'\n=== RISULTATI FINALI ===')
print(f'Accuratezza Clustering: {clustering_accuracy:.1%}')
print(f'Accuratezza Classificazione Supervisionata: {supervised_accuracy:.1%}')

# Visualizzazione finale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuratezza per metodo
methods = ['Clustering\n(Non supervisionato)', 'Classificazione\n(Supervisionata)']
accuracies = [clustering_accuracy, supervised_accuracy]
colors = ['lightblue', 'lightcoral']

ax1.bar(methods, accuracies, color=colors)
ax1.set_ylabel('Accuratezza')
ax1.set_title('Confronto Accuratezza dei Metodi')
ax1.set_ylim(0, 1)
for i, acc in enumerate(accuracies):
    ax1.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontweight='bold')

# Distribuzione delle predizioni
pred_counts = df_comparison['Predizione_Supervisionata_Nome'].value_counts()
ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
ax2.set_title('Distribuzione Predizioni\nClassificazione Supervisionata')

plt.tight_layout()
plt.show()

print('\n✅ Analisi completa terminata!')
print('📊 Sono stati implementati sia clustering che classificazione supervisionata')
print('🎯 I risultati mostrano le performance di entrambi gli approcci')

# ============================================================================
# FINE DEL CODICE
# ============================================================================

print('\n' + '='*80)
print('ANALISI COMPLETATA SUCCESSFULLY!')
print('='*80)
print('\nRiepilogo:')
print(f'- Dataset analizzato: {df.shape[0]} pazienti, {df.shape[1]} variabili')
print(f'- Clustering: 3 gruppi identificati (ASMA, BPCO, MISTO/ALTRO)')
print(f'- Classificazione: {best_model_name} con accuratezza {test_accuracy:.1%}')
print(f'- Confronto: Clustering {clustering_accuracy:.1%} vs Supervisionato {supervised_accuracy:.1%}')
print('='*80)