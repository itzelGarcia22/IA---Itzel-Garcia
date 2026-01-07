# Script para generar gráficas del análisis de corpus y embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("Cargando datos...")
df_corpus = pd.read_csv('corpus_final_corregido.csv')
df_corpus['longitud_texto'] = df_corpus['texto_limpio'].str.len()

# Cargar modelo y generar embeddings de muestra
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sample_size = min(1000, len(df_corpus))
df_sample = df_corpus.sample(n=sample_size, random_state=42)
texts_sample = df_sample['texto_limpio'].tolist()

print(f"Generando embeddings para {sample_size} documentos...")
embeddings_sample = model.encode(texts_sample, show_progress_bar=False)

# 1. Dashboard de temas
print("Generando gráfica 1: Dashboard de temas...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

tema_counts = df_corpus['tema'].value_counts()
tipo_counts = df_corpus['tipo'].value_counts()

tema_counts.head(10).plot(kind='barh', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Top 10 Temas por Frecuencia', fontsize=12, fontweight='bold')
ax1.set_xlabel('Número de Documentos')
ax1.grid(True, alpha=0.3)

tipo_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
ax2.set_title('Distribución por Tipo de Documento', fontsize=12, fontweight='bold')
ax2.set_ylabel('')

longitud_por_tema_top = df_corpus[df_corpus['tema'].isin(tema_counts.head(5).index)]
sns.boxplot(data=longitud_por_tema_top, x='longitud_texto', y='tema', ax=ax3, palette='Set3')
ax3.set_title('Distribución de Longitud de Texto por Tema (Top 5)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitud (caracteres)')
ax3.set_ylabel('Tema')

def calcular_diversidad_lexica(texto):
    palabras = texto.lower().split()
    return len(set(palabras)) / len(palabras) if palabras else 0

df_corpus['diversidad_lexica'] = df_corpus['texto_limpio'].apply(calcular_diversidad_lexica)
sns.barplot(data=df_corpus, x='tipo', y='diversidad_lexica', ax=ax4, palette='pastel')
ax4.set_title('Diversidad Léxica por Tipo de Documento', fontsize=12, fontweight='bold')
ax4.set_ylabel('Diversidad Léxica')
ax4.set_xlabel('Tipo de Documento')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafica_1_dashboard_temas.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Dashboard de embeddings
print("Generando gráfica 2: Dashboard de embeddings...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

normas = np.linalg.norm(embeddings_sample, axis=1)
ax1.hist(normas, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax1.axvline(normas.mean(), color='red', linestyle='--', linewidth=2, label=f'Promedio: {normas.mean():.3f}')
ax1.axvline(np.median(normas), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(normas):.3f}')
ax1.set_title('Distribución de Normas de Embeddings', fontsize=12, fontweight='bold')
ax1.set_xlabel('Norma del Vector')
ax1.set_ylabel('Frecuencia')
ax1.legend()
ax1.grid(True, alpha=0.3)

df_normas = pd.DataFrame({'norma': normas, 'tipo': df_sample['tipo'].values})
sns.boxplot(data=df_normas, x='tipo', y='norma', ax=ax2, palette='Set2')
ax2.set_title('Normas de Embeddings por Tipo de Documento', fontsize=12, fontweight='bold')
ax2.set_ylabel('Norma del Embedding')
ax2.set_xlabel('Tipo de Documento')
ax2.grid(True, alpha=0.3)

# Silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
k_range = range(2, 11)
silhouette_scores = []
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = kmeans_temp.fit_predict(embeddings_sample)
    score = silhouette_score(embeddings_sample, clusters_temp)
    silhouette_scores.append(score)

best_k = k_range[np.argmax(silhouette_scores)]
ax3.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
ax3.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'k óptimo = {best_k}')
ax3.set_title('Evolución del Silhouette Score por Número de Clusters', fontsize=12, fontweight='bold')
ax3.set_xlabel('Número de Clusters (k)')
ax3.set_ylabel('Silhouette Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

similitudes_promedio = []
for i in range(len(embeddings_sample)):
    sims = cosine_similarity([embeddings_sample[i]], embeddings_sample)[0]
    similitudes_promedio.append(sims.mean())

ax4.scatter(normas, similitudes_promedio, alpha=0.6, c=df_sample['tipo'].map({'tweet_genz': 'blue', 'estudio_academico': 'red'}).fillna('gray'), s=50)
ax4.set_title('Relación entre Norma del Embedding y Similitud Promedio', fontsize=12, fontweight='bold')
ax4.set_xlabel('Norma del Embedding')
ax4.set_ylabel('Similitud Promedio')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafica_2_dashboard_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PCA y t-SNE
print("Generando gráfica 3: Visualizaciones PCA y t-SNE...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PCA
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings_sample)
df_viz = pd.DataFrame({
    'PC1': embeddings_2d[:, 0],
    'PC2': embeddings_2d[:, 1],
    'tipo': df_sample['tipo'].values
})

sns.scatterplot(data=df_viz, x='PC1', y='PC2', hue='tipo', palette='Set1', alpha=0.7, ax=ax1)
ax1.set_title('Visualización 2D de Embeddings por Tipo (PCA)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Componente Principal 1')
ax1.set_ylabel('Componente Principal 2')
ax1.legend(title='Tipo')
ax1.grid(True, alpha=0.3)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_tsne = tsne.fit_transform(embeddings_sample)
df_viz['tSNE1'] = embeddings_tsne[:, 0]
df_viz['tSNE2'] = embeddings_tsne[:, 1]

# Visualización simple por tipo
sns.scatterplot(data=df_viz, x='tSNE1', y='tSNE2', hue='tipo', palette='tab10', alpha=0.7, s=50, ax=ax2)
ax2.set_title('Visualización 2D de Embeddings por Tipo (t-SNE)', fontsize=14, fontweight='bold')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.legend(title='Tipo', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafica_3_pca_tsne.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Clusters y heatmap
print("Generando gráfica 4: Clusters y heatmap...")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_sample = df_sample.assign(cluster=kmeans_final.fit_predict(embeddings_sample))

centroides = np.array([embeddings_sample[df_sample['cluster'] == c].mean(axis=0) for c in range(best_k)])
cluster_sim_matrix = cosine_similarity(centroides)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Heatmap
sns.heatmap(cluster_sim_matrix, annot=True, cmap='YlOrRd', fmt='.3f',
            xticklabels=[f'Cluster {i}' for i in range(best_k)],
            yticklabels=[f'Cluster {i}' for i in range(best_k)], ax=ax1)
ax1.set_title('Similitud entre Centroides de Clusters', fontsize=14, fontweight='bold')

# Clusters en t-SNE
scatter = ax2.scatter(df_sample['tSNE1'], df_sample['tSNE2'],
                     c=df_sample['cluster'], cmap='tab10', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

for c in range(best_k):
    centroid_tsne = df_sample[df_sample['cluster'] == c][['tSNE1', 'tSNE2']].mean()
    ax2.scatter(centroid_tsne['tSNE1'], centroid_tsne['tSNE2'],
               marker='X', s=200, c=f'C{c}', edgecolors='black', linewidth=3, label=f'Centroide {c}')

ax2.set_title('Clusters Semánticos con Centroides - Espacio t-SNE', fontsize=14, fontweight='bold')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig('grafica_4_clusters_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Dashboard ejecutivo
print("Generando gráfica 5: Dashboard ejecutivo...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Dashboard Ejecutivo: Crisis de Sentido en la Generación Z', fontsize=16, fontweight='bold')

corpus_composition = df_corpus['tipo'].value_counts()
axes[0,0].pie(corpus_composition.values, labels=corpus_composition.index,
              autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
axes[0,0].set_title('Composición del Corpus', fontweight='bold')

top_5_temas = tema_counts.head(5)
axes[0,1].barh(range(len(top_5_temas)), top_5_temas.values, color='#45B7D1')
axes[0,1].set_yticks(range(len(top_5_temas)))
axes[0,1].set_yticklabels([t[:20] + '...' for t in top_5_temas.index])
axes[0,1].set_title('Top 5 Temas', fontweight='bold')
axes[0,1].set_xlabel('Documentos')

axes[0,2].hist(normas, bins=30, alpha=0.7, color='#96CEB4', edgecolor='black')
axes[0,2].axvline(normas.mean(), color='red', linestyle='--', linewidth=2)
axes[0,2].set_title('Distribución de Normas\nEmbeddings', fontweight='bold')
axes[0,2].set_xlabel('Norma')
axes[0,2].set_ylabel('Frecuencia')

cluster_sizes = df_sample['cluster'].value_counts().sort_index()
axes[1,0].bar(range(len(cluster_sizes)), cluster_sizes.values, color='#FECA57')
axes[1,0].set_title('Tamaño de Clusters Semánticos', fontweight='bold')
axes[1,0].set_xlabel('Cluster')
axes[1,0].set_ylabel('Documentos')
axes[1,0].set_xticks(range(len(cluster_sizes)))

# Simular query scores
test_queries = [
    "¿Qué expresiones utiliza la Gen Z para describir el vacío existencial?",
    "¿Cómo influyen los algoritmos en la construcción de identidad?",
    "¿Qué emociones aparecen cuando se habla de burnout digital?",
    "¿La Gen Z percibe la autonomía como algo propio o condicionado?",
    "¿Existen señales de crisis de sentido en los datos?"
]

query_scores = []
for query in test_queries:
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings_sample)[0]
    query_scores.append(similarities.max())

axes[1,1].barh(range(len(query_scores)), query_scores, color='#FF9FF3')
axes[1,1].set_yticks(range(len(query_scores)))
axes[1,1].set_yticklabels([f'Q{i+1}' for i in range(len(query_scores))])
axes[1,1].set_title('Scores de Búsqueda RAG', fontweight='bold')
axes[1,1].set_xlabel('Similitud Máxima')

metrics_text = f"""
📈 MÉTRICAS FINALES

Corpus: {len(df_corpus)} documentos
Embeddings: {embeddings_sample.shape[1]} dimensiones
Clusters: {best_k} grupos semánticos
Similitud Promedio: {cosine_similarity(embeddings_sample).mean():.3f}

🎯 HALLAZGOS CLAVE:
• Crisis existencial: {tema_counts.get('Generación Z y crisis de sentido', 0)/len(df_corpus)*100:.1f}%
• Clusters semánticos: {best_k} identificados
• Búsqueda RAG: {np.mean(query_scores):.3f} avg score
• Calidad embeddings: Norma {normas.mean():.3f} ± {normas.std():.3f}
"""

axes[1,2].text(0.1, 0.5, metrics_text, transform=axes[1,2].transAxes,
               fontsize=10, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F7F7"))
axes[1,2].set_title('Resumen Ejecutivo', fontweight='bold')
axes[1,2].set_xlim(0, 1)
axes[1,2].set_ylim(0, 1)
axes[1,2].axis('off')

plt.tight_layout()
plt.savefig('grafica_5_dashboard_ejecutivo.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Todas las gráficas generadas y guardadas como imágenes PNG")
print("Archivos creados:")
print("- grafica_1_dashboard_temas.png")
print("- grafica_2_dashboard_embeddings.png")
print("- grafica_3_pca_tsne.png")
print("- grafica_4_clusters_heatmap.png")
print("- grafica_5_dashboard_ejecutivo.png")