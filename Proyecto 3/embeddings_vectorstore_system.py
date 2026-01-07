"""
SISTEMA DE EMBEDDINGS Y VECTOR STORE
Proyecto: Crisis de Sentido en la Generación Z mediante RAG
Embeddings semánticos + Vector Store (FAISS/Chroma)
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# ========================================
# PARTE 1: SISTEMA DE EMBEDDINGS
# ========================================

class EmbeddingGenerator:
    """
    Generador de embeddings semánticos usando Sentence Transformers
    Soporta múltiples modelos según necesidad
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa el generador de embeddings
        
        Modelos recomendados:
        - paraphrase-multilingual-MiniLM-L12-v2: Rápido, buen balance (384 dims)
        - paraphrase-multilingual-mpnet-base-v2: Mejor calidad (768 dims)
        - distiluse-base-multilingual-cased-v1: Alternativa rápida
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Cargando modelo de embeddings: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Modelo cargado exitosamente")
            print(f"   Dimensionalidad: {self.embedding_dim}")
        except ImportError:
            print("ERROR: sentence-transformers no está instalado")
            print("   Instalar con: pip install sentence-transformers")
            raise
    
    def generate_embeddings(self, texts: List[str], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos a vectorizar
            batch_size: Tamaño del lote para procesamiento
            show_progress: Mostrar barra de progreso
        
        Returns:
            Array numpy con embeddings (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("La lista de textos está vacía")
        
        print(f"\nGenerando embeddings para {len(texts)} documentos...")
        print(f"   Batch size: {batch_size}")
        
        # Filtrar textos vacíos
        valid_texts = [str(t).strip() if pd.notna(t) else "" for t in texts]
        valid_texts = [t if len(t) > 0 else "texto vacío" for t in valid_texts]
        
        # Generar embeddings
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Embeddings generados: {embeddings.shape}")
        return embeddings
    
    def evaluate_embedding_quality(self, texts: List[str], 
                                   embeddings: np.ndarray,
                                   sample_size: int = 5) -> Dict:
        """
        Evalúa la calidad de los embeddings generados
        """
        print(f"\nEVALUANDO CALIDAD DE EMBEDDINGS")
        print("=" * 60)
        
        # Calcular estadísticas básicas
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        std_norm = np.linalg.norm(embeddings, axis=1).std()
        
        print(f"   Norma promedio: {mean_norm:.4f}")
        print(f"   Desviación estándar: {std_norm:.4f}")
        
        # Calcular similitudes de ejemplo
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Tomar muestra aleatoria
        sample_indices = np.random.choice(len(texts), 
                                         min(sample_size, len(texts)), 
                                         replace=False)
        
        print(f"\n   Ejemplos de similitud semántica:")
        for idx in sample_indices[:3]:
            # Calcular similitudes con todos los demás
            similarities = cosine_similarity(
                embeddings[idx:idx+1], 
                embeddings
            )[0]
            
            # Encontrar los 3 más similares (excluyendo el mismo)
            most_similar = np.argsort(similarities)[::-1][1:4]
            
            print(f"\n   Texto original: {texts[idx][:100]}...")
            for similar_idx in most_similar:
                sim_score = similarities[similar_idx]
                print(f"      → Similitud {sim_score:.3f}: {texts[similar_idx][:80]}...")
        
        return {
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name
        }


# ========================================
# PARTE 2: VECTOR STORE CON FAISS
# ========================================

class FAISSVectorStore:
    """
    Vector Store usando FAISS para búsqueda eficiente de similitud
    """
    
    def __init__(self, embedding_dim: int):
        """
        Inicializa el vector store FAISS
        
        Args:
            embedding_dim: Dimensionalidad de los embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.metadata = []
        
        try:
            import faiss
            self.faiss = faiss
            print(f"FAISS disponible")
        except ImportError:
            print("ERROR: faiss no está instalado")
            print("   Instalar con: pip install faiss-cpu")
            raise
    
    def create_index(self, embeddings: np.ndarray, 
                     documents: List[str],
                     metadata: List[Dict] = None,
                     index_type: str = 'Flat'):
        """
        Crea el índice FAISS con los embeddings
        
        Args:
            embeddings: Array numpy con vectores
            documents: Lista de textos originales
            metadata: Metadatos asociados a cada documento
            index_type: Tipo de índice ('Flat', 'IVF', 'HNSW')
        """
        print(f"\nCreando índice FAISS...")
        print(f"   Tipo de índice: {index_type}")
        print(f"   Documentos: {len(documents)}")
        
        # Normalizar embeddings para cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        
        # Crear índice según tipo
        if index_type == 'Flat':
            # Búsqueda exacta (mejor para <100k documentos)
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == 'IVF':
            # Búsqueda aproximada con cuantización
            nlist = min(100, len(embeddings) // 10)
            quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
            self.index = self.faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist
            )
            self.index.train(embeddings_normalized.astype('float32'))
        else:
            raise ValueError(f"Tipo de índice no soportado: {index_type}")
        
        # Agregar vectores al índice
        self.index.add(embeddings_normalized.astype('float32'))
        
        # Guardar documentos y metadata
        self.documents = documents
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        print(f"Índice creado: {self.index.ntotal} vectores indexados")
    
    def search(self, query_embedding: np.ndarray, 
               k: int = 5,
               score_threshold: float = 0.0) -> List[Dict]:
        """
        Busca los k documentos más similares a la query
        
        Args:
            query_embedding: Vector de la query
            k: Número de resultados a retornar
            score_threshold: Umbral mínimo de similitud
        
        Returns:
            Lista de diccionarios con resultados y scores
        """
        if self.index is None:
            raise ValueError("Índice no creado. Ejecutar create_index() primero")
        
        # Normalizar query
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        # Buscar
        k_search = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_normalized, k_search)
        
        # Formatear resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """Guarda el índice y metadatos"""
        if self.index is None:
            raise ValueError("No hay índice para guardar")
        
        # Guardar índice FAISS
        self.faiss.write_index(self.index, f"{filepath}.index")
        
        # Guardar documentos y metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Vector store guardado: {filepath}")
    
    def load(self, filepath: str):
        """Carga el índice y metadatos"""
        # Cargar índice FAISS
        self.index = self.faiss.read_index(f"{filepath}.index")
        
        # Cargar documentos y metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Vector store cargado: {filepath}")
        print(f"   Documentos: {len(self.documents)}")


# ========================================
# PARTE 3: VECTOR STORE CON CHROMADB
# ========================================

class ChromaVectorStore:
    """
    Vector Store usando ChromaDB (alternativa moderna)
    """
    
    def __init__(self, collection_name: str = "genz_corpus"):
        """Inicializa ChromaDB"""
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            print(f"ChromaDB disponible")
        except ImportError:
            print("ERROR: chromadb no está instalado")
            print("   Instalar con: pip install chromadb")
            raise
    
    def create_collection(self, embeddings: np.ndarray,
                         documents: List[str],
                         metadata: List[Dict] = None):
        """Crea colección en ChromaDB"""
        print(f"\nCreando colección ChromaDB: {self.collection_name}")
        
        # Eliminar colección si existe
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # Crear nueva colección
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Corpus Gen Z para análisis filosófico"}
        )
        
        # Preparar datos
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = metadata if metadata else [{} for _ in documents]
        
        # Agregar documentos
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Colección creada: {len(documents)} documentos")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Busca documentos similares"""
        if self.collection is None:
            raise ValueError("Colección no creada")
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Formatear resultados
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # Convertir distancia a similitud
                'id': results['ids'][0][i]
            })
        
        return formatted


# ========================================
# PARTE 4: PIPELINE COMPLETO
# ========================================

class RAGVectorPipeline:
    """Pipeline completo de embeddings + vector store"""
    
    def __init__(self, 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 vector_store_type: str = 'faiss'):
        """
        Inicializa el pipeline
        
        Args:
            model_name: Modelo de sentence transformers
            vector_store_type: 'faiss' o 'chroma'
        """
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.vector_store_type = vector_store_type
        self.vector_store = None
        self.embeddings = None
    
    def build_from_csv(self, csv_path: str, 
                      text_column: str = 'texto_limpio',
                      metadata_columns: List[str] = None):
        """
        Construye el pipeline completo desde un CSV
        
        Args:
            csv_path: Ruta al CSV con corpus limpio
            text_column: Columna con el texto a vectorizar
            metadata_columns: Columnas adicionales como metadata
        """
        print("\n" + "="*80)
        print("CONSTRUYENDO PIPELINE RAG COMPLETO")
        print("="*80)
        
        # 1. Cargar datos
        print(f"\nCargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"{len(df)} documentos cargados")
        
        # 2. Preparar textos
        texts = df[text_column].tolist()
        
        # 3. Preparar metadata
        if metadata_columns:
            metadata = df[metadata_columns].to_dict('records')
        else:
            metadata = [{'index': i} for i in range(len(texts))]
        
        # 4. Generar embeddings
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # 5. Evaluar calidad
        self.embedding_generator.evaluate_embedding_quality(texts, self.embeddings)
        
        # 6. Crear vector store
        if self.vector_store_type == 'faiss':
            self.vector_store = FAISSVectorStore(
                self.embedding_generator.embedding_dim
            )
            self.vector_store.create_index(
                self.embeddings, 
                texts, 
                metadata,
                index_type='Flat'
            )
        elif self.vector_store_type == 'chroma':
            self.vector_store = ChromaVectorStore()
            self.vector_store.create_collection(
                self.embeddings,
                texts,
                metadata
            )
        
        print("\nPipeline RAG construido exitosamente")
        return self
    
    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Realiza una búsqueda semántica
        
        Args:
            query_text: Texto de consulta
            k: Número de resultados
        
        Returns:
            Lista de documentos relevantes con scores
        """
        # Generar embedding de la query
        query_embedding = self.embedding_generator.generate_embeddings(
            [query_text], 
            show_progress=False
        )[0]
        
        # Buscar en vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    def save(self, base_path: str = 'rag_vectorstore'):
        """Guarda el pipeline completo"""
        if self.vector_store_type == 'faiss':
            self.vector_store.save(base_path)
        
        # Guardar configuración
        config = {
            'model_name': self.embedding_generator.model_name,
            'vector_store_type': self.vector_store_type,
            'embedding_dim': self.embedding_generator.embedding_dim
        }
        with open(f"{base_path}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Pipeline guardado: {base_path}")


# ========================================
# PARTE 5: TESTING Y DEMOSTRACIÓN
# ========================================

def test_pipeline():
    """Función de prueba del pipeline completo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*80)
    print("TESTING PIPELINE RAG")
    print("="*80)
    
    # Construir pipeline
    pipeline = RAGVectorPipeline(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        vector_store_type='faiss'
    )
    
    # Cargar corpus limpio
    pipeline.build_from_csv(
        os.path.join(script_dir, 'corpus_final_corregido.csv'),
        text_column='texto_limpio',
        metadata_columns=['tema', 'fuente', 'tipo']
    )
    
    # Preguntas de prueba del proyecto
    test_queries = [
        "¿Qué expresiones utiliza la Gen Z para describir el vacío existencial?",
        "¿Cómo influyen los algoritmos en la construcción de identidad?",
        "¿Qué emociones aparecen cuando se habla de burnout digital?",
        "¿La Gen Z percibe la autonomía como algo propio o condicionado?",
        "¿Existen señales de crisis de sentido en los datos?"
    ]
    
    print("\nPROBANDO BUSQUEDAS SEMANTICAS")
    print("="*80)
    
    for query in test_queries[:3]:  # Probar con 3 primeras
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        results = pipeline.query(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Score: {result['score']:.4f}")
            print(f"      Tema: {result['metadata'].get('tema', 'N/A')}")
            print(f"      Texto: {result['document'][:200]}...")
    
    # Guardar pipeline
    pipeline.save(os.path.join(script_dir, 'rag_vectorstore_genz'))
    
    print("\n" + "="*80)
    print("TESTING COMPLETADO")
    print("="*80)


# ========================================
# EJECUCIÓN PRINCIPAL
# ========================================

if __name__ == "__main__":
    test_pipeline()
