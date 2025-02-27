# src\rag_engine.py

import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize global variables
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
faiss_index = None
stored_chunks = []
schema_embeddings = None
db_schema_index = None
db_schema_chunks = []

def chunk_text(text, max_chunk_size=512, overlap=50):
    """Splits the text into manageable chunks with overlap for context."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        words = para.split()
        if (current_length + len(words)) <= max_chunk_size:
            current_chunk.append(para)
            current_length += len(words)
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
            if len(words) > max_chunk_size:
                for i in range(0, len(words), max_chunk_size - overlap):
                    chunk_words = words[i:i + max_chunk_size]
                    chunks.append(" ".join(chunk_words))
                current_chunk = []
                current_length = 0
            else:
                current_chunk = [para]
                current_length = len(words)
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def search_relevant_chunks(query, top_k=3):
    """Finds the most relevant document chunks for the query."""
    global faiss_index, stored_chunks
    
    if faiss_index is None:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)

    # Get chunks and sort by relevance score (distance)
    results = [(stored_chunks[i], distances[0][j]) for j, i in enumerate(indexes[0]) if i < len(stored_chunks)]
    results.sort(key=lambda x: x[1])  # Sort by distance (lower is better)

    return [chunk for chunk, _ in results]

def train_database_embeddings(db_schema):
    """Creates embeddings from database schema for RAG."""
    global db_schema_index, db_schema_chunks
    
    if not db_schema:
        print("❌ No schema to train on!")
        return
    
    # Convert schema to text representation
    schema_text = "\n".join([
        f"Table: {table}\n  Columns: {', '.join([f'{col['column']} ({col['type']})' for col in columns])}\n  Relationships: {', '.join([f'{col['fk_column']} -> {col['referenced_table']}.{col['referenced_column']}' for col in columns if col['constraint'] == 'FOREIGN KEY']) or 'None'}"
        for table, columns in db_schema.items()
    ])
    
    # Add synonyms if available
    try:
        from synonyms_seham_winoffre import TABLE_SYNONYMS, COLUMN_SYNONYMS
        synonyms_text = "\nSynonyms:\n" + "\n".join([f"{k} -> {v}" for k, v in TABLE_SYNONYMS.items()]) + "\n" + "\n".join([f"{k} -> {v}" for k, v in COLUMN_SYNONYMS.items()])
        full_schema_text = schema_text + "\n" + synonyms_text
    except ImportError:
        full_schema_text = schema_text
        print("Note: Synonyms module not found, using raw schema only.")

    # Create chunks and embeddings
    db_schema_chunks = chunk_text(full_schema_text, max_chunk_size=512)
    schema_embeddings = embedding_model.encode(db_schema_chunks)
    
    # Create FAISS index
    db_schema_index = faiss.IndexFlatL2(schema_embeddings.shape[1])
    db_schema_index.add(np.array(schema_embeddings).astype('float32'))
    
    print("✅ Database schema successfully indexed for RAG.")

def retrieve_relevant_schema(query, top_k=3):
    """Retrieves relevant schema information for a query."""
    global db_schema_index, db_schema_chunks
    
    if db_schema_index is None:
        return ""
        
    query_embedding = embedding_model.encode([query])
    distances, indexes = db_schema_index.search(np.array(query_embedding).astype('float32'), top_k)
    
    return "\n\n".join([db_schema_chunks[i] for i in indexes[0] if i < len(db_schema_chunks)])
