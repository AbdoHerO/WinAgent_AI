##### ai_agent_db.py #####

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, simpledialog
import fitz  # PyMuPDF for PDFs
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import ollama  # For local AI chat via Ollama
import threading
import json
import time
import psycopg2  # For PostgreSQL database connection
from synonyms_seham_winoffre import TABLE_SYNONYMS, COLUMN_SYNONYMS

# 1) Embedding model for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Global variables
faiss_index = None
stored_chunks = []
document_name = ""
document_loaded = False
conversation_history = []
ollama_available = True  # Flag to track if Ollama is available
full_document_text = ""  # Store the complete document text
db_connection = None  # Database connection object
db_schema = None  # Store the database schema

# 3) Database connection and schema retrieval
def connect_to_database(dbname, user, password, host, port):
    global db_connection, db_schema
    try:
        db_connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Database connection established.")
        # Fetch the database schema
        db_schema = fetch_database_schema()
        chat_window.insert(tk.END, "✅ Connected to database and schema loaded.\n")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        db_connection = None
        db_schema = None
        chat_window.insert(tk.END, f"❌ Error connecting to database: {e}\n")

def fetch_database_schema():
    """Fetches the schema (tables and columns) from the database."""
    if db_connection is None:
        return None
    try:
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, column_name;
        """)
        schema = {}
        for table_name, column_name, data_type in cursor.fetchall():
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append({"column": column_name, "type": data_type})
        cursor.close()
        print("Database schema fetched:", schema)
        return schema
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return None

def prompt_connect_to_database():
    """Prompt user for database credentials via GUI."""
    dbname = simpledialog.askstring("Database Name", "Enter database name:", initialvalue="winOffre_AiAgent")
    user = simpledialog.askstring("Username", "Enter username:", initialvalue="postgres")
    password = simpledialog.askstring("Password", "Enter password:", initialvalue="postgres")
    host = simpledialog.askstring("Host", "Enter host (e.g., localhost):", initialvalue="localhost")
    port = simpledialog.askstring("Port", "Enter port (e.g., 5432):", initialvalue="5432")
    if dbname and user and password and host and port:
        connect_to_database(dbname, user, password, host, port)

# Connect to the database on startup (you can replace with your default credentials)
# connect_to_database("your_dbname", "your_username", "your_password", "your_host", "your_port")

# ------------------------------------------------------------------ #
#                           TEXT EXTRACTION                          #
# ------------------------------------------------------------------ #

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        total_pages = len(doc)
        if total_pages > 10:
            chat_window.insert(tk.END, f"Processing PDF with {total_pages} pages...\n")
            root.update()
        for i, page in enumerate(doc):
            if total_pages > 50 and i % 10 == 0:
                chat_window.insert(tk.END, f"Processing page {i + 1}/{total_pages}...\n")
                root.update()
            full_text += page.get_text() + "\n\n"
        return full_text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def extract_text_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        return df.to_string()
    except Exception as e:
        print(f"Error extracting text from Excel: {e}")
        raise

# ------------------------------------------------------------------ #
#                           CHUNKING                                 #
# ------------------------------------------------------------------ #

def chunk_text(text, max_chunk_size=300, overlap=50):
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

# ------------------------------------------------------------------ #
#                          FAISS INDEX                               #
# ------------------------------------------------------------------ #

def create_vector_store(text_data):
    global faiss_index, stored_chunks, document_loaded, full_document_text
    full_document_text = text_data
    chunks = chunk_text(text_data, max_chunk_size=300)
    stored_chunks = chunks
    chat_window.insert(tk.END, f"Creating {len(chunks)} document chunks for indexing...\n")
    root.update()
    chat_window.insert(tk.END, "Creating document embeddings...\n")
    root.update()
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype('float32'))
    document_loaded = True
    chat_window.insert(tk.END, f"📄 Successfully indexed {len(chunks)} chunks from document.\n")

# ------------------------------------------------------------------ #
#                       FILE PROCESSING                              #
# ------------------------------------------------------------------ #

def process_file():
    global document_name, document_loaded, full_document_text
    file_path = filedialog.askopenfilename(filetypes=[("PDF & Excel Files", "*.pdf;*.xlsx;*.xls")])
    if not file_path:
        return
    document_name = os.path.basename(file_path)
    status_label.config(text=f"Processing: {document_name}")
    root.update()
    upload_btn.config(state=tk.DISABLED)
    send_btn.config(state=tk.DISABLED)
    chat_window.insert(tk.END, f"📂 Loading {document_name}...\n")
    chat_window.update()

    def process_thread():
        try:
            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                text = extract_text_from_excel(file_path)
            else:
                root.after(0, lambda: chat_window.insert(tk.END, "⚠️ Unsupported file format. Use PDF or Excel.\n"))
                root.after(0, lambda: status_label.config(text="No document loaded"))
                root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return
            if not text.strip():
                root.after(0, lambda: chat_window.insert(tk.END, "⚠️ No text found in the document!\n"))
                root.after(0, lambda: status_label.config(text="No document loaded"))
                root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return
            full_document_text = text
            create_vector_store(text)
            root.after(0, lambda: chat_window.insert(tk.END, f"✅ {document_name} processed successfully!\n\n"))
            root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
        except Exception as e:
            error_msg = str(e)
            root.after(0, lambda: chat_window.insert(tk.END, f"❌ Error processing document: {error_msg}\n"))
            root.after(0, lambda: status_label.config(text="Error processing document"))
        finally:
            root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
            root.after(0, lambda: send_btn.config(state=tk.NORMAL))

    threading.Thread(target=process_thread).start()

# ------------------------------------------------------------------ #
#                          RETRIEVAL                                #
# ------------------------------------------------------------------ #

def search_relevant_chunks(query, top_k=3):
    if faiss_index is None:
        return []
    query_embedding = embedding_model.encode([query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    results = [(stored_chunks[i], distances[0][j]) for j, i in enumerate(indexes[0]) if i < len(stored_chunks)]
    results.sort(key=lambda x: x[1])
    return [chunk for chunk, _ in results]

# ------------------------------------------------------------------ #
#                     MODEL SELECTION                               #
# ------------------------------------------------------------------ #

def check_ollama_availability():
    global ollama_available
    try:
        client = ollama.Client(host="http://localhost:11434")
        client.list()
        ollama_available = True
        print("Ollama is available.")
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        ollama_available = False
        return False

def get_best_available_model():
    global ollama_available
    if not check_ollama_availability():
        print("Ollama availability check failed.")
        return None
    try:
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
        available_models = []
        if hasattr(models_response, 'models'):
            available_models = [m.model for m in models_response.models if hasattr(m, 'model')]
        elif isinstance(models_response, dict) and 'models' in models_response:
            available_models = [m.get('model', '') for m in models_response['models']]
        print("Available models:", available_models)
        if not available_models:
            print("No models found in response.")
            return None
        preferred_models = ["llama3", "mistral", "phi"]
        for model in preferred_models:
            matching_models = [m for m in available_models if model in m]
            if matching_models:
                print(f"Selected model: {matching_models[0]}")
                return matching_models[0]
        if available_models:
            print(f"Selected first available model: {available_models[0]}")
            return available_models[0]
        print("No suitable models found.")
        return None
    except Exception as e:
        print(f"Error selecting model: {e}")
        ollama_available = False
        return None

# ------------------------------------------------------------------ #
#                     DATABASE QUERY HANDLING                      #
# ------------------------------------------------------------------ #

def fetch_full_database_schema():
    """Fetches all tables, columns, and relationships from PostgreSQL."""
    if db_connection is None:
        return None
    try:
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT c.table_name, c.column_name, c.data_type, 
                   tc.constraint_type, kcu.column_name AS fk_column,
                   ccu.table_name AS referenced_table, ccu.column_name AS referenced_column
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
                ON c.table_name = kcu.table_name AND c.column_name = kcu.column_name
            LEFT JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            LEFT JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE c.table_schema = 'public'
            ORDER BY c.table_name, c.ordinal_position;
        """)
        schema = {}
        for row in cursor.fetchall():
            table_name, column_name, data_type, constraint_type, fk_column, referenced_table, referenced_column = row
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append({
                "column": column_name,
                "type": data_type,
                "constraint": constraint_type,
                "fk_column": fk_column,
                "referenced_table": referenced_table,
                "referenced_column": referenced_column
            })
        cursor.close()
        return schema
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return None


def train_database_embeddings():
    global schema_embeddings, faiss_index
    schema = fetch_full_database_schema()
    if not schema:
        print("❌ Failed to fetch database schema!")
        return
    
    schema_text = "\n".join([
        f"Table: {table} \n  Columns: {', '.join([col['column'] for col in columns])}"
        for table, columns in schema.items()
    ])

    # Create embeddings
    schema_chunks = chunk_text(schema_text, max_chunk_size=512)
    schema_embeddings = embedding_model.encode(schema_chunks)

    # Store in FAISS index
    faiss_index = faiss.IndexFlatL2(schema_embeddings.shape[1])
    faiss_index.add(np.array(schema_embeddings).astype('float32'))
    print("✅ Database schema successfully indexed for AI training.")



def generate_sql_query_rag(user_query):
    """Generates SQL using Retrieval-Augmented Generation (RAG)."""
    global faiss_index, schema_embeddings
    
    # Retrieve the most relevant schema parts
    query_embedding = embedding_model.encode([user_query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), 3)
    relevant_schema = "\n\n".join([stored_chunks[i] for i in indexes[0] if i < len(stored_chunks)])
    
    prompt = f"""You are a SQL expert trained on this PostgreSQL database schema:
    
{relevant_schema}

User question: "{user_query}"

Generate a **valid** SQL query based on the schema above. 

CRITICAL RULES:
1. Use only the table and column names provided.
2. If a column name is missing, return "Unknown column".
3. If a join is required, match foreign keys.
4. If no valid query can be generated, return "Invalid query."

SQL Query:
"""
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    sql_query = response["message"]["content"].strip()
    
    # Validate response
    if "Invalid query" in sql_query:
        return None
    return sql_query


def store_successful_query(user_query, sql_query):
    """Stores successful queries in a log file."""
    with open("query_log.json", "a", encoding="utf-8") as file:
        json.dump({"user_query": user_query, "sql_query": sql_query}, file)
        file.write("\n")

def retrieve_past_query(user_query):
    """Checks if a similar query was previously executed."""
    try:
        with open("query_log.json", "r", encoding="utf-8") as file:
            queries = [json.loads(line) for line in file]
        for q in queries:
            if q["user_query"] == user_query:
                return q["sql_query"]
    except FileNotFoundError:
        return None
    return None


def is_database_query(query):
    """Determines if the query is likely database-related."""
    db_keywords = ["how many", "total", "sum", "count", "user", "customer", "order", "sales", "turnover", "amount", "what is", "client", "commande"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in db_keywords) and db_connection is not None

def reconnect_database():
    global db_connection
    try:
        db_connection.close()
    except:
        pass
    prompt_connect_to_database()
    
    
def execute_database_query(sql_query):
    """Executes SQL query and auto-reconnects if needed."""
    global db_connection
    if db_connection is None:
        reconnect_database()

    try:
        with db_connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
        return "\n".join(
            [f"{', '.join([f'{col}: {val}' for col, val in zip(column_names, row)])}" for row in result]
        ) if result else "No data found."
    except psycopg2.OperationalError:
        print("⚠️ Connection lost. Reconnecting...")
        reconnect_database()
        return execute_database_query(sql_query)
    except Exception as e:
        return f"Error executing query: {e}"

def has_placeholders(sql_query: str) -> bool:
    # Return True if the model included obvious placeholders
    # You can expand this check as needed
    placeholders = ["WHERE ...", "AND ...", "-- missing condition", "..."]
    for ph in placeholders:
        if ph in sql_query:
            return True
    return False

def generate_sql_query(user_query):
    """Uses LLaMA to generate an SQL query dynamically based on the database schema."""
    if db_schema is None:
        return None
    
    schema_text = "Database Schema:\n"
    for table, columns in db_schema.items():
        schema_text += f"Table: {table}\n  Columns: {', '.join([f'{col['column']} ({col['type']})' for col in columns])}\n"
        
    synonyms_text = "Synonyms:\n"
    synonyms_text += "Table synonyms:\n"
    for k, v in TABLE_SYNONYMS.items():
        synonyms_text += f'  "{k}" => "{v}"\n'

    synonyms_text += "\nColumn synonyms:\n"
    for k, v in COLUMN_SYNONYMS.items():
        synonyms_text += f'  "{k}" => "{v}"\n'

    prompt = f"""You are an AI assistant that generates SQL queries for a PostgreSQL database based on natural language questions from non-technical users who do not know the database schema.
    
{schema_text}

{synonyms_text}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Generate a valid PostgreSQL SQL query using ONLY the tables and columns listed in the schema above.
2. Do NOT assume the existence of tables or columns not explicitly listed (e.g., avoid inventing 'date_commande').
3. Map terms intelligently to schema tables and columns:
   - 'commande', 'order', or 'commandes' refers to 'entete_commande_marche' (order headers) for overall order data or 'detail_commande_marche' (order details) for line-item data.
   - 'client' or 'customer' refers to 'societe' (where 'nom_responsable' is found) or 'acces_client'.
   - 'nombre total' or 'total number' means COUNT(*) for counting orders, not summing values, unless explicitly stated.
   - For 'total' or 'amount' of orders (monetary value), use 'valeur_cmd_net_ttc' from 'entete_commande_marche' unless line-item totals are requested (then use 'total_net_ttc' from 'detail_commande_marche').
   - For date filters (e.g., 'pendant l'année 2023' or 'in 2023'):
     - Use 'date_creation' (timestamp) from 'entete_commande_marche' to filter by the year an order was created.
     - Use EXTRACT(YEAR FROM date_creation) = <year> for year-based conditions.
4. Use joins ONLY when necessary, based on explicit relationships:
   - 'entete_commande_marche' joins 'societe' via 'client_id = societe.id' for client details like 'nom_responsable'.
   - 'entete_commande_marche' joins 'detail_commande_marche' via 'commande_id = entete_commande_marche.id' for order details.
   - Avoid unrelated tables (e.g., 'service_client') unless required.
   - produit is linked to commandes via detail_commande_marche (NOT directly in entete_commande_marche)
   - Use detail_commande_marche.commande_id = entete_commande_marche.id
   - Use detail_commande_marche.produit_id = produit.id
   - DO NOT assume produit_id exists in entete_commande_marche. It must come from detail_commande_marche.
5. If the query mentions 'par client' or 'by client', group results by 'client_id' from 'entete_commande_marche' and join with 'societe' if client details are needed.
6. Keep the query as simple as possible while answering accurately.
7. Return JUST the SQL query as plain text, without explanations or additional text.
8. If the query cannot be generated due to insufficient information or ambiguity, return "Unable to generate SQL query."
9. Do NOT use ellipses (...) or placeholders in the query (like 'WHERE ...'); either provide the full condition or return "Unable to generate SQL query."


SQL Query:"""

    try:
        client = ollama.Client(host="http://localhost:11434")
        model_to_use = "llama3:latest"
        response = client.chat(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        sql_query = response["message"]["content"].strip()
        sql_query = clean_sql_response(sql_query)

        # Check for placeholders
        if has_placeholders(sql_query):
            print("Model returned incomplete query with placeholders.")
            return None

        return sql_query if sql_query != "Unable to generate SQL query" else None
    except Exception as e:
        print(f"Error generating SQL query with Ollama: {e}")
        return None
    
# ------------------------------------------------------------------ #
#                     FALLBACK RESPONSE                            #
# ------------------------------------------------------------------ #

def generate_fallback_response(query, context):
    query_lower = query.lower()
    if "name" in query_lower or "who" in query_lower:
        name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', context)
        return name_match.group(1) if name_match else "Could not find a name in the document."
    elif "date" in query_lower or "when" in query_lower or "time" in query_lower:
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}, \d{4}|\d{1,2} \w+ \d{4})\b', context)
        return date_match.group(1) if date_match else "Could not find a date in the document."
    elif "email" in query_lower or "contact" in query_lower:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', context)
        return email_match.group(0) if email_match else "Could not find email information in the document."
    else:
        return "I found some relevant information but cannot process it without Ollama."

# ------------------------------------------------------------------ #
#                          OLLAMA CHAT                             #
# ------------------------------------------------------------------ #

def preprocess_user_query(user_query: str) -> str:
    """
    Replace synonyms in the user's query with actual table (and optionally column) names.
    """
    # 1) Replace table synonyms
    for synonym, actual_table_name in TABLE_SYNONYMS.items():
        pattern = rf"\b{re.escape(synonym)}\b"  # match whole word
        user_query = re.sub(pattern, actual_table_name, user_query, flags=re.IGNORECASE)

    # 2) Replace column synonyms, if you want to handle columns similarly
    for synonym, actual_column_name in COLUMN_SYNONYMS.items():
        pattern = rf"\b{re.escape(synonym)}\b"
        user_query = re.sub(pattern, actual_column_name, user_query, flags=re.IGNORECASE)

    return user_query

def clean_sql_response(sql_query: str) -> str:
    """
    Removes any extraneous text after the final semicolon
    (or if there's no semicolon, returns the entire text).
    """
    # Split on semicolon
    parts = sql_query.split(";")
    if len(parts) > 1:
        # Keep everything before the last semicolon, then add the semicolon back
        main_sql = ";".join(parts[:-1]) + ";"
        return main_sql.strip()
    else:
        # No semicolon found, just return as-is (though we prefer queries to end with ;)
        return sql_query.strip()

def generate_response():
    global conversation_history, ollama_available
    user_query = query_entry.get().strip()
    if not user_query:
        return
    send_btn.config(state=tk.DISABLED)
    query_entry.delete(0, tk.END)
    chat_window.insert(tk.END, f"\n🧑‍💼 You: {user_query}\n")
    chat_window.update()

    def process_query():
        try:
            # Handle greetings or simple non-queries
            if user_query.lower() in ["hi", "hello", "hey"]:
                response = "Hello! How can I assist you with the database or document today?"
            # Check if it's a database query
            elif db_connection is not None and is_database_query(user_query):
                # Preprocess the user's query to replace synonyms
                processed_query = preprocess_user_query(user_query)
                sql_query = generate_sql_query(processed_query)
                
                if sql_query:
                    print(f"Executing SQL Query: {sql_query}")  # Log before execution
                    result = execute_database_query(sql_query)
                    response = result
                else:
                    response = "I'm sorry, I couldn't generate a valid SQL query for your question. Please try rephrasing it."
            # Check if a document is loaded for document-based queries
            elif document_loaded:
                retrieved_chunks = search_relevant_chunks(user_query, top_k=3)
                if not retrieved_chunks:
                    response = "I couldn't find relevant information in the document."
                else:
                    combined_context = "\n\n".join(retrieved_chunks)
                    conversation_history.append({"role": "user", "content": user_query})
                    if not check_ollama_availability():
                        fallback_response = generate_fallback_response(user_query, combined_context)
                        conversation_history.append({"role": "assistant", "content": fallback_response})
                        response = f"{fallback_response}\n\n⚠️ Note: Ollama is not available."
                    else:
                        model_to_use = get_best_available_model()
                        if not model_to_use:
                            fallback_response = generate_fallback_response(user_query, combined_context)
                            conversation_history.append({"role": "assistant", "content": fallback_response})
                            response = f"{fallback_response}\n\n⚠️ Note: No suitable Ollama models found."
                        else:
                            print("Using model:", model_to_use)
                            question_type = "general"
                            if any(word in user_query.lower() for word in ["name", "who", "person"]):
                                question_type = "name"

                            if question_type == "name":
                                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the full name of the person from the document
2. Return JUST the name with no additional text or explanation
3. Format your response as a single line with just the name
4. If multiple names exist, return the main person's name (likely the CV owner)
5. If you can't find the name with certainty, respond with ONLY "Name not found"
6. DO NOT include any of the document context in your response
7. DO NOT include phrases like "Based on the document" or "According to the text"

Answer:"""
                            else:
                                recent_history = conversation_history[-3:] if len(conversation_history) > 1 else []
                                history_text = ""
                                if recent_history:
                                    history_text = "Recent conversation:\n" + "\n".join(
                                        [f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in recent_history[:-1]])

                                prompt = f"""You are an AI assistant helping with document questions.

Document context:
{combined_context}

{history_text}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Answer based ONLY on the information in the document
2. Be extremely direct and concise - use 25 words or less if possible
3. DO NOT include phrases like "Based on the document" or "According to the text"
4. DO NOT provide explanations unless specifically asked
5. If the answer is a list, use bullet points
6. DO NOT include any of the document context in your response
7. If the information isn't in the document, respond with ONLY "Information not found"

Answer:"""

                            client = ollama.Client(host="http://localhost:11434")
                            ollama_response = client.chat(
                                model=model_to_use,
                                messages=[{"role": "user", "content": prompt}],
                                options={"temperature": 0.2}
                            )
                            raw_text = ollama_response["message"]["content"].strip()
                            clean_response = re.sub(
                                r'^(Answer:|Based on the document|According to the text|The document states that|Document context:)',
                                '', raw_text, flags=re.IGNORECASE).strip()
                            if "Document context:" in clean_response:
                                clean_response = clean_response.split("Document context:")[0].strip()
                            conversation_history.append({"role": "assistant", "content": clean_response})
                            response = clean_response
            # Fallback if neither database nor document is loaded
            else:
                response = "⚠️ No document loaded and no database connected! Please upload a document or connect to a database."

            root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {response}\n"))
            root.after(0, lambda: chat_window.yview(tk.END))
        except Exception as process_error:
            error_message = str(process_error)
            print(f"Query processing error: {error_message}")
            root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: Error processing query: {error_message}\n"))
        finally:
            root.after(0, lambda: send_btn.config(state=tk.NORMAL))

    threading.Thread(target=process_query).start()
    
# ------------------------------------------------------------------ #
#                      CLEAR CONVERSATION                          #
# ------------------------------------------------------------------ #

def clear_conversation():
    global conversation_history
    conversation_history = []
    chat_window.delete(1.0, tk.END)
    chat_window.insert(tk.END, "💬 Conversation cleared. Document and database knowledge remain available.\n")

# ------------------------------------------------------------------ #
#                      CHECK OLLAMA STATUS                         #
# ------------------------------------------------------------------ #

def check_ollama_status():
    global ollama_available
    if check_ollama_availability():
        model = get_best_available_model()
        status_label.config(text=f"Ollama ready: {model}" if model else "Ollama ready but no models found")
    else:
        status_label.config(text="⚠️ Ollama not available")
    root.after(30000, check_ollama_status)

# ------------------------------------------------------------------ #
#                      SAVE/LOAD DOCUMENT                          #
# ------------------------------------------------------------------ #

def save_document_state():
    if not document_loaded or not full_document_text:
        messagebox.showwarning("Warning", "No document loaded to save.")
        return
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{document_name.split('.')[0]}_state.json"
        )
        if not file_path:
            return
        state = {"document_name": document_name, "document_text": full_document_text, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        chat_window.insert(tk.END, f"✅ Document state saved to {os.path.basename(file_path)}\n")
    except Exception as e:
        chat_window.insert(tk.END, f"❌ Error saving document state: {str(e)}\n")

def load_document_state():
    global document_name, full_document_text, document_loaded
    try:
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        document_name = state.get("document_name", "Loaded document")
        full_document_text = state.get("document_text", "")
        if not full_document_text:
            messagebox.showerror("Error", "Invalid state file: No document text found.")
            return
        upload_btn.config(state=tk.DISABLED)
        send_btn.config(state=tk.DISABLED)
        chat_window.insert(tk.END, f"📂 Loading saved document state: {document_name}...\n")

        def process_thread():
            try:
                create_vector_store(full_document_text)
                root.after(0, lambda: chat_window.insert(tk.END, f"✅ Document state loaded successfully!\n"))
                root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
            except Exception as e:
                root.after(0, lambda: chat_window.insert(tk.END, f"❌ Error processing loaded document: {str(e)}\n"))
                root.after(0, lambda: status_label.config(text="Error loading document"))
            finally:
                root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))

        threading.Thread(target=process_thread).start()
    except Exception as e:
        chat_window.insert(tk.END, f"❌ Error loading document state: {str(e)}\n")

# --------------------------- GUI SETUP --------------------------- #

root = tk.Tk()
root.title("Ai Agent - Company Assistant")
root.geometry("800x600")
root.configure(bg="#f5f5f5")

# Top frame for buttons and status
top_frame = tk.Frame(root, bg="#f5f5f5")
top_frame.pack(fill=tk.X, pady=10, padx=10)

upload_btn = tk.Button(top_frame, text="📂 Upload Document", command=process_file, font=("Arial", 11), bg="#4CAF50", fg="white", padx=10)
upload_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(top_frame, text="🗑️ Clear Chat", command=clear_conversation, font=("Arial", 11), bg="#FF5722", fg="white", padx=10)
clear_btn.pack(side=tk.LEFT, padx=5)

refresh_btn = tk.Button(top_frame, text="🔄 Check Ollama", command=check_ollama_status, font=("Arial", 11), bg="#2196F3", fg="white", padx=10)
refresh_btn.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(top_frame, text="💾 Save Doc", command=save_document_state, font=("Arial", 11), bg="#9C27B0", fg="white", padx=10)
save_btn.pack(side=tk.LEFT, padx=5)

load_btn = tk.Button(top_frame, text="📤 Load Doc", command=load_document_state, font=("Arial", 11), bg="#9C27B0", fg="white", padx=10)
load_btn.pack(side=tk.LEFT, padx=5)

db_connect_btn = tk.Button(top_frame, text="🗄️ Connect DB", command=prompt_connect_to_database, font=("Arial", 11), bg="#FF9800", fg="white", padx=10)
db_connect_btn.pack(side=tk.LEFT, padx=5)

status_label = tk.Label(top_frame, text="Checking Ollama status...", font=("Arial", 10, "italic"), bg="#f5f5f5")
status_label.pack(side=tk.RIGHT, padx=10)

# Chat Window
chat_frame = tk.Frame(root, bg="#f5f5f5")
chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Arial", 12), bg="white")
chat_window.pack(fill=tk.BOTH, expand=True)
chat_window.insert(tk.END, "💬 Welcome to Ai Agent! Upload a document or connect to a database to start.\n")

# Input Frame
input_frame = tk.Frame(root, bg="#f5f5f5")
input_frame.pack(fill=tk.X, pady=10, padx=10)

query_entry = tk.Entry(input_frame, font=("Arial", 12), bd=1, relief=tk.SOLID)
query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
query_entry.bind("<Return>", lambda event: generate_response())

send_btn = tk.Button(input_frame, text="Send", command=generate_response, font=("Arial", 11, "bold"), bg="#2196F3", fg="white", padx=15)
send_btn.pack(side=tk.RIGHT)

# Check Ollama status on startup
root.after(1000, check_ollama_status)

# Start the GUI
root.mainloop()