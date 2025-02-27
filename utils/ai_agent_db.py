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
ollama_available = True
full_document_text = ""
db_connection = None
db_schema = None
schema_embeddings = None

# 3) Database connection and schema retrieval
def connect_to_database(dbname, user, password, host, port):
    global db_connection, db_schema, faiss_index, stored_chunks
    try:
        db_connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Database connection established.")
        db_schema = fetch_full_database_schema()
        train_database_embeddings()  # Train embeddings on schema
        chat_window.insert(tk.END, "✅ Connected to database and schema loaded.\n")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        db_connection = None
        db_schema = None
        chat_window.insert(tk.END, f"❌ Error connecting to database: {e}\n")

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

def prompt_connect_to_database():
    """Prompt user for database credentials via GUI."""
    dbname = simpledialog.askstring("Database Name", "Enter database name:", initialvalue="winOffre_AiAgent")
    user = simpledialog.askstring("Username", "Enter username:", initialvalue="postgres")
    password = simpledialog.askstring("Password", "Enter password:", initialvalue="postgres")
    host = simpledialog.askstring("Host", "Enter host (e.g., localhost):", initialvalue="localhost")
    port = simpledialog.askstring("Port", "Enter port (e.g., 5432):", initialvalue="5432")
    if dbname and user and password and host and port:
        connect_to_database(dbname, user, password, host, port)

# 4) Schema Embedding for RAG
def train_database_embeddings():
    global schema_embeddings, faiss_index, stored_chunks
    schema = fetch_full_database_schema()
    if not schema:
        print("❌ Failed to fetch database schema!")
        return
    
    schema_text = "\n".join([
        f"Table: {table}\n  Columns: {', '.join([f'{col['column']} ({col['type']})' for col in columns])}\n  Relationships: {', '.join([f'{col['fk_column']} -> {col['referenced_table']}.{col['referenced_column']}' for col in columns if col['constraint'] == 'FOREIGN KEY']) or 'None'}"
        for table, columns in schema.items()
    ])
    synonyms_text = "\nSynonyms:\n" + "\n".join([f"{k} -> {v}" for k, v in TABLE_SYNONYMS.items()]) + "\n" + "\n".join([f"{k} -> {v}" for k, v in COLUMN_SYNONYMS.items()])
    full_schema_text = schema_text + "\n" + synonyms_text

    stored_chunks = chunk_text(full_schema_text, max_chunk_size=512)
    schema_embeddings = embedding_model.encode(stored_chunks)
    faiss_index = faiss.IndexFlatL2(schema_embeddings.shape[1])
    faiss_index.add(np.array(schema_embeddings).astype('float32'))
    print("✅ Database schema successfully indexed for RAG.")

# 5) Text Extraction and Chunking (unchanged from your code)
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

# 6) File Processing (unchanged)
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

# 7) Retrieval (unchanged)
def search_relevant_chunks(query, top_k=3):
    if faiss_index is None:
        return []
    query_embedding = embedding_model.encode([query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    results = [(stored_chunks[i], distances[0][j]) for j, i in enumerate(indexes[0]) if i < len(stored_chunks)]
    results.sort(key=lambda x: x[1])
    return [chunk for chunk, _ in results]

# 8) Model Selection (unchanged)
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

# 9) Database Query Handling
def is_database_query(query):
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
    global db_connection
    if db_connection is None:
        reconnect_database()
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        if result:
            formatted_result = "\n".join([f"{', '.join([f'{col}: {val}' for col, val in zip(column_names, row)])}" for row in result])
            return formatted_result
        return "No data found for this query."
    except psycopg2.OperationalError:
        print("⚠️ Connection lost. Reconnecting...")
        reconnect_database()
        return execute_database_query(sql_query)
    except Exception as e:
        return f"Error executing query: {e}"

def store_successful_query(user_query, sql_query):
    with open("query_log.json", "a", encoding="utf-8") as file:
        json.dump({"user_query": user_query, "sql_query": sql_query}, file)
        file.write("\n")

def retrieve_past_query(user_query):
    try:
        with open("query_log.json", "r", encoding="utf-8") as file:
            queries = [json.loads(line) for line in file]
        for q in queries:
            if q["user_query"].lower() == user_query.lower():
                return q["sql_query"]
    except FileNotFoundError:
        return None
    return None

def preprocess_user_query(user_query):
    for synonym, actual in TABLE_SYNONYMS.items():
        user_query = re.sub(rf"\b{re.escape(synonym)}\b", actual, user_query, flags=re.IGNORECASE)
    for synonym, actual in COLUMN_SYNONYMS.items():
        user_query = re.sub(rf"\b{re.escape(synonym)}\b", actual, user_query, flags=re.IGNORECASE)
    return user_query

def generate_sql_query(user_query):
    if db_schema is None:
        return None

    # Check cache first
    cached_query = retrieve_past_query(user_query)
    if cached_query:
        print(f"Using cached SQL Query: {cached_query}")
        return cached_query

    # Preprocess query with synonyms
    processed_query = preprocess_user_query(user_query)

    # Retrieve relevant schema chunks using RAG
    query_embedding = embedding_model.encode([processed_query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), 3)
    relevant_schema = "\n\n".join([stored_chunks[i] for i in indexes[0] if i < len(stored_chunks)])

    prompt = f"""You are an AI assistant that generates SQL queries for a PostgreSQL database based on natural language questions from non-technical users.

Relevant Database Schema (subset):
{relevant_schema}

User question: "{processed_query}"

CRITICAL INSTRUCTIONS:
1. Generate a valid PostgreSQL SQL query using ONLY the tables and columns listed in the schema above.
2. Do NOT assume the existence of tables or columns not explicitly listed.
3. Map terms intelligently:
   - 'commande', 'order', or 'commandes' refers to 'entete_commande_marche' (headers) or 'detail_commande_marche' (details).
   - 'client' or 'customer' refers to 'societe' or 'acces_client'.
   - 'nombre total' or 'total number' means COUNT(*); 'total' or 'amount' (money) uses 'valeur_cmd_net_ttc' (entete) or 'total_net_ttc' (detail).
   - Date filters (e.g., '2023') use 'date_creation' with EXTRACT(YEAR FROM date_creation) = <year>.
4. Use joins only when necessary:
   - 'entete_commande_marche' joins 'societe' via 'client_id = societe.id'.
   - 'entete_commande_marche' joins 'detail_commande_marche' via 'commande_id = entete_commande_marche.id'.
5. Group by 'client_id' for 'par client' or 'by client'.
6. Return JUST the SQL query, no explanations.
7. If unable to generate, return "Unable to generate SQL query."

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
        print(f"Generated SQL Query: {sql_query}")
        if sql_query != "Unable to generate SQL query":
            # Validate query (basic check)
            if "..." not in sql_query and "SELECT" in sql_query.upper():
                store_successful_query(user_query, sql_query)
            return sql_query
        return None
    except Exception as e:
        print(f"Error generating SQL query with Ollama: {e}")
        return None

# 10) Chat Handling
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
            if user_query.lower() in ["hi", "hello", "hey"]:
                response = "Hello! How can I assist you today?"
            elif db_connection is not None and is_database_query(user_query):
                sql_query = generate_sql_query(user_query)
                if sql_query:
                    result = execute_database_query(sql_query)
                    response = result
                else:
                    response = "I'm sorry, I couldn't generate a valid SQL query. Try rephrasing your request."
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

# 11) Clear Conversation (unchanged)
def clear_conversation():
    global conversation_history
    conversation_history = []
    chat_window.delete(1.0, tk.END)
    chat_window.insert(tk.END, "💬 Conversation cleared. Document and database knowledge remain available.\n")

# 12) Check Ollama Status (unchanged)
def check_ollama_status():
    global ollama_available
    if check_ollama_availability():
        model = get_best_available_model()
        status_label.config(text=f"Ollama ready: {model}" if model else "Ollama ready but no models found")
    else:
        status_label.config(text="⚠️ Ollama not available")
    root.after(30000, check_ollama_status)

# 13) Save/Load Document (unchanged)
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

# 14) GUI Setup (unchanged)
root = tk.Tk()
root.title("Ai Agent - Company Assistant")
root.geometry("800x600")
root.configure(bg="#f5f5f5")

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

chat_frame = tk.Frame(root, bg="#f5f5f5")
chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Arial", 12), bg="white")
chat_window.pack(fill=tk.BOTH, expand=True)
chat_window.insert(tk.END, "💬 Welcome to Ai Agent! Upload a document or connect to a database to start.\n")

input_frame = tk.Frame(root, bg="#f5f5f5")
input_frame.pack(fill=tk.X, pady=10, padx=10)

query_entry = tk.Entry(input_frame, font=("Arial", 12), bd=1, relief=tk.SOLID)
query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
query_entry.bind("<Return>", lambda event: generate_response())

send_btn = tk.Button(input_frame, text="Send", command=generate_response, font=("Arial", 11, "bold"), bg="#2196F3", fg="white", padx=15)
send_btn.pack(side=tk.RIGHT)

root.after(1000, check_ollama_status)
root.mainloop()