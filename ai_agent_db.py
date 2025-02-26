import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
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
import sqlite3  # For database connection
import mysql.connector  # For MySQL connection
import configparser  # For database config

# 1) Embedding model for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Global variables
faiss_index = None
stored_chunks = []
document_name = ""
document_loaded = False
conversation_history = []
db_connection = None
db_type = None  # "sqlite" or "mysql"


# ------------------------------------------------------------------ #
#                       DATABASE FUNCTIONS                          #
# ------------------------------------------------------------------ #

def load_db_config():
    """Load database configuration from config file"""
    config = configparser.ConfigParser()

    # Check if config file exists, if not create a default one
    if not os.path.exists('db_config.ini'):
        config['DEFAULT'] = {
            'db_type': 'sqlite',
            'sqlite_db': 'company_data.db',
            'mysql_host': 'localhost',
            'mysql_user': 'root',
            'mysql_password': '',
            'mysql_database': 'company_data'
        }
        with open('db_config.ini', 'w') as configfile:
            config.write(configfile)
        return config['DEFAULT']

    config.read('db_config.ini')
    return config['DEFAULT']


def connect_to_database():
    """Connect to the database based on configuration"""
    global db_connection, db_type

    config = load_db_config()
    db_type = config['db_type']

    try:
        if db_type == 'sqlite':
            db_path = config['sqlite_db']
            db_connection = sqlite3.connect(db_path)
            chat_window.insert(tk.END, f"✅ Connected to SQLite database: {db_path}\n")
        elif db_type == 'mysql':
            db_connection = mysql.connector.connect(
                host=config['mysql_host'],
                user=config['mysql_user'],
                password=config['mysql_password'],
                database=config['mysql_database']
            )
            chat_window.insert(tk.END, f"✅ Connected to MySQL database: {config['mysql_database']}\n")
        else:
            chat_window.insert(tk.END, "❌ Invalid database type in config. Using SQLite as default.\n")
            db_type = 'sqlite'
            db_connection = sqlite3.connect('company_data.db')

        return True
    except Exception as e:
        chat_window.insert(tk.END, f"❌ Database connection error: {str(e)}\n")
        return False


def execute_query(query, params=None, fetch=True):
    """Execute a SQL query and return results"""
    global db_connection

    if db_connection is None:
        if not connect_to_database():
            return None

    try:
        cursor = db_connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetch:
            if db_type == 'sqlite':
                # Get column names from cursor description
                columns = [description[0] for description in cursor.description]
                # Fetch all rows
                rows = cursor.fetchall()
                # Convert to list of dictionaries
                results = [dict(zip(columns, row)) for row in rows]
                return results
            else:  # MySQL
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                return results
        else:
            db_connection.commit()
            return {"affected_rows": cursor.rowcount}
    except Exception as e:
        print(f"Query execution error: {str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()


def get_table_schema():
    """Get database schema information for context"""
    if db_connection is None:
        if not connect_to_database():
            return "No database connected"

    schema_info = []

    try:
        if db_type == 'sqlite':
            # Get list of tables
            tables = execute_query("SELECT name FROM sqlite_master WHERE type='table';")

            for table in tables:
                table_name = table['name']
                # Skip SQLite system tables
                if table_name.startswith('sqlite_'):
                    continue

                # Get columns for this table
                columns = execute_query(f"PRAGMA table_info({table_name});")
                column_info = [f"{col['name']} ({col['type']})" for col in columns]

                schema_info.append(f"Table: {table_name}")
                schema_info.append("Columns: " + ", ".join(column_info))
                schema_info.append("")  # Empty line for readability

        else:  # MySQL
            # Get database name
            db_name = execute_query("SELECT DATABASE();")[0]['DATABASE()']

            # Get list of tables
            tables = execute_query(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}';")

            for table in tables:
                table_name = table['table_name']
                # Get columns for this table
                columns = execute_query(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = '{db_name}' AND table_name = '{table_name}';
                """)

                column_info = [f"{col['column_name']} ({col['data_type']})" for col in columns]

                schema_info.append(f"Table: {table_name}")
                schema_info.append("Columns: " + ", ".join(column_info))
                schema_info.append("")  # Empty line for readability

        return "\n".join(schema_info)
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


def detect_sql_intent(query):
    """Detect if the user query is likely a database query"""
    db_keywords = [
        'database', 'sql', 'query', 'table', 'record', 'select', 'from', 'where',
        'order', 'group by', 'join', 'insert', 'update', 'delete', 'count',
        'how many', 'list all', 'show me', 'find', 'search for',
        'orders', 'customers', 'users', 'products', 'sales', 'revenue',
        'total', 'average', 'sum', 'maximum', 'minimum'
    ]

    query_lower = query.lower()

    # Check if query contains database keywords
    if any(keyword in query_lower for keyword in db_keywords):
        return True

    # Check for question patterns about data
    data_question_patterns = [
        r'how many .+ (are there|do we have)',
        r'what is the (total|sum|average|count) of',
        r'who (has|have|is|are) the (most|highest|lowest)',
        r'list all .+ (from|in) the (database|system)',
        r'show me .+ (data|information|records)',
        r'find .+ (in|from) (the|our) (database|system|records)'
    ]

    for pattern in data_question_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


def generate_sql_from_question(question, schema_info):
    """Generate SQL query from natural language question"""
    # Get the best available model
    model_to_use = get_best_available_model()

    prompt = f"""You are a SQL expert assistant that converts natural language questions into SQL queries.

DATABASE SCHEMA:
{schema_info}

USER QUESTION: "{question}"

CRITICAL INSTRUCTIONS:
1. Generate ONLY the SQL query that answers the question, nothing else
2. Make sure the query is valid SQL that matches the schema
3. Use proper table and column names from the schema
4. For aggregations, use COUNT(), SUM(), AVG(), etc. as appropriate
5. Include appropriate JOINs if needed to connect tables
6. If the question cannot be answered with the given schema, respond with "Cannot generate SQL from this question with the available schema"
7. DO NOT include any explanations, only return the SQL query

SQL QUERY:"""

    try:
        # Get response from Ollama with lower temperature for more precise SQL
        ollama_response = ollama.chat(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}  # Very low temperature for precise SQL
        )

        sql_query = ollama_response["message"]["content"].strip()

        # Clean the response
        sql_query = re.sub(r'^(SQL QUERY:|```sql|```)', '', sql_query, flags=re.IGNORECASE).strip()
        sql_query = re.sub(r'```$', '', sql_query).strip()

        # Validate if it looks like SQL
        if sql_query.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
            return sql_query
        else:
            return None
    except Exception as e:
        print(f"Error generating SQL: {str(e)}")
        return None


def execute_and_format_results(sql_query):
    """Execute SQL query and format results for display"""
    try:
        results = execute_query(sql_query)

        if not results:
            return "No results found."

        if isinstance(results, dict) and 'affected_rows' in results:
            return f"Query executed successfully. {results['affected_rows']} rows affected."

        # Format results as a table
        if len(results) == 1:
            # Single row result
            result_str = []
            for key, value in results[0].items():
                result_str.append(f"{key}: {value}")
            return "\n".join(result_str)
        else:
            # Multiple rows
            # Get all keys from all dictionaries
            all_keys = set()
            for row in results:
                all_keys.update(row.keys())

            # Create header
            header = " | ".join(all_keys)
            separator = "-" * len(header)

            # Create rows
            rows = []
            for row in results:
                row_values = [str(row.get(key, "")) for key in all_keys]
                rows.append(" | ".join(row_values))

            # Combine all parts
            table = [header, separator] + rows
            return "\n".join(table)
    except Exception as e:
        return f"Error executing query: {str(e)}"


def handle_database_query(user_query):
    """Process a database-related query"""
    # Get database schema for context
    schema_info = get_table_schema()

    # Generate SQL from the question
    sql_query = generate_sql_from_question(user_query, schema_info)

    if not sql_query:
        return "I couldn't generate a SQL query from your question. Could you rephrase it?"

    # Log the generated SQL (for debugging)
    print(f"Generated SQL: {sql_query}")

    # Execute the query and format results
    result = execute_and_format_results(sql_query)

    # Format the response
    response = f"Based on your question, I ran this query:\n```sql\n{sql_query}\n```\n\nResults:\n{result}"
    return response


# ------------------------------------------------------------------ #
#                           TEXT EXTRACTION                          #
# ------------------------------------------------------------------ #

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    return full_text.strip()


def extract_text_from_excel(excel_path):
    """Extracts data from an Excel file."""
    df = pd.read_excel(excel_path)
    return df.to_string()


# ------------------------------------------------------------------ #
#                           CHUNKING                                #
# ------------------------------------------------------------------ #

def chunk_text(text, max_chunk_size=300, overlap=50):
    """Splits the text into manageable chunks with overlap for context."""
    paragraphs = re.split(r'\n\s*\n', text)  # Better paragraph splitting
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
            # If current chunk has content, save it
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            # Start new chunk, possibly breaking long paragraphs
            if len(words) > max_chunk_size:
                # Handle very long paragraphs by breaking them
                for i in range(0, len(words), max_chunk_size - overlap):
                    chunk_words = words[i:i + max_chunk_size]
                    chunks.append(" ".join(chunk_words))
                current_chunk = []
                current_length = 0
            else:
                current_chunk = [para]
                current_length = len(words)

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ------------------------------------------------------------------ #
#                          FAISS INDEX                               #
# ------------------------------------------------------------------ #

def create_vector_store(text_data):
    """Embeds and stores document text into FAISS for retrieval."""
    global faiss_index, stored_chunks, document_loaded

    chunks = chunk_text(text_data, max_chunk_size=300)
    stored_chunks = chunks

    embeddings = embedding_model.encode(chunks)

    # Use L2 normalization for better semantic search
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype('float32'))

    document_loaded = True
    chat_window.insert(tk.END, f"📄 Indexed {len(chunks)} chunks from document.\n")


# ------------------------------------------------------------------ #
#                       FILE PROCESSING                              #
# ------------------------------------------------------------------ #

def process_file():
    """Handles file selection and processing."""
    global document_name, document_loaded

    file_path = filedialog.askopenfilename(
        filetypes=[("PDF & Excel Files", "*.pdf;*.xlsx;*.xls")]
    )
    if not file_path:
        return

    document_name = os.path.basename(file_path)

    # Update status
    status_label.config(text=f"Processing: {document_name}")
    root.update()

    # Disable buttons during processing
    upload_btn.config(state=tk.DISABLED)
    sen
