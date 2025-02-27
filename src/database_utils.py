# src\database_utils.py

import psycopg2
import tkinter as tk
from rag_engine import train_database_embeddings
from tkinter import simpledialog

# Global variables (shared via imports)
db_connection = None
db_schema = None

def connect_to_database(dbname, user, password, host, port, chat_window):
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
        db_schema = fetch_full_database_schema()
        train_database_embeddings(db_schema)  # Train RAG embeddings
        chat_window.insert(tk.END, "✅ Connected to database and schema loaded.\n")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        db_connection = None
        db_schema = None
        chat_window.insert(tk.END, f"❌ Error connecting to database: {e}\n")

def fetch_full_database_schema():
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

def prompt_connect_to_database(chat_window):
    dbname = simpledialog.askstring("Database Name", "Enter database name:", initialvalue="winOffre_AiAgent")
    user = simpledialog.askstring("Username", "Enter username:", initialvalue="postgres")
    password = simpledialog.askstring("Password", "Enter password:", initialvalue="postgres")
    host = simpledialog.askstring("Host", "Enter host (e.g., localhost):", initialvalue="localhost")
    port = simpledialog.askstring("Port", "Enter port (e.g., 5432):", initialvalue="5432")
    if dbname and user and password and host and port:
        connect_to_database(dbname, user, password, host, port, chat_window)

def reconnect_database(chat_window):
    global db_connection
    try:
        db_connection.close()
    except:
        pass
    prompt_connect_to_database(chat_window)

def execute_database_query(sql_query, chat_window):
    """Executes the SQL query and returns the result."""
    global db_connection
    if db_connection is None:
        reconnect_database(chat_window)
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
        reconnect_database(chat_window)
        return execute_database_query(sql_query, chat_window)  # Pass chat_window here too
    except Exception as e:
        return f"Error executing query: {e}"