# src\main.py

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, simpledialog
from database_utils import prompt_connect_to_database
from document_processor import process_file, save_document_state, load_document_state, set_root as set_doc_root
from query_handler_db import generate_response as generate_db_response, clear_conversation as clear_db_conversation, set_root as set_db_root
from query_handler_doc import generate_response as generate_doc_response, clear_conversation as clear_doc_conversation, set_root as set_doc_query_root  # Distinct alias
from ollama_utils import check_ollama_status
import document_processor
import database_utils

root = tk.Tk()
root.title("Ai Agent - Company Assistant")
root.geometry("800x600")
root.configure(bg="#f5f5f5")

# Set root in query handlers and document processor with distinct names
set_db_root(root)          # For query_handler_db
set_doc_query_root(root)   # For query_handler_doc
set_doc_root(root)         # For document_processor

top_frame = tk.Frame(root, bg="#f5f5f5")
top_frame.pack(fill=tk.X, pady=10, padx=10)

upload_btn = tk.Button(top_frame, text="📂 Upload Document", command=lambda: process_file(chat_window, upload_btn, send_btn, status_label), font=("Arial", 11), bg="#4CAF50", fg="white", padx=10)
upload_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(top_frame, text="🗑️ Clear Chat", command=lambda: handle_clear(chat_window), font=("Arial", 11), bg="#FF5722", fg="white", padx=10)
clear_btn.pack(side=tk.LEFT, padx=5)

refresh_btn = tk.Button(top_frame, text="🔄 Check Ollama", command=lambda: check_ollama_status(status_label), font=("Arial", 11), bg="#2196F3", fg="white", padx=10)
refresh_btn.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(top_frame, text="💾 Save Doc", command=lambda: save_document_state(chat_window), font=("Arial", 11), bg="#9C27B0", fg="white", padx=10)
save_btn.pack(side=tk.LEFT, padx=5)

load_btn = tk.Button(top_frame, text="📤 Load Doc", command=lambda: load_document_state(chat_window, upload_btn, send_btn, status_label), font=("Arial", 11), bg="#9C27B0", fg="white", padx=10)
load_btn.pack(side=tk.LEFT, padx=5)

db_connect_btn = tk.Button(top_frame, text="🗄️ Connect DB", command=lambda: prompt_connect_to_database(chat_window), font=("Arial", 11), bg="#FF9800", fg="white", padx=10)
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
query_entry.bind("<Return>", lambda event: handle_query(chat_window, query_entry, send_btn))

send_btn = tk.Button(input_frame, text="Send", command=lambda: handle_query(chat_window, query_entry, send_btn), font=("Arial", 11, "bold"), bg="#2196F3", fg="white", padx=15)
send_btn.pack(side=tk.RIGHT)

def handle_query(chat_window, query_entry, send_btn):
    """Decides which query handler to use based on document or database state."""
    if document_processor.document_loaded and database_utils.db_connection is not None:
        # Both are active; prompt user to clarify intent (prioritize DB here)
        chat_window.insert(tk.END, "⚠️ Both document and database are loaded. Assuming database query. Use specific terms for document questions.\n")
        generate_db_response(chat_window, query_entry, send_btn)
    elif document_processor.document_loaded:
        generate_doc_response(chat_window, query_entry, send_btn)
    elif database_utils.db_connection is not None:
        generate_db_response(chat_window, query_entry, send_btn)
    else:
        chat_window.insert(tk.END, "⚠️ No document loaded and no database connected! Please upload a document or connect to a database.\n")
        send_btn.config(state=tk.NORMAL)

def handle_clear(chat_window):
    """Clears conversation for both handlers."""
    clear_db_conversation(chat_window)
    clear_doc_conversation(chat_window)

root.after(1000, lambda: check_ollama_status(status_label))
root.mainloop()