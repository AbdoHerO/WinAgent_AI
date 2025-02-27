# src\document_processor.py

import fitz
import numpy as np
import pandas as pd
import tkinter as tk  # Import tkinter for root access
from tkinter import messagebox, filedialog
import threading
import json
import time
import os
import faiss  # Add this import
from rag_engine import chunk_text, embedding_model, faiss_index

# Global variables
document_name = ""
document_loaded = False
full_document_text = ""
stored_chunks = []
tk_root = None  # Will be set by main.py

def set_root(root):
    """Set the global tk_root variable from main.py."""
    global tk_root
    tk_root = root

def extract_text_from_pdf(pdf_path, chat_window):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        total_pages = len(doc)
        if total_pages > 10:
            chat_window.insert(tk.END, f"Processing PDF with {total_pages} pages...\n")
            tk_root.update()
        for i, page in enumerate(doc):
            if total_pages > 50 and i % 10 == 0:
                chat_window.insert(tk.END, f"Processing page {i + 1}/{total_pages}...\n")
                tk_root.update()
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

def create_vector_store(text_data, chat_window):
    global faiss_index, stored_chunks, document_loaded, full_document_text
    full_document_text = text_data
    chunks = chunk_text(text_data, max_chunk_size=300)
    stored_chunks = chunks
    chat_window.insert(tk.END, f"Creating {len(chunks)} document chunks for indexing...\n")
    tk_root.update()
    chat_window.insert(tk.END, "Creating document embeddings...\n")
    tk_root.update()
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # This line requires the faiss module
    faiss_index.add(np.array(embeddings).astype('float32'))
    document_loaded = True
    chat_window.insert(tk.END, f"📄 Successfully indexed {len(chunks)} chunks from document.\n")

def process_file(chat_window, upload_btn, send_btn, status_label):
    global document_name, document_loaded, full_document_text
    file_path = filedialog.askopenfilename(filetypes=[("PDF & Excel Files", "*.pdf;*.xlsx;*.xls")])
    if not file_path:
        return
    document_name = os.path.basename(file_path)
    status_label.config(text=f"Processing: {document_name}")
    tk_root.update()
    upload_btn.config(state=tk.DISABLED)
    send_btn.config(state=tk.DISABLED)
    chat_window.insert(tk.END, f"📂 Loading {document_name}...\n")
    chat_window.update()

    def process_thread():
        try:
            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path, chat_window)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                text = extract_text_from_excel(file_path)
            else:
                tk_root.after(0, lambda: chat_window.insert(tk.END, "⚠️ Unsupported file format. Use PDF or Excel.\n"))
                tk_root.after(0, lambda: status_label.config(text="No document loaded"))
                tk_root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return
            if not text.strip():
                tk_root.after(0, lambda: chat_window.insert(tk.END, "⚠️ No text found in the document!\n"))
                tk_root.after(0, lambda: status_label.config(text="No document loaded"))
                tk_root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return
            full_document_text = text
            create_vector_store(text, chat_window)
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"✅ {document_name} processed successfully!\n\n"))
            tk_root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
        except Exception as e:
            error_msg = str(e)
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"❌ Error processing document: {error_msg}\n"))
            tk_root.after(0, lambda: status_label.config(text="Error processing document"))
        finally:
            tk_root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
            tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))

    threading.Thread(target=process_thread).start()

def save_document_state(chat_window):
    global document_loaded, full_document_text, document_name
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

def load_document_state(chat_window, upload_btn, send_btn, status_label):
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
                create_vector_store(full_document_text, chat_window)
                tk_root.after(0, lambda: chat_window.insert(tk.END, f"✅ Document state loaded successfully!\n"))
                tk_root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
            except Exception as e:
                tk_root.after(0, lambda: chat_window.insert(tk.END, f"❌ Error processing loaded document: {str(e)}\n"))
                tk_root.after(0, lambda: status_label.config(text="Error loading document"))
            finally:
                tk_root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))

        threading.Thread(target=process_thread).start()
    except Exception as e:
        chat_window.insert(tk.END, f"❌ Error loading document state: {str(e)}\n")