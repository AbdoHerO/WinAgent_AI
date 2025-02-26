import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import fitz  # PyMuPDF for PDFs
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import ollama  # For local AI chat via Ollama
import threading

# 1) Embedding model for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Global variables
faiss_index = None
stored_chunks = []
document_name = ""
document_loaded = False
conversation_history = []


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
    send_btn.config(state=tk.DISABLED)

    chat_window.insert(tk.END, f"📂 Loading {document_name}...\n")
    chat_window.update()

    try:
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            text = extract_text_from_excel(file_path)
        else:
            chat_window.insert(tk.END, "⚠️ Unsupported file format. Use PDF or Excel.\n")
            status_label.config(text="No document loaded")
            upload_btn.config(state=tk.NORMAL)
            send_btn.config(state=tk.NORMAL)
            return

        if not text.strip():
            chat_window.insert(tk.END, "⚠️ No text found in the document!\n")
            status_label.config(text="No document loaded")
            upload_btn.config(state=tk.NORMAL)
            send_btn.config(state=tk.NORMAL)
            return

        # Process in a separate thread to avoid UI freezing
        def process_thread():
            create_vector_store(text)

            # Update UI from main thread
            root.after(0, lambda: chat_window.insert(tk.END, f"✅ {document_name} processed successfully!\n\n"))
            root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
            root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
            root.after(0, lambda: send_btn.config(state=tk.NORMAL))

        threading.Thread(target=process_thread).start()

    except Exception as e:
        chat_window.insert(tk.END, f"❌ Error processing file: {str(e)}\n")
        status_label.config(text="Error loading document")
        upload_btn.config(state=tk.NORMAL)
        send_btn.config(state=tk.NORMAL)


# ------------------------------------------------------------------ #
#                          RETRIEVAL                                #
# ------------------------------------------------------------------ #

def search_relevant_chunks(query, top_k=3):
    """Finds the most relevant document chunks for the query."""
    if faiss_index is None:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indexes = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)

    # Get chunks and sort by relevance score (distance)
    results = [(stored_chunks[i], distances[0][j]) for j, i in enumerate(indexes[0]) if i < len(stored_chunks)]
    results.sort(key=lambda x: x[1])  # Sort by distance (lower is better)

    return [chunk for chunk, _ in results]


# ------------------------------------------------------------------ #
#                          OLLAMA CHAT                               #
# ------------------------------------------------------------------ #

def generate_response():
    """Generates a response using Ollama AI."""
    global conversation_history

    user_query = query_entry.get().strip()
    if not user_query:
        return

    # Disable send button during processing
    send_btn.config(state=tk.DISABLED)
    query_entry.delete(0, tk.END)

    chat_window.insert(tk.END, f"\n🧑‍💼 You: {user_query}\n")
    chat_window.update()

    if not document_loaded:
        response = "⚠️ No document loaded! Please upload a PDF or Excel file first."
        chat_window.insert(tk.END, f"🤖 Ai Agent: {response}\n")
        send_btn.config(state=tk.NORMAL)
        return

    # Process in a separate thread to keep UI responsive
    def process_query():
        try:
            # Get relevant document chunks
            retrieved_chunks = search_relevant_chunks(user_query, top_k=3)

            if not retrieved_chunks:
                root.after(0, lambda: chat_window.insert(tk.END,
                                                         "🤖 Ai Agent: I couldn't find relevant information in the document.\n"))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            combined_context = "\n\n".join(retrieved_chunks)

            # Print retrieved context for debugging
            print("\n📄 Retrieved Context for Query:\n", combined_context)

            # Add query to conversation history
            conversation_history.append({"role": "user", "content": user_query})

            # Create a focused prompt based on the query type
            if "name" in user_query.lower():
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

IMPORTANT INSTRUCTIONS:
1. Find all informations in the document
2. Keep it (result) short and clear.
3. If you can't find the info, respond with "Not found"

Answer:"""
            else:
                # Include some conversation history for context
                recent_history = conversation_history[-3:] if len(conversation_history) > 1 else []
                history_text = ""
                if recent_history:
                    history_text = "Recent conversation:\n" + "\n".join(
                        [f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in
                         recent_history[:-1]])

                prompt = f"""You are an AI assistant helping with document questions.

Document context:
{combined_context}

{history_text}

User question: "{user_query}"

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information in the document
2. Be concise and direct
3. If the information isn't in the document, say "I couldn't find that information"
4. DO NOT include phrases like "Based on the document" or "According to the text"

Answer:"""

            # Get response from Ollama
            model_to_use = "llama3" if "llama3" in ollama.list()['models'] else "phi"
            print("Using model:", model_to_use)
            ollama_response = ollama.chat(model=model_to_use, messages=[{"role": "user", "content": prompt}])
            raw_text = ollama_response["message"]["content"].strip()

            # Clean the response
            clean_response = raw_text.replace('\n\n', ' ').strip()

            # Add response to conversation history
            conversation_history.append({"role": "assistant", "content": clean_response})

            # Update UI from main thread
            root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {clean_response}\n"))
            root.after(0, lambda: chat_window.yview(tk.END))

        except Exception as e:
            root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: Error generating response: {str(e)}\n"))
        finally:
            root.after(0, lambda: send_btn.config(state=tk.NORMAL))

    threading.Thread(target=process_query).start()


# ------------------------------------------------------------------ #
#                      CLEAR CONVERSATION                           #
# ------------------------------------------------------------------ #

def clear_conversation():
    """Clears the conversation history and chat window."""
    global conversation_history

    conversation_history = []
    chat_window.delete(1.0, tk.END)
    chat_window.insert(tk.END, "💬 Conversation cleared. Document knowledge is still loaded.\n")


# --------------------------- GUI SETUP --------------------------- #

root = tk.Tk()
root.title("Ai Agent - Company Assistant")
root.geometry("800x600")
root.configure(bg="#f5f5f5")

# Top frame for buttons and status
top_frame = tk.Frame(root, bg="#f5f5f5")
top_frame.pack(fill=tk.X, pady=10, padx=10)

# Upload Button
upload_btn = tk.Button(
    top_frame,
    text="📂 Upload Document",
    command=process_file,
    font=("Arial", 11),
    bg="#4CAF50",
    fg="white",
    padx=10
)
upload_btn.pack(side=tk.LEFT, padx=5)

# Clear conversation button
clear_btn = tk.Button(
    top_frame,
    text="🗑️ Clear Chat",
    command=clear_conversation,
    font=("Arial", 11),
    bg="#FF5722",
    fg="white",
    padx=10
)
clear_btn.pack(side=tk.LEFT, padx=5)

# Status label
status_label = tk.Label(
    top_frame,
    text="No document loaded",
    font=("Arial", 10, "italic"),
    bg="#f5f5f5"
)
status_label.pack(side=tk.RIGHT, padx=10)

# Chat Window
chat_frame = tk.Frame(root, bg="#f5f5f5")
chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

chat_window = scrolledtext.ScrolledText(
    chat_frame,
    wrap=tk.WORD,
    font=("Arial", 12),
    bg="white"
)
chat_window.pack(fill=tk.BOTH, expand=True)
chat_window.insert(tk.END, "💬 Welcome to Ai Agent! Upload a document to start chatting.\n")

# Input Frame
input_frame = tk.Frame(root, bg="#f5f5f5")
input_frame.pack(fill=tk.X, pady=10, padx=10)

# Input Field
query_entry = tk.Entry(input_frame, font=("Arial", 12))
query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
query_entry.bind("<Return>", lambda event: generate_response())

# Send Button
send_btn = tk.Button(
    input_frame,
    text="💬 Send",
    command=generate_response,
    font=("Arial", 12),
    bg="#2196F3",
    fg="white",
    padx=15
)
send_btn.pack(side=tk.RIGHT)

# Run GUI
root.mainloop()
