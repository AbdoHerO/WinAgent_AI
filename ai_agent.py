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
import json
import time

# 1) Embedding model for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Global variables
faiss_index = None
stored_chunks = []
document_name = ""
document_loaded = False
conversation_history = []
ollama_available = True  # Flag to track if Ollama is available


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
            try:
                create_vector_store(text)
                # Update UI from main thread
                root.after(0, lambda: chat_window.insert(tk.END, f"✅ {document_name} processed successfully!\n\n"))
                root.after(0, lambda: status_label.config(text=f"Active document: {document_name}"))
            except Exception as e:
                # Handle exceptions in the thread
                error_msg = str(e)
                root.after(0, lambda: chat_window.insert(tk.END, f"❌ Error processing document: {error_msg}\n"))
                root.after(0, lambda: status_label.config(text="Error processing document"))
            finally:
                # Always re-enable buttons
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
#                     MODEL SELECTION                               #
# ------------------------------------------------------------------ #

def check_ollama_availability():
    """Checks if Ollama is available and running."""
    global ollama_available
    try:
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
        ollama_available = True
        print("Ollama is available. Models:", models_response)
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        ollama_available = False
        return False


def get_best_available_model():
    """Returns the best available model from Ollama."""
    global ollama_available

    if not check_ollama_availability():
        print("Ollama availability check failed.")
        return None

    try:
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
        print("Raw models_response:", models_response)

        # Extract model names
        available_models = [model.model if hasattr(model, 'model') else model.get("model", "") for model in models_response.get("models", [])]
        print("Available models:", available_models)

        if not available_models:
            print("No models found in response.")
            return None

        # Preferred models in order
        preferred_models = ["llama3", "mistral", "phi"]

        # Find the first available preferred model
        for model in preferred_models:
            matching_models = [m for m in available_models if model in m]
            if matching_models:
                print(f"Selected model: {matching_models[0]}")
                return matching_models[0]

        # Return first available model if no preferred ones are found
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
#                     FALLBACK RESPONSE                             #
# ------------------------------------------------------------------ #

def generate_fallback_response(query, context):
    """Generates a simple response when Ollama is not available."""
    # Very basic keyword matching for when Ollama is not available
    query_lower = query.lower()

    # Extract name if asked about name
    if "name" in query_lower or "who" in query_lower:
        name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', context)
        if name_match:
            return name_match.group(1)
        else:
            return "Could not find a name in the document."

    # Extract date if asked about date
    elif "date" in query_lower or "when" in query_lower or "time" in query_lower:
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}, \d{4}|\d{1,2} \w+ \d{4})\b', context)
        if date_match:
            return date_match.group(1)
        else:
            return "Could not find a date in the document."

    # Extract email if asked about contact
    elif "email" in query_lower or "contact" in query_lower:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', context)
        if email_match:
            return email_match.group(0)
        else:
            return "Could not find email information in the document."

    # Default response
    else:
        return "I found some relevant information but cannot process it in detail without Ollama. Please check that Ollama is running."


# ------------------------------------------------------------------ #
#                          OLLAMA CHAT                               #
# ------------------------------------------------------------------ #

def generate_response():
    """Generates a response using Ollama AI."""
    global conversation_history, ollama_available

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
            print("\n📄 Retrieved Context for Query:\n", combined_context)

            # Add query to conversation history
            conversation_history.append({"role": "user", "content": user_query})

            # Check if Ollama is available
            if not check_ollama_availability():
                fallback_response = generate_fallback_response(user_query, combined_context)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                root.after(0, lambda: chat_window.insert(tk.END,
                                                         f"🤖 Ai Agent: {fallback_response}\n\n⚠️ Note: Ollama is not available. Please ensure Ollama is running.\n"))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            # Get the best available model
            model_to_use = get_best_available_model()
            if not model_to_use:
                fallback_response = generate_fallback_response(user_query, combined_context)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                root.after(0, lambda: chat_window.insert(tk.END,
                                                         f"🤖 Ai Agent: {fallback_response}\n\n⚠️ Note: No suitable Ollama models found.\n"))
                root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            print("Using model:", model_to_use)

            # Determine question type
            question_type = "general"
            if any(word in user_query.lower() for word in ["name", "who", "person"]):
                question_type = "name"
            elif any(word in user_query.lower() for word in ["when", "date", "time", "year"]):
                question_type = "date"
            elif any(word in user_query.lower() for word in ["where", "location", "place", "address"]):
                question_type = "location"
            elif any(word in user_query.lower() for word in ["skill", "ability", "competence", "knowledge"]):
                question_type = "skill"
            elif any(word in user_query.lower() for word in ["experience", "work", "job", "position"]):
                question_type = "experience"

            # Create a focused prompt based on the query type
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
            elif question_type == "location":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the specific location or address information requested
2. Return JUST the location with no additional text or explanation
3. Format your response as a single line with just the location
4. DO NOT include any of the document context in your response
5. If you can't find the location with certainty, respond with ONLY "Location information not found"

Answer:"""
            elif question_type == "skill":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the specific skills or competencies requested
2. List the skills in a concise, bullet-point format
3. Do not include explanations or additional text
4. DO NOT include any of the document context in your response
5. If you can't find the skills with certainty, respond with ONLY "Skill information not found"

Answer:"""
            elif question_type == "experience":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the specific work experience information requested
2. Format as concise bullet points with company, position, and dates
3. Do not include explanations or additional text
4. DO NOT include any of the document context in your response
5. If you can't find the experience with certainty, respond with ONLY "Experience information not found"

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

            # Get response from Ollama using a consistent client
            client = ollama.Client(host="http://localhost:11434")
            try:
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
                root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {clean_response}\n"))
                root.after(0, lambda: chat_window.yview(tk.END))

            except Exception as ollama_error:
                print(f"Ollama chat error: {ollama_error}")
                fallback_response = generate_fallback_response(user_query, combined_context)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                root.after(0, lambda: chat_window.insert(tk.END,
                                                         f"🤖 Ai Agent: {fallback_response}\n\n⚠️ Note: Error communicating with Ollama: {str(ollama_error)}\n"))

        except Exception as process_error:
            error_message = str(process_error)
            print(f"Query processing error: {error_message}")
            root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: Error processing query: {error_message}\n"))
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


# ------------------------------------------------------------------ #
#                      CHECK OLLAMA STATUS                          #
# ------------------------------------------------------------------ #

def check_ollama_status():
    """Checks Ollama status and updates UI accordingly."""
    global ollama_available

    if check_ollama_availability():
        model = get_best_available_model()
        if model:
            status_label.config(text=f"Ollama ready: {model}")
        else:
            status_label.config(text="Ollama ready but no models found")
    else:
        status_label.config(text="⚠️ Ollama not available")

    # Schedule next check
    root.after(30000, check_ollama_status)  # Check every 30 seconds


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

# Refresh Ollama button
refresh_btn = tk.Button(
    top_frame,
    text="🔄 Check Ollama",
    command=check_ollama_status,
    font=("Arial", 11),
    bg="#2196F3",
    fg="white",
    padx=10
)
refresh_btn.pack(side=tk.LEFT, padx=5)

# Status label
status_label = tk.Label(
    top_frame,
    text="Checking Ollama status...",
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

# Check Ollama status on startup
root.after(1000, check_ollama_status)

# Run GUI
root.mainloop()