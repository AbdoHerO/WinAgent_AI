# src\query_handler_doc.py

import json
import re
import ollama
import tkinter as tk
import threading
import document_processor  # Import module for live state
from rag_engine import search_relevant_chunks  # Restore this import
from ollama_utils import check_ollama_availability, get_best_available_model

conversation_history = []
ollama_available = True

def generate_fallback_response(query, context):
    """Generates a simple response when Ollama is not available."""
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
        return "I found some relevant information but cannot process it without Ollama. Please check that Ollama is running."

def generate_response(chat_window, query_entry, send_btn):
    """First tries document-based retrieval, then falls back to general AI chat."""
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
            combined_context = ""
            retrieved_chunks = []

            # Step 1️⃣: Check if document is loaded and search for relevant chunks
            if document_processor.document_loaded:
                retrieved_chunks = search_relevant_chunks(user_query, top_k=5)
                if retrieved_chunks:
                    combined_context = "\n\n".join(retrieved_chunks)

            # Step 2️⃣: If relevant document chunks are found, ask Llama3
            if combined_context:
                prompt = f"""
                You are a smart AI assistant analyzing a document.

                📄 **Document Context**:
                {combined_context}

                🧐 **User Question**: "{user_query}"

                🔎 **Guidelines**:
                - Answer **ONLY** based on the document content.
                - Keep the response short and accurate.
                - If the document does not contain relevant info, respond with: "Information not found."

                ✍️ **Response**:
                """

                client = ollama.Client(host="http://localhost:11434")
                ollama_response = client.chat(
                    model=get_best_available_model(),
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2}
                )

                raw_text = ollama_response["message"]["content"].strip()
                clean_response = re.sub(r'^(Answer:|Document context:)', '', raw_text, flags=re.IGNORECASE).strip()

                if clean_response.lower() != "information not found":
                    conversation_history.append({"role": "assistant", "content": clean_response})
                    tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {clean_response}\n"))
                    tk_root.after(0, lambda: chat_window.yview(tk.END))
                    tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                    return  # Stop here since we got a relevant document answer

            # Step 3️⃣: If no relevant document response, fall back to AI conversation
            general_prompt = f"""
            You are an AI assistant. The user asked:

            🧐 **User Question**: "{user_query}"

            🔎 **Guidelines**:
            - If you have document context, prioritize using it.
            - If no document context exists, respond normally.
            - Keep responses clear and helpful.

            ✍️ **Response**:
            """

            ollama_response = client.chat(
                model=get_best_available_model(),
                messages=[{"role": "user", "content": general_prompt}],
                options={"temperature": 0.5}  # More creativity in general chat
            )

            general_response = ollama_response["message"]["content"].strip()
            conversation_history.append({"role": "assistant", "content": general_response})
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {general_response}\n"))
            tk_root.after(0, lambda: chat_window.yview(tk.END))

        except Exception as process_error:
            error_message = str(process_error)
            print(f"Query processing error: {error_message}")
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: Erreur: {error_message}\n"))
        finally:
            tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))

    threading.Thread(target=process_query).start()

def clear_conversation(chat_window):
    global conversation_history
    conversation_history = []
    chat_window.delete(1.0, tk.END)
    chat_window.insert(tk.END, "💬 Conversation cleared. Document knowledge remains available.\n")

tk_root = None
def set_root(root):
    global tk_root
    tk_root = root
