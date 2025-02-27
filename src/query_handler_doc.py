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
    """Generates a response using Ollama AI."""
    global conversation_history, ollama_available

    user_query = query_entry.get().strip()
    if not user_query:
        return

    send_btn.config(state=tk.DISABLED)
    query_entry.delete(0, tk.END)
    chat_window.insert(tk.END, f"\n🧑‍💼 You: {user_query}\n")
    chat_window.update()

    # Use live state from document_processor
    if not document_processor.document_loaded:
        response = "⚠️ No document loaded! Please upload a PDF or Excel file first."
        chat_window.insert(tk.END, f"🤖 Ai Agent: {response}\n")
        send_btn.config(state=tk.NORMAL)
        return

    def process_query():
        try:
            # Get relevant document chunks
            retrieved_chunks = search_relevant_chunks(user_query, top_k=3)
            print(f"DEBUG: Retrieved chunks for '{user_query}': {retrieved_chunks}")  # Debug retrieval

            if not retrieved_chunks:
                tk_root.after(0, lambda: chat_window.insert(tk.END,
                                                           "🤖 Ai Agent: I couldn't find relevant information in the document.\n"))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            combined_context = "\n\n".join(retrieved_chunks)
            print(f"DEBUG: Combined context: {combined_context}")  # Debug context

            conversation_history.append({"role": "user", "content": user_query})

            if not check_ollama_availability():
                fallback_response = generate_fallback_response(user_query, combined_context)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                tk_root.after(0, lambda: chat_window.insert(tk.END,
                                                           f"🤖 Ai Agent: {fallback_response}\n\n⚠️ Note: Ollama is not available. Please ensure Ollama is running.\n"))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            model_to_use = get_best_available_model()
            if not model_to_use:
                fallback_response = generate_fallback_response(user_query, combined_context)
                conversation_history.append({"role": "assistant", "content": fallback_response})
                tk_root.after(0, lambda: chat_window.insert(tk.END,
                                                           f"🤖 Ai Agent: {fallback_response}\n\n⚠️ Note: No suitable Ollama models found.\n"))
                tk_root.after(0, lambda: send_btn.config(state=tk.NORMAL))
                return

            print("Using model:", model_to_use)

            question_type = "general"
            if any(word in user_query.lower() for word in ["name", "qui", "who", "person"]):
                question_type = "name"
            elif any(word in user_query.lower() for word in ["when", "date", "time", "year"]):
                question_type = "date"
            elif any(word in user_query.lower() for word in ["where", "location", "place", "address"]):
                question_type = "location"
            elif any(word in user_query.lower() for word in ["prix", "price", "cost"]):
                question_type = "price"

            if question_type == "name":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the full name of the person from the document
2. Return JUST the name with no additional text or explanation
3. Format your response as a single line with just the name
4. If multiple names exist, return the most relevant name based on context
5. If you can't find the name, respond with ONLY "Name not found"
6. DO NOT include any of the document context in your response
7. DO NOT include phrases like "Based on the document" or "According to the text"

Answer:"""
            elif question_type == "date":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the specific date or time information requested
2. Return JUST the date/time with no additional text or explanation
3. Format your response as a single line with just the date/time
4. If you can't find the date, respond with ONLY "Date not found"
5. DO NOT include any of the document context in your response
6. DO NOT include phrases like "Based on the document" or "According to the text"

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
4. If you can't find the location, respond with ONLY "Location not found"
5. DO NOT include any of the document context in your response
6. DO NOT include phrases like "Based on the document" or "According to the text"

Answer:"""
            elif question_type == "price":
                prompt = f"""You are an AI assistant analyzing a document.

Document context:
{combined_context}

User question: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Extract ONLY the price information requested from the document
2. Return JUST the price with no additional text or explanation (e.g., "$1.99" or "2.50 €")
3. Format your response as a single line with just the price
4. If you can't find the price, respond with ONLY "Price not found"
5. DO NOT include any of the document context in your response
6. DO NOT include phrases like "Based on the document" or "According to the text"

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
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {clean_response}\n"))
            tk_root.after(0, lambda: chat_window.yview(tk.END))

        except Exception as process_error:
            error_message = str(process_error)
            print(f"Query processing error: {error_message}")
            tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: Error processing query: {error_message}\n"))
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