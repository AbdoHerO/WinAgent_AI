# src\query_handler.py
import json
import re
import ollama
import tkinter as tk
import threading
import database_utils
from synonyms_seham_winoffre import TABLE_SYNONYMS, COLUMN_SYNONYMS
from rag_engine import retrieve_relevant_schema, search_relevant_chunks
from document_processor import document_loaded, full_document_text
from ollama_utils import check_ollama_availability, get_best_available_model

conversation_history = []
ollama_available = True

def is_database_query(query):
    db_keywords = ["how many", "total", "sum", "count", "user", "customer", "order", "sales", "turnover", "amount", "what is", "client", "commande"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in db_keywords)

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
    if database_utils.db_schema is None:  # Updated here
        return None

    cached_query = retrieve_past_query(user_query)
    if cached_query:
        print(f"Using cached SQL Query: {cached_query}")
        return cached_query

    processed_query = preprocess_user_query(user_query)
    relevant_schema = retrieve_relevant_schema(processed_query)

    prompt = f"""You are an AI assistant generating SQL queries for a PostgreSQL database based on natural language questions from non-technical users.

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
   - 'user' refers to 'users'.
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
        if sql_query != "Unable to generate SQL query" and "..." not in sql_query and "SELECT" in sql_query.upper():
            store_successful_query(user_query, sql_query)
            return sql_query
        return None
    except Exception as e:
        print(f"Error generating SQL query with Ollama: {e}")
        return None

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

def generate_response(chat_window, query_entry, send_btn):
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
            print(f"DEBUG: db_connection={database_utils.db_connection}, document_loaded={document_loaded}")
            if user_query.lower() in ["hi", "hello", "hey"]:
                response = "Hello! How can I assist you today?"
            elif database_utils.db_connection is not None and is_database_query(user_query):
                print("DEBUG: Entering database query path")
                sql_query = generate_sql_query(user_query)
                if sql_query:
                    result = database_utils.execute_database_query(sql_query, chat_window)
                    response = result
                else:
                    response = "I'm sorry, I couldn't generate a valid SQL query. Try rephrasing your request."
            elif document_loaded:
                print("DEBUG: Entering document query path")
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

            tk_root.after(0, lambda: chat_window.insert(tk.END, f"🤖 Ai Agent: {response}\n"))
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
    chat_window.insert(tk.END, "💬 Conversation cleared. Document and database knowledge remain available.\n")

# Pass root from main.py
tk_root = None
def set_root(root):
    global tk_root
    tk_root = root