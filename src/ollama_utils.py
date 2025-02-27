# src\ollama_utils.py

import ollama

ollama_available = True  # Global flag

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

def check_ollama_status(status_label):
    """Check Ollama availability and update the status label."""
    if check_ollama_availability():
        model = get_best_available_model()
        if model:
            status_label.config(text=f"Ollama ready: {model}")
        else:
            status_label.config(text="Ollama ready but no models found")
    else:
        status_label.config(text="⚠️ Ollama not available")