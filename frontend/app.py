import streamlit as st
import requests
import json
import os

# Configure page
st.set_page_config(
    page_title="Ollama Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get backend URL from environment variable or use default for local development
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# --- Helper Functions to Communicate with the Backend ---

def check_backend_health():
    """Check if the backend is healthy and what models are available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("models", [])
        return False, []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return False, []
    
def stream_response(prompt, system_prompt, temperature, max_tokens):
    """
    Streams a response from the backend using the /chat endpoint.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True) as response:
        partial = ""
        for chunk in response.iter_content(chunk_size=None):
            text = chunk.decode("utf-8")
            partial += text
            yield text

def generate_response(prompt, system_prompt, temperature, max_tokens):
    """
    Generates a response from the backend using the /chat endpoint.
    This function now constructs the full message history for a stateful chat.
    """
    try:
        # 1. Construct the full message history for the chat API
        # Start with the system prompt to set the AI's behavior
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add the existing conversation history from session state
        messages.extend(st.session_state.messages)
        
        # Add the current user prompt to the end of the list
        messages.append({"role": "user", "content": prompt})

        # 2. Prepare the payload for the /chat endpoint
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 3. Make the POST request to the new /chat endpoint
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=60
        )
        
        # 4. Handle the response
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# --- Sidebar for Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Check backend health and display status
    is_healthy, models = check_backend_health()
    
    if is_healthy:
        st.success("✅ Backend is healthy")
        st.session_state.model_loaded = True
        
        # Model selection dropdown
        if models:
            selected_model = st.selectbox("Select Model", models)
        else:
            st.warning("No models available on the backend.")
            selected_model = None
        
    else:
        st.error("❌ Backend is not healthy")
        st.info("Please check the backend logs on Railway or ensure it's running locally.")
        st.session_state.model_loaded = False
    
    # User-configurable generation parameters
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=100
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Higher values make the output more creative."
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=1024,
        value=512,
        step=1,
        help="Maximum length of the generated response."
    )

# --- Main Chat Interface ---
st.title("🤖 Ollama Chat")

if not st.session_state.model_loaded:
    st.warning("Please wait for the model to load or check the backend connection.")
else:
    # Display the existing chat messages from the session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input from the chat input field
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                for token in stream_response(prompt, system_prompt, temperature, max_tokens):
                    full_response += token
                    message_placeholder.markdown(full_response)
        
        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Button to clear the conversation history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()