import streamlit as st
import requests
import base64
import time
from langchain_community.llms.ollama import Ollama




BASE_URL = "http://192.168.86.29:11434"

def list_models():
    response = requests.get(f"{BASE_URL}/api/tags")
    if response.status_code == 200:
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    else:
        st.error("Failed to fetch models")
        return []

def pull_model(model_name):
    response = requests.post(f"{BASE_URL}/api/pull", json={"name": model_name})
    if response.status_code == 200:
        st.success(f"Model '{model_name}' pulled successfully")
    else:
        st.error(f"Failed to pull model '{model_name}': {response.text}")

def get_image_base64(uploaded_image):
    bytes_data = uploaded_image.getvalue()
    base64_string = base64.b64encode(bytes_data).decode("utf-8")
    return base64_string

def generate_response(input_text, llm, image_b64=None):
    # Instantiate the Ollama object with the selected model

    # If an image is uploaded, bind it to the llm object
    if image_b64:
        llm_with_image_context = llm.bind(images=[image_b64])
        return llm_with_image_context.invoke(input_text)
    else:
        # Generate a response using the bound llm object
        return llm.invoke(input_text)

 

# List available models
available_models = list_models()


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Ollama Server")
# Select Model Section
with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Choose a model:", available_models)

    st.header("Pull New Model")
    new_model_name = st.text_input("Enter model name:")
    if st.button("Pull Model"):
        pull_model(new_model_name)

# Image Upload Section
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image_b64 = None
llm = Ollama(
        model=selected_model,
        base_url=BASE_URL,
        num_gpu=1,
        #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


if uploaded_image is not None:
    image_b64 = get_image_base64(uploaded_image)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
# Set a default model




# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        for response in generate_response(st.session_state.messages[-1]["content"], llm , image_b64=image_b64):
            full_response += (response  or "")
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})



    
    
    
    
