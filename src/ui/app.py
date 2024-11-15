# import sys
# from pathlib import Path

# # Add the src directory to sys.path
# src_path = Path(__file__).resolve().parent.parent
# print(src_path)
# sys.path.append(str(src_path))
import streamlit as st

# from llama_index.core.query_engine import RetrieverQueryEngine
from src.llm.deployments import AvailableChatModels, get_chat_model
from src.llm.embeddings import AvailableEmbeddingModels, get_embedding_model

# from llama_index import RetrieverQueryEngine  # Placeholder for later integration
# Placeholder import for your model loading function
# from src.models.llama_model import load_model


# retriever = VectorDBRetriever(vector_store, embeddings)
# query_engine = RetrieverQueryEngine(llm)
# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")
# Title at the top of the page
st.title("Dungeon Master Assistant Chat")
# Sidebar for additional options
st.sidebar.title("Dungeon Master Assistant")
st.sidebar.write("This is a prototype for a Dungeon Master assistant app.")


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# Model selection
model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Button to load the model
load_model_button = st.sidebar.button("Load Model")


@st.cache_resource
def load_llm(model_name: AvailableChatModels):
    return get_chat_model(
        model_name=model_name,
        max_new_tokens=20000,
    )


@st.cache_resource
def load_emb():
    return get_embedding_model(model_name=AvailableEmbeddingModels.BGE_LARGE_EN)


# Load the model if the button is clicked
if load_model_button and model_name_str:
    model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
    st.session_state.llm = load_llm(model_name=model_name)
    st.session_state.embeddings = load_emb()
    st.success("Model and embeddings loaded successfully!")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field for asking questions
if prompt := st.chat_input("Ask me anything"):
    # Display user's message in the chat and add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the response from the model if it's loaded
    if st.session_state.llm:
        response = st.session_state.llm.complete(prompt)
    else:
        response = "Model is not loaded. Please load the model from the sidebar."

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
