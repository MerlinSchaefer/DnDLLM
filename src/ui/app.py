# import sys
# from pathlib import Path

# # Add the src directory to sys.path
# src_path = Path(__file__).resolve().parent.parent
# print(src_path)
# sys.path.append(str(src_path))
import streamlit as st
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore

from src.db import get_vector_store
from src.llm.deployments import AvailableChatModels, get_chat_model
from src.llm.embeddings import AvailableEmbeddingModels, get_embedding_model
from src.llm.retriever import VectorDBRetriever


# Funcs to create page resources
@st.cache_resource
def load_llm(model_name: AvailableChatModels) -> LlamaCPP:
    return get_chat_model(
        model_name=model_name,
        max_new_tokens=20000,
    )


@st.cache_resource
def load_emb() -> HuggingFaceEmbedding:
    return get_embedding_model(model_name=AvailableEmbeddingModels.BGE_SMALL_EN)  # BGE_LARGE_EN


@st.cache_resource
def load_vector_store() -> PGVectorStore:
    return get_vector_store(
        table_name="dev_vectors",  # TODO: adjust
        embed_dim=384,  # TODO: adjust
    )


@st.cache_resource
def load_retriever() -> VectorDBRetriever:
    vector_store = load_vector_store()
    embeddings = load_emb()
    return VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embeddings,
        query_mode="default",
        similarity_top_k=2,
    )


@st.cache_resource
def get_query_engine(llm: LlamaCPP) -> RetrieverQueryEngine:
    retriever = load_retriever()
    return RetrieverQueryEngine.from_args(retriever, llm=llm)


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
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Model selection
model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Button to load the model and query engine
load_model_button = st.sidebar.button("Load Model and Engine")

# Load the model and query engine if the button is clicked
if load_model_button and model_name_str:
    model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
    st.session_state.llm = load_llm(model_name=model_name)
    st.session_state.query_engine = get_query_engine(st.session_state.llm)
    st.success("Model and query engine loaded successfully!")

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

    # Get the response from the query engine if it's loaded
    if st.session_state.query_engine:
        response = st.session_state.query_engine.query(prompt).response
    else:
        response = "Model and query engine are not loaded. Please load them from the sidebar."

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
