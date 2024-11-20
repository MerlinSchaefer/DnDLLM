# import sys
# from pathlib import Path

# # Add the src directory to sys.path
# src_path = Path(__file__).resolve().parent.parent
# print(src_path)
# sys.path.append(str(src_path))

from pathlib import Path

import streamlit as st
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore

from src.db import get_vector_store
from src.llm.deployments import AvailableChatModels, get_chat_model
from src.llm.embeddings import AvailableEmbeddingModels, get_embedding_model
from src.llm.retriever import VectorDBRetriever
from src.parsers import DocumentParser


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
def get_query_engine(_llm: LlamaCPP) -> RetrieverQueryEngine:
    retriever = load_retriever()
    return RetrieverQueryEngine.from_args(retriever, llm=_llm, response_mode=ResponseMode.COMPACT)


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None


# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")
# Title at the top of the page
st.title("Dungeon Master Assistant Chat")
# Sidebar for additional options
st.sidebar.title("Options")
st.sidebar.write("Manage models, documents, and queries.")


# Sidebar: Model selection
model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Sidebar: Button to load the model and query engine
load_model_button = st.sidebar.button("Load Model and Engine")

# Sidebar: File Upload for Document Parsing
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "md"])
doc_type = st.sidebar.selectbox("Document type", options=["pdf", "txt", "md"])
process_file_button = st.sidebar.button("Process File")


# Load the model and query engine if the button is clicked
if load_model_button and model_name_str:
    model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
    st.session_state.llm = load_llm(model_name=model_name)
    st.session_state.query_engine = get_query_engine(st.session_state.llm)
    st.success("Model and query engine loaded successfully!")

# Process the uploaded document
if process_file_button and uploaded_file:
    # Save the uploaded file to a temporary path
    temp_file_path = Path(f"temp_{uploaded_file.name}")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse and embed the document
    st.write(f"Processing document: {uploaded_file.name} as {doc_type}...")
    document_parser = DocumentParser(temp_file_path, doc_type)
    try:
        nodes = document_parser.load_chunk_and_embed()
        st.success("Document processed into nodes successfully!")

        # Add nodes to the vector store
        vectorstore = load_vector_store()
        vectorstore.add(nodes)
        st.success("Nodes added to the vector store!")

        # Optionally, display the nodes
        st.write("Retrieved nodes:")
        node_ids = [node.id_ for node in nodes]
        nodes_retrieved = vectorstore.get_nodes(node_ids)
        for node in nodes_retrieved:
            st.write(node)

    except Exception as e:
        st.error(f"Error processing document: {e}")
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()


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
        response = st.session_state.query_engine.query(prompt)
        print(response.source_nodes)
    else:
        response = "Model and query engine are not loaded. Please load them from the sidebar."

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response.response})
    with st.chat_message("assistant"):
        st.markdown(response.response)
