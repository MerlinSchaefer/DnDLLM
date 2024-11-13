import streamlit as st

# from llama_index import RetrieverQueryEngine  # Placeholder for later integration
# Placeholder import for your model loading function
# from src.models.llama_model import load_model


# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")

# Sidebar for additional options
st.sidebar.title("Dungeon Master Assistant")
st.sidebar.write("This is a prototype for a Dungeon Master assistant app.")

# Title at the top of the page
st.title("Dungeon Master Assistant Chat")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # Echo response from assistant for demonstration purposes
    response = f"Echo: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
