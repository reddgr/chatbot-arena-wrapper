import streamlit as st
import pandas as pd
import duckdb
import requests
import tempfile
import random
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys

dotenv_path = "../../apis/.env"
hf_token, hf_token_write = env_options.check_env(colab=False, use_dotenv=True, dotenv_path=dotenv_path)

# Streamlit App Title
st.title("Chatbot Arena Dataset Explorer")

# Initialize DatasetWrapper
wrapper = lmsys.DatasetWrapper(hf_token, request_timeout=10)

# Display Active Dataframe with Pagination
st.write(f"{len(wrapper.active_df)} conversations loaded")
st.dataframe(wrapper.active_df.head(5))

# Function to Display Conversation in Streamlit
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

# Display Initial Active Conversation
st.write("---")
if wrapper.active_conversation:
    st.text(f"Conversation Preview (ID: {wrapper.active_conversation.conversation_id}):")
    display_conversation(wrapper.active_conversation)
