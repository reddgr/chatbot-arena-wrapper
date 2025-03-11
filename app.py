import streamlit as st
import pandas as pd
import duckdb
import requests
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys

# Function to Display Conversation in Streamlit
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

# Streamlit App Title
st.title("Chatbot Arena Dataset Explorer")

dotenv_path = "../../apis/.env"

# Initialize session state for dataset only if not already loaded
if "wrapper" not in st.session_state:
    hf_token, hf_token_write = env_options.check_env(use_dotenv=True, dotenv_path=dotenv_path)

    with st.spinner('Loading...'):
        st.session_state.wrapper = lmsys.DatasetWrapper(hf_token, request_timeout=10)
        st.session_state.initial_sample = st.session_state.wrapper.extract_sample_conversations(50)

    st.session_state.page_number = 1  # Initialize page state

# Alias to session state variables
wrapper = st.session_state.wrapper
page_number = st.session_state.page_number

# Display Active Dataframe
st.write(f"{len(wrapper.active_df)} conversations loaded")

# Pagination setup
page_size = 5
total_pages = (len(wrapper.active_df) + page_size - 1) // page_size

start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size

# st.dataframe(wrapper.active_df.iloc[start_idx:end_idx])

# Replace the st.dataframe call with st.data_editor to enable row selection
df_display = wrapper.active_df.iloc[start_idx:end_idx].copy()
df_display["select"] = False  # Add a selection column to capture row clicks
df_display = df_display[["select", "conversation_id", "conversation", "model"]]

edited_df = st.data_editor(df_display, key="data_editor", num_rows="dynamic")

# Check for any selected row and update the active conversation if one is found
selected_rows = edited_df[edited_df["select"] == True]
if not selected_rows.empty:
    selected_idx = selected_rows.index[0]
    st.session_state.wrapper.active_conversation = lmsys.Conversation(wrapper.active_df.iloc[selected_idx])
    st.write("---")
    st.text(f"Conversation ID: {wrapper.active_conversation.conversation_metadata.get('conversation_id')}:")
    st.text(f"Model: {wrapper.active_conversation.conversation_metadata.get('model')}:")
    display_conversation(wrapper.active_conversation)

# Pagination with Buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button('Previous page') and page_number > 1:
        st.session_state.page_number -= 1

with col3:
    if st.button('Next page') and page_number < total_pages:
        st.session_state.page_number += 1

st.write(f"Page {st.session_state.page_number} of {total_pages}")

# Display Initial Active Conversation
st.write("---")
if wrapper.active_conversation:
    st.text(f"Conversation ID: {wrapper.active_conversation.conversation_metadata.get('conversation_id')}:")
    st.text(f"Model: {wrapper.active_conversation.conversation_metadata.get('model')}:")
    display_conversation(wrapper.active_conversation)