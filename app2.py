import streamlit as st
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys

# Function to display a conversation
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

st.title("Chatbot Arena Dataset Explorer")
dotenv_path = "../../apis/.env"

# Initialize session state for dataset only if not already loaded
if "wrapper" not in st.session_state:
    hf_token, hf_token_write = env_options.check_env(use_dotenv=True, dotenv_path=dotenv_path)
    with st.spinner('Loading...'):
        st.session_state.wrapper = lmsys.DatasetWrapper(hf_token, request_timeout=10)
        st.session_state.initial_sample = st.session_state.wrapper.extract_sample_conversations(50)
    st.session_state.page_number = 1  # Initialize page state

wrapper = st.session_state.wrapper
page_number = st.session_state.page_number

st.write(f"{len(wrapper.active_df)} conversations loaded")

# Pagination setup
page_size = 5
total_pages = (len(wrapper.active_df) + page_size - 1) // page_size

start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size

# Prepare the dataframe for display (without a selection column)
df_display = wrapper.active_df.iloc[start_idx:end_idx].copy()
df_table = df_display[["conversation_id", "conversation", "model"]]

# Display the table in standard format with pagination
st.dataframe(df_table)

# --- NEW CODE: Use a radio button for single selection ---
# Build a list of indices (or any unique identifier) for the current page
options = df_display.index.tolist()

# Create labels to help the user distinguish conversations (e.g., using conversation_id and model)
option_labels = [
    f"ID: {df_display.loc[i, 'conversation_id']} - {df_display.loc[i, 'model']}" 
    for i in options
]

# st.radio automatically allows only one option to be selected. 
# Setting index=0 defaults the selection to the first row on the current page.
selected_index = st.radio(
    "Select a conversation",
    options,
    format_func=lambda x: option_labels[options.index(x)],
    index=0
)

# Update the active conversation based on the radio selection.
st.session_state.wrapper.active_conversation = lmsys.Conversation(wrapper.active_df.iloc[selected_index])

st.write("---")
st.text(f"Conversation ID: {wrapper.active_conversation.conversation_metadata.get('conversation_id')}:")
st.text(f"Model: {wrapper.active_conversation.conversation_metadata.get('model')}:")
display_conversation(wrapper.active_conversation)
# --- END NEW CODE ---

# Pagination with Buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button('Previous page') and page_number > 1:
        st.session_state.page_number -= 1
with col3:
    if st.button('Next page') and page_number < total_pages:
        st.session_state.page_number += 1

st.write(f"Page {st.session_state.page_number} of {total_pages}")

# Display the active conversation details again
st.write("---")
if wrapper.active_conversation:
    st.text(f"Conversation ID: {wrapper.active_conversation.conversation_metadata.get('conversation_id')}:")
    st.text(f"Model: {wrapper.active_conversation.conversation_metadata.get('model')}:")
    display_conversation(wrapper.active_conversation)
