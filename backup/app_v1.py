import streamlit as st
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(layout="wide")  # Set base layout to wide

# Streamlit App Title
# st.title("Chatbot Arena Dataset Explorer")

dotenv_path = "../../apis/.env"

# Initialize session state for dataset only if not already loaded
if "wrapper" not in st.session_state:
    hf_token, hf_token_write = env_options.check_env(use_dotenv=True, dotenv_path=dotenv_path)

    with st.spinner('Loading...'):
        st.session_state.wrapper = lmsys.DatasetWrapper(hf_token, request_timeout=10)
        # st.session_state.initial_sample = st.session_state.wrapper.extract_sample_conversations(50)

    st.session_state.page_number = 1  # Initialize page state

# Alias to session state variables
wrapper = st.session_state.wrapper
page_number = st.session_state.page_number

# Pagination setup
page_size = 5
total_pages = (len(wrapper.active_df) + page_size - 1) // page_size

start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size

# st.dataframe(wrapper.active_df.iloc[start_idx:end_idx])

# Replace the st.dataframe call with st.data_editor to enable row selection
df_display = wrapper.active_df.iloc[start_idx:end_idx].copy()

# Extract the first message content from each conversation as preview
df_display["Prompt preview"] = df_display.apply(
    lambda row: row.conversation[0].get("content", "")[:100] + "..." 
    if len(row.conversation) > 0 else "No content", 
    axis=1
)
df_display["Response preview"] = df_display.apply(
    lambda row: row.conversation[1].get("content", "")[:100] + "..." 
    if len(row.conversation) > 0 else "No content", 
    axis=1
)

df_display = df_display[["conversation_id", "Prompt preview", "Response preview", "model", "language", "turn", "conversation"]]
df_display = df_display.rename(columns={"turn": "n_turns"})


def click_button():
    st.session_state.clicked = True

# Literal text search
search_col1, search_col2, search_col3 = st.columns([3, 1, 3])

with search_col1:
    search_text = st.text_input(
    "Search conversations", 
    key="search_box",
    label_visibility="collapsed",
    placeholder="Enter search text..."
    )

with search_col2:
    search_button = st.button("Search", key="search_button", on_click=click_button)

if search_button and search_text:
    with st.spinner('Searching...'):
        wrapper.literal_text_search(filter_str=search_text, min_results=6)
        st.session_state.page_number = 1
        # Update the session state to indicate that search is in progress
        if "search_in_progress" not in st.session_state:
            st.session_state.search_in_progress = True
        wrapper.literal_text_search(filter_str=search_text, min_results=6)
        st.session_state.page_number = 1
        # Reset the search_in_progress flag
        st.session_state.search_in_progress = False
        st.rerun()

# Configure and display the AgGrid
gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection(selection_mode='single', use_checkbox=True, pre_selected_rows=[0])  # First row selected by default
#gb.configure_column("select", hide=True)  # Hide the select column as we're using checkbox selection
gb.configure_column("conversation", hide=True)  # Hide the conversation object column
gb.configure_column("Prompt preview", header_name="Prompt preview")
gb.configure_column("Response preview", header_name="Response preview")
gb.configure_column("conversation_id", header_name="Conversation ID")
gb.configure_column("Model", header_name="Model")
gb.configure_column("Language", header_name="Language")
gb.configure_column("Turns", header_name="Number of turns")
gb.configure_grid_options(domLayout='normal')

grid_options = gb.build()
grid_options['columnDefs'] = [
    {'field': 'View', 'headerCheckboxSelection': True, 'checkboxSelection': True, 'width': 50},
    {'field': 'conversation_id', 'width': 150},
    {'field': 'Prompt preview', 'width': 300}, 
    {'field': 'Response preview', 'width': 300}, 
    {'field': 'model', 'width': 70},
    {'field': 'language', 'width': 55},
    {'field': 'n_turns', 'width': 45}
]

grid_response = AgGrid(
    df_display,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
    height=200,
    allow_unsafe_jscode=True
)

# Get the selected rows from AgGrid
selected_rows = grid_response["selected_rows"]
# print(f"Traza: {selected_rows} - {type(selected_rows)}")

# Ensure that a row is always selected
if (selected_rows is None or len(selected_rows) == 0) and len(df_display) > 0:
    selected_rows = df_display.iloc[[0]]  # Force selection of the first row


# edited_df = grid_response["data"]

st.write(f"{len(wrapper.active_df)} conversations loaded")
col1, col2 = st.columns([2, 8])

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

with col1:
    col_layout = st.columns([1.1, 1.2, 1])  # Make pagination buttons more compact
    
    with col_layout[0]:
        if st.button('Previous', use_container_width=True, on_click=click_button) and page_number > 1:
            st.session_state.page_number -= 1
    
    with col_layout[1]:
        st.markdown(f"<div style='text-align: center'>Page {st.session_state.page_number} of {total_pages}</div>", unsafe_allow_html=True)
    
    with col_layout[2]:
        if st.button('Next', use_container_width=True, on_click=click_button) and page_number < total_pages:
            st.session_state.page_number += 1

# Function to Display Conversation in Streamlit
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

if len(selected_rows) > 0:
    selected_row = selected_rows.iloc[0]  # Get the "first selected row" (there's only one, but it's a DataFrame)
    conversation_id = selected_row["conversation_id"]  # Extract the conversation ID
    conversation_row = wrapper.active_df.loc[wrapper.active_df["conversation_id"] == conversation_id].iloc[0]
    st.session_state.wrapper.active_conversation = lmsys.Conversation(conversation_row)
    st.write("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_conversation(st.session_state.wrapper.active_conversation)
    
    with col2:
        model_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('model', 'Unknown')
        id_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('conversation_id', 'Unknown')
        table_data = {"Attribute": ["Model", "Conversation ID"], "Value": [model_print, id_print]}
        st.table(table_data)

        # additional elements
        st.write("---")






