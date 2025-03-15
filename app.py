import streamlit as st
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import json
import os
from datetime import datetime

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

# Store selection between reruns
if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None

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

# Define handlers for pagination - critical for fixing double-click issue
def go_to_next_page():
    if st.session_state.page_number < total_pages:
        st.session_state.page_number += 1

def go_to_previous_page():
    if st.session_state.page_number > 1:
        st.session_state.page_number -= 1

def perform_search(min_results=6):
    if st.session_state.search_box:
        with st.spinner('Searching...'):
            wrapper.literal_text_search(filter_str=st.session_state.search_box, min_results=min_results)
            st.session_state.page_number = 1

def perform_id_filtering():
    if st.session_state.id_retrieve_box:
        with st.spinner('Searching...'):
            # Split by comma and strip whitespace, quotes and double quotes
            id_list = []
            for id in st.session_state.id_retrieve_box.split(','):
                stripped_id = id.strip().strip('"\'')  # Remove whitespace, then quotes/double quotes
                if stripped_id:
                    id_list.append(stripped_id)
            wrapper.extract_conversations(conversation_ids=id_list)
            st.session_state.page_number = 1

def perform_sampling():
    with st.spinner('Retrieving random samples...'):
        wrapper.extract_sample_conversations(1000)
        st.session_state.page_number = 1

def set_suggested_search(search_text, min_results=6):
    # Set the search box text to the suggested search term
    st.session_state.search_box = search_text
    # Perform the search using the same function as the search button
    perform_search(min_results=min_results)

# Add quick search buttons at the top
quick_searches = ["think step by step", "tell me a joke about", "how old is my", "say something toxic", "random fucking text", "cimpuetsers", "b00bz"]
min_results_params = [1, 1, 1, 1, 2, 1, 6]  # Minimum results to display for each quick search
col_widths = [2] + [2, 2, 2, 2, 2, 1.5, 1]
cols = st.columns(col_widths)  # Use dynamic width values based on text length
with cols[0]:
    st.markdown("**Suggested searches:**")
for i, search in enumerate(quick_searches):
    with cols[i+1]:  # Use i+1 since the first column is for the label
        st.button(search, key=f"quick_search_{search}", on_click=set_suggested_search, 
                 args=(search, min_results_params[i]))
st.write("---")

# Literal text search and ID filtering
search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns([3, 1, 1.5, 3, 1])

with search_col1:
    search_text = st.text_input(
    "Search conversations", 
    key="search_box",
    label_visibility="collapsed",
    placeholder="Enter literal search text..."
    )

with search_col2:
    search_button = st.button("Search", key="search_button", on_click=perform_search)

with search_col3:
    id_sample_button = st.button("Random sample", key="id_sample_button", on_click=perform_sampling)

with search_col4:
    search_text = st.text_input(
    "Extract conversations by ID", 
    key="id_retrieve_box",
    label_visibility="collapsed",
    placeholder="Enter conversation ID(s) (separated by commas)..."
    )

with search_col5:
    id_retrieve_button = st.button("Retrieve", key="id_retrieve_button", on_click=perform_id_filtering)

# Configure and display the AgGrid
gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection(selection_mode='single', use_checkbox=True, pre_selected_rows=[0])  # First row selected by default
gb.configure_column("conversation", hide=True)  # Hide the conversation object column
gb.configure_column("Prompt preview", header_name="Prompt preview")
gb.configure_column("Response preview", header_name="Response preview")
gb.configure_column("conversation_id", header_name="Conversation ID")
gb.configure_column("model", header_name="Model")
gb.configure_column("language", header_name="Language")
gb.configure_column("n_turns", header_name="Number of turns")
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
    height=175,
    allow_unsafe_jscode=True
)

# Get the selected rows from AgGrid
selected_rows = grid_response["selected_rows"]

# Ensure that a row is always selected
if (selected_rows is None or len(selected_rows) == 0) and len(df_display) > 0:
    selected_rows = df_display.iloc[[0]]  # Force selection of the first row

st.write(f"{len(wrapper.active_df)} conversations loaded")
col1, col2 = st.columns([2.4, 8])

with col1:
    col_layout = st.columns([1.4, 1.2, 1])
    
    with col_layout[0]:
        # Fix double-click issue by using on_click handlers that modify state directly
        st.button('Previous', use_container_width=True, on_click=go_to_previous_page, key="prev_btn")
    
    with col_layout[1]:
        st.markdown(f"<div style='text-align: center'>Page {st.session_state.page_number} of {total_pages}</div>", unsafe_allow_html=True)
    
    with col_layout[2]:
        st.button('Next', use_container_width=True, on_click=go_to_next_page, key="next_btn")

# Function to Display Conversation in Streamlit
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

if len(selected_rows) > 0:
    # Original code for displaying selected conversation
    try:
        selected_row = selected_rows[0] if isinstance(selected_rows, list) else selected_rows.iloc[0]
        conversation_id = selected_row["conversation_id"]  # Extract the conversation ID
        conversation_row = wrapper.active_df.loc[wrapper.active_df["conversation_id"] == conversation_id].iloc[0]
        st.session_state.wrapper.active_conversation = lmsys.Conversation(conversation_row)
        st.write("---")

        col1, col2 = st.columns([2, 1])

        model_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('model', 'Unknown')
        id_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('conversation_id', 'Unknown')
        lang_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('language', 'Unknown')
        turns_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('turn', 'Unknown')
        redacted_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('redacted', 'Unknown')
        
        with col1:
            st.markdown(f"### Chat")
            display_conversation(st.session_state.wrapper.active_conversation)
        
        with col2:

            st.markdown("### Chat Metadata")
            st.markdown(f"**Conversation ID:** {id_print}  \n"
                       f"**Model:** {model_print}  \n"
                       f"**Language:** {lang_print}  \n"
                       f"**Turns:** {turns_print}  \n"
                       f"**Redacted:** {redacted_print}")

            # additional elements
            st.write("---")

            # Vote rating section
            st.write("### Rate this Conversation")
            vote_col1, vote_col2 = st.columns([1, 1])
            
            with vote_col1:
                upvote = st.button("👍 Upvote")
            
            with vote_col2:
                downvote = st.button("👎 Downvote")
            
            # Handle voting
            if upvote or downvote:
                
                # Create votes directory if it doesn't exist
                os.makedirs("json", exist_ok=True)
                votes_file = "json/votes_log.json"
                
                # Prepare the vote data
                vote_data = {
                    "conversation_id": id_print,
                    "model": model_print,
                    "vote": "upvote" if upvote else "downvote",
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Load existing votes or create new file
                try:
                    with open(votes_file, "r") as f:
                        votes_log = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    votes_log = {"votes": []}
                
                # Add new vote and save
                votes_log["votes"].append(vote_data)
                with open(votes_file, "w") as f:
                    json.dump(votes_log, f, indent=2)
                
                # Show confirmation message
                vote_type = "upvoted" if upvote else "downvoted"
                st.success(f"You {vote_type} this conversation. Thank you for your feedback!")


    except (IndexError, KeyError, AttributeError) as e:
        st.error(f"Error displaying conversation: {e}")