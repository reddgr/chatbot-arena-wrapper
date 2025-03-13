import streamlit as st
import sys

sys.path.append("./src")
import env_options
import lmsys_dataset_wrapper as lmsys
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# Function to Display Conversation in Streamlit
def display_conversation(conversation):
    for message in conversation.conversation_data:
        if message['role'] == 'user':
            st.markdown(f"😎 {message['content']}")
        elif message['role'] == 'assistant':
            st.markdown(f"🤖 {message['content']}")

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
df_display = df_display[["conversation_id", "conversation", "model"]]

df_display["select"] = False
if len(df_display) > 0:
    df_display.loc[df_display.index[0], "select"] = True  # Set first row's select to True


# edited_df = st.data_editor(df_display, key="data_editor", num_rows="dynamic")
# Configure and display the AgGrid
gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection(selection_mode='single', use_checkbox=True, pre_selected_rows=[0])  # First row selected by default
gb.configure_column("select", hide=True)  # Hide the select column as we're using checkbox selection
gb.configure_column("conversation", hide=True)  # Hide the conversation object column
gb.configure_column("conversation_id", header_name="Conversation ID")
gb.configure_column("model", header_name="Model")
gb.configure_grid_options(domLayout='normal')

grid_options = gb.build()
grid_response = AgGrid(
    df_display,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
    height=300
)

# Get the selected rows from AgGrid
selected_rows = grid_response["selected_rows"]
print(f"Traza: {selected_rows} - {type(selected_rows)}")

# Store edited dataframe for reference
edited_df = grid_response["data"]

# Pagination with Buttons (below both elements)
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button('Previous page') and page_number > 1:
        st.session_state.page_number -= 1

with col3:
    if st.button('Next page') and page_number < total_pages:
        st.session_state.page_number += 1

st.write(f"Page {st.session_state.page_number} of {total_pages}")

if len(selected_rows) > 0:
    selected_row = selected_rows.iloc[0]  # Get the "first selected row" (there's only one, but it's a DataFrame)
    conversation_id = selected_row["conversation_id"]  # Extract the conversation ID
    
    # Find the corresponding row in the original dataframe using the conversation ID
    conversation_row = wrapper.active_df.loc[wrapper.active_df["conversation_id"] == conversation_id].iloc[0]

    # Create Conversation object from the selected row
    st.session_state.wrapper.active_conversation = lmsys.Conversation(conversation_row)

    st.write("---")
    model_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('model')
    id_print = st.session_state.wrapper.active_conversation.conversation_metadata.get('conversation_id')
    st.text(f"Model: {model_print} | Conversation ID: {id_print}")
    
    # Display the conversation
    display_conversation(st.session_state.wrapper.active_conversation)