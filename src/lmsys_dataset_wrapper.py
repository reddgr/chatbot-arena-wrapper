import os
import pandas as pd
import textwrap
import random
import duckdb
import requests
import json
import tempfile
from datasets import load_dataset
from collections import defaultdict

class DatasetWrapper:
    def __init__(self, hf_token, dataset_name="lmsys/lmsys-chat-1m", verbose=True, 
                 conversations_index="json/conversations_index.json", cache_size=50, request_timeout=20):
        self.hf_token = hf_token
        self.dataset_name = dataset_name
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        self.timeout = request_timeout
        self.cache_size = cache_size
        self.verbose = verbose
        parquet_list_url = f"https://datasets-server.huggingface.co/parquet?dataset={self.dataset_name}"
        response = self._safe_get(parquet_list_url)
        # Extract URLs from the response JSON
        if response is not None:
            self.parquet_urls = [file['url'] for file in response.json()['parquet_files']]
            if self.verbose:
                print("\nParquet URLs:")
                for url in self.parquet_urls:
                    print(url)
                    head_response = self._safe_head(url)
                    file_size = int(head_response.headers['Content-Length'])
                    print(f"{url.split('/')[-1]}: {file_size} bytes")

        # Loading the index
        try:
            with open(conversations_index, "r", encoding="utf-8") as f:
                self.conversations_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Conversations index file not found or invalid. Creating a new one at {conversations_index}.")
            # Ensure directory exists
            os.makedirs(os.path.dirname(conversations_index), exist_ok=True)
            self.create_conversations_index(output_index_file=conversations_index)
            with open(conversations_index, "r", encoding="utf-8") as f:
                self.conversations_index = json.load(f)

        # Initialize active conversation and DataFrame        
        # Read from "pkl/cached_chats.pkl" if available:
        try:
            self.active_df = pd.read_pickle("pkl/cached_chats.pkl")
            print(f"Loaded {len(self.active_df)} cached chats")
            self.active_df = self.active_df.sample(self.cache_size).reset_index(drop=True)
        except (FileNotFoundError, ValueError):
            self.active_df = pd.DataFrame()
            print("No cached chats found")
        if not self.active_df.empty:
            try:
                self.active_conversation = Conversation(self.active_df.iloc[0])
            except Exception as e:
                print(f"No conversations available: {e}")
        else:
            self.active_conversation = None

    def _safe_get(self, url):
        if self.timeout == 0:
            print("Timeout is set to 0. Skipping GET request.")
            return None
        else:
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                if response.status_code != 200:
                    raise ValueError(f"Failed to retrieve {url}. Status code: {response.status_code}")
                return response
            except requests.exceptions.Timeout:
                print(f"Timeout occurred for GET {url}. Skipping.")
                return None
            
    def _safe_head(self, url):
        if self.timeout == 0:
            print("Timeout is set to 0. Skipping HEAD request.")
            return None
        try:
            response = requests.head(url, allow_redirects=True, headers=self.headers, timeout=self.timeout)
            return response
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for GET {url}. Skipping.")
            return None

    def extract_sample_conversations(self, n_samples):
        url = random.choice(self.parquet_urls)
        print(f"Sampling conversations from {url}")
        # Download file with auth headers using requests
        r = self._safe_get(url)
        if r is None:
            print(f"Timeout occurred for GET {url}. Skipping sample extraction.")
            return self.active_df
        # Write the downloaded content into a temporary file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp.write(r.content)
            # tmp.flush()
            tmp_path = tmp.name
        try:
            query_result = duckdb.query(f"SELECT * FROM read_parquet('{tmp_path}') USING SAMPLE {n_samples}").df()
            self.active_df = query_result
            try:
                self.active_conversation = Conversation(query_result.iloc[0])
            except Exception as e:
                print(f"No conversations available: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return query_result

    def extract_conversations(self, conversation_ids):

        # Create a lookup table for file names -> URLs
        file_url_map = {url.split("/")[-1]: url for url in self.parquet_urls}

        # Group conversation IDs by file
        file_to_conversations = defaultdict(list)
        for convid in conversation_ids:
            if convid in self.conversations_index:
                file_to_conversations[self.conversations_index[convid]].append(convid)

        result_df = pd.DataFrame()

        for file_name, conv_ids in file_to_conversations.items():
            if file_name not in file_url_map:
                print(f"File {file_name} not found in URL list, skipping.")
                continue

            file_url = file_url_map[file_name]
            print(f"Querying file: {file_name} for {len(conv_ids)} conversations")

            try:
                r = self._safe_get(file_url)
                if r == None:
                    print(f"Timeout occurred for GET {file_url}. Skipping file {file_name}.")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp.write(r.content)
                    tmp_path = tmp.name
                try:
                    conv_id_list = "', '".join(conv_ids)
                    query_str = f"""
                        SELECT * FROM read_parquet('{tmp_path}') 
                        WHERE conversation_id IN ('{conv_id_list}')
                    """
                    df = duckdb.query(query_str).df()
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                if not df.empty:
                    print(f"Found {len(df)} conversations in {file_name}")
                    result_df = pd.concat([result_df, df], ignore_index=True)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        self.active_df = result_df
        try:
            self.active_conversation = Conversation(self.active_df.iloc[0])
        except Exception as e:
            print(f"No conversations available: {e}")

        return result_df
    
    def literal_text_search(self, filter_str, min_results=1):
        # If filter_str is empty, sample random conversations
        if filter_str == "":
            result_df = self.extract_sample_conversations(50)
        urls = self.parquet_urls.copy()
        random.shuffle(urls)
        
        result_df = pd.DataFrame()

        for url in urls:
            print(f"Querying file: {url}")
            r = self._safe_get(url)
            if r == None:
                print(f"Timeout occurred for GET {url}. Skipping file {url}.")
                continue
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            try:
                query_str = f"""
                    SELECT * FROM read_parquet('{tmp_path}') 
                    WHERE contains(lower(cast(conversation as VARCHAR)), lower('{filter_str}'))
                    """
                df = duckdb.query(query_str).df()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            print(f"Found {len(df)} result(s) in {url.split('/')[-1]}")
            
            if len(df) > 0:
                result_df = pd.concat([result_df, df], ignore_index=True)
                
            if len(result_df) >= min_results:
                break
        if len(result_df) == 0:
            print("No results found. Returning empty DataFrame.")
            placeholder_row = {'conversation_id': "No result found",
                               'model': "-",
                               'conversation': [
                                {'content': '-', 'role': 'user'},
                                {'content': '-', 'role': 'assistant'}
                               ],
                               'turn': "-",
                               'language': "-",
                               'openai_moderation': "[{'-': '-', '-': '-'}]",
                               'redacted': "-",}
            result_df = pd.DataFrame([placeholder_row])
            print(result_df)
        self.active_df = result_df
        try:
            self.active_conversation = Conversation(self.active_df.iloc[0])
        except Exception as e:
            print(f"No conversations available: {e}")
        return result_df
    
    def create_conversations_index(self, output_index_file="json/conversations_index.json"):
        """
        Builds an index of conversation IDs from a list of Parquet file URLs.
        Stores the index as a JSON mapping conversation IDs to their respective file names.
        """
        index = {}

        for url in self.parquet_urls:
            file_name = url.split('/')[-1]  # Extract file name from URL
            print(f"Indexing file: {file_name}")

            try:
                # Download the file temporarily
                r = requests.get(url, headers=self.headers)
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp.write(r.content)
                    # tmp.flush()
                    tmp_path = tmp.name
                try:
                    query = f"SELECT conversation_id FROM read_parquet('{tmp_path}')"
                    df = duckdb.query(query).to_df()
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                # Map conversation IDs to file name (not the full URL)
                for _, row in df.iterrows():
                    index[row["conversation_id"]] = file_name

            except Exception as e:
                print(f"Error indexing {file_name}: {e}")

        # Save index for fast lookup
        with open(output_index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        return output_index_file


class Conversation:
    def __init__(self, data):
        """
        Initialize a conversation object either from conversation data directly or from a DataFrame row.
        
        Parameters:
        - data: Can be either a list of conversation messages or a pandas Series/dict containing conversation data
        """
        # Handle both direct conversation data and DataFrame row
        if isinstance(data, (pd.Series, dict)):
            # Store all metadata separately
            self.conversation_metadata = {}
            for key, value in (data.items() if isinstance(data, pd.Series) else data.items()):
                if key == 'conversation':
                    self.conversation_data = value
                else:
                    self.conversation_metadata[key] = value
        else:
            # Direct initialization with conversation data
            self.conversation_data = data
            self.conversation_metadata = {}

    def add_turns(self):
        """
        Adds a 'turn' key to each dictionary in the conversation,
        identifying the turn (pair of user and assistant messages).

        Returns:
        - list: The updated conversation with 'turn' keys added.
        """
        turn_counter = 0
        for message in self.conversation_data:
            if message['role'] == 'user':
                turn_counter += 1
            message['turn'] = turn_counter
        return self.conversation_data
    
    def pretty_print(self, user_prefix, assistant_prefix, width=80):
        """
        Prints the conversation with specified prefixes and wrapped text.

        Parameters:
        - user_prefix (str): Prefix to prepend to user messages.
        - assistant_prefix (str): Prefix to prepend to assistant messages.
        - width (int): Maximum characters per line for wrapping.
        """
        wrapper = textwrap.TextWrapper(width=width)
        
        for message in self.conversation_data:
            if message['role'] == 'user':
                prefix = user_prefix
            elif message['role'] == 'assistant':
                prefix = assistant_prefix
            else:
                continue  # Ignore roles other than 'user' and 'assistant'
            
            # Split on existing newlines, wrap each line, and join back with newlines
            wrapped_content = "\n".join(
                wrapper.fill(line) for line in message['content'].splitlines()
            )
            print(f"{prefix} {wrapped_content}\n")