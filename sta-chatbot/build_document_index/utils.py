from databricks.sdk.runtime import spark, dbutils
from databricks.sdk import WorkspaceClient
import time

def list_files(dbutils, directory: str):

    """
    Recursively lists all files in a given directory and its subdirectories.

    Args:
        dbutils: A utility object for interacting with the file system, in Databricks.
        directory (str): The root directory to start listing files from.

    Returns:
        List[str]: A list of file paths found in the directory and its subdirectories.

    Raises:
        Prints an error message if there's an issue accessing the directory.
    """

    files = []
    try:
        items = dbutils.fs.ls(directory)
        for item in items:
            if item.isDir():
                files.extend(list_files(dbutils, item.path))
            else:
                files.append(item.path)
    except Exception as e:
        print(f"Error accessing files in directory {directory}: {e}")
    return files

def flatten_directory_copy(source_path, flatten_target_path):
    """
    Copies all files from the source directory and its subdirectories 
    to a flat structure in the target directory.

    Args:
    source_path (str): The source directory path.
    flatten_target_path (str): The target directory path where files will be copied without maintaining the original structure.
    """

    # Get all files in the source directory
    all_files = list_files(dbutils, source_path)

    # Copy each file to the target directory (flattening structure)
    for file_path in all_files:
        file_name = file_path.split("/")[-1]  # Extract the file name
        # Ensure the target path is correctly formatted for DBFS or a mounted volume
        dbutils.fs.cp(file_path, f"{flatten_target_path}/{file_name}")

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from pypdf import PdfReader
def extract_text_with_text_loader(file_path):

    """
    Extract text/content from a text file (.txt or .md)

    Args:
        file_path (str): Path to text file
    
    Returns:
        str: Return document's text content
    """

    loader = TextLoader(file_path)
    data = loader.load()
    return data[0].page_content

def extract_text_with_microsoft_word_loader(file_path):

    """
    Extract text/content from a microsoft word file

    Args:
        file_path (str): Path to microsoft word file
    
    Returns:
        str: Return microsoft word document's text content
    """

    loader = Docx2txtLoader(file_path)
    data = loader.load()
    return data[0].page_content

def extract_text_with_pdf_reader(file_path):

    """
    Extract text/content from a PDF file

    Args:
        file_path (str): Path to PDF file
    
    Returns:
        str: Return PDF document's text content
    """

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

import os
def extract_text(file_path):

    """
    Function to identify file extension (.pdf, .docx, or .txt/.md) and apply 
    the appropriate text extracting function.

    Args:
        file_path (str): Path to file
    
    Returns:
        str: Return document text content. None if file extension is not .pdf, .docx, .txt or .md.
    """

    _, path_extension = os.path.splitext(file_path.lower())
    if path_extension == ".pdf":
        return extract_text_with_pdf_reader(file_path)
    elif path_extension == ".md" or path_extension == ".txt":
        return extract_text_with_text_loader(file_path)
    elif path_extension == ".docx":
        return extract_text_with_microsoft_word_loader(file_path)
    else:
        print("ERROR: ", path_extension, "file extension not allowed.")
        return None
    
def get_chunks_with_metadata(text, recursive_splitter):
        
        """
        Function to extract chunk, chunk size, chunk location and total chunks in the document

        Args:
            text (str): Text content of document
            recursive_splitter (langchain.text_splitter.RecursiveCharacterTextSplitter):
        
        Returns:
            List[str]: List of strings with, respectively, chunk content, chunk size, chunk location and total chunks in document
        """

        chunks = recursive_splitter.split_text(text)
        return [(chunk, len(chunk), i, len(chunks)) for i, chunk in enumerate(chunks)]
import sys
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch
import time

def handle_retry_exception(e, attempt, max_retries, debug_operation):
    if attempt < max_retries - 1:
        print(f"While {debug_operation}: Attempt {attempt + 1} failed: {e}. Retrying...")
        time.sleep(60)#2 ** attempt)  # Exponential backoff
    else:
        print(f"All {max_retries} attempts failed.")
        raise

def get_vector_search(index, embedding_endpoint_name):
    max_retries = 5
    for attempt in range(max_retries):
        try:    
            vs = DatabricksVectorSearch(
                index=index,
                embedding=DatabricksEmbeddings(endpoint=embedding_endpoint_name),
                text_column="text_chunk"
            )
            return vs
        except Exception as e:
            handle_retry_exception(e, attempt, max_retries, "retrieving Vector Search")

def upsert_chunks(vs, chunks_texts, chunks_ids, chunks_metadata):
    # Retry logic for transient errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            vs.add_texts(
                texts=chunks_texts,
                ids=chunks_ids,
                metadatas=chunks_metadata
            )
            break
        except Exception as e:
            handle_retry_exception(e, attempt, max_retries, "adding text to Vector Search")

def upsert_chunks_into_index(chunks_list, current_chunk_it, index, embedding_endpoint_name):

    """
    Upserting chunks into Vector search table index

    Args:
        chunk_list (List[dict]): List of dictionaries, each containing a row from the chunks' table
        current_chunk_it (int): Starting location of chunk in table. Starting iterator. If whole table, then 0.
        index (databricks.vector_search.index.VectorSearchIndex): Table index object obtained from vector search client
        embedding_endpoint_name (str): Name of embedding endpoint
    
    Raises:
        Exception: Raises any exception that occurs while accessing the vector search object from databricks, as well as
         adding nwe chunk information into the vector search.

    """
    
    UPSERT_BYTES_SIZE_LIMIT = 10000
    chunks_ids = []
    chunks_texts = []
    chunks_metadata = []
    retry = False
    print(f"Indexing chunks info from chunk {current_chunk_it}")
    for i in range(current_chunk_it, len(chunks_list)):
        chunk_id = chunks_list[i]['id']
        chunk_text = chunks_list[i]['text_chunk']
        chunk_metadata = {
            'file_name': chunks_list[i]['file_name'],
            'folder_name': chunks_list[i]['folder_name'],
            'document_type': chunks_list[i]['document_type'],
            'chunk_location_in_doc': chunks_list[i]['chunk_location_in_doc'],
            'total_chunks_in_doc': chunks_list[i]['total_chunks_in_doc'],
            'chunk_length_in_chars': chunks_list[i]['chunk_length_in_chars']
        }

        upsert_request_size = (
            sys.getsizeof(chunks_ids) +
            sys.getsizeof(chunks_texts) + 
            sys.getsizeof(chunks_metadata) +
            sys.getsizeof(chunk_id) +
            sys.getsizeof(chunk_text) +
            sys.getsizeof(chunk_metadata)
        )
        if upsert_request_size >= UPSERT_BYTES_SIZE_LIMIT:
            retry = True
            current_chunk_it = i
            break
        else:
            chunks_ids.append(chunk_id)
            chunks_texts.append(chunk_text)
            chunks_metadata.append(chunk_metadata)

    vs = get_vector_search(index, embedding_endpoint_name)

    upsert_chunks(vs, chunks_texts, chunks_ids, chunks_metadata)

    if retry:
        upsert_chunks_into_index(chunks_list=chunks_list, current_chunk_it=current_chunk_it, index=index, embedding_endpoint_name=embedding_endpoint_name)

##############################################################################################################################################################################
# ENDPOINTS FUNCTIONS
##############################################################################################################################################################################
import requests
import json

def endpoint_exists(serving_name):
  """
    Check if an endpoint with the serving_endpoint_name exists

    Args:
        serving_name (str): Serving endpoint's name

    Returns:
        bool: True if serving endpoints exists. False otherwise. 

  """
  w = WorkspaceClient()
  try:
    w.serving_endpoints.get(name=serving_name)
    return True
  except Exception as e:
      return False

def update_endpoint(serving_name, served_entities):

  """
    Update endpoint

    Args:
        serving_name (str): Serving endpoint's name
        served_entities (Array<ServedEntityInput>): Array containing served entities classes
      
    Raises:
        HTTPError: Raises any HTTP error while calling the endpoint

  """

  w=WorkspaceClient()

  print(f"Updating config for serving endpoint: {serving_name}")
  w.serving_endpoints.update_config_and_wait(
        name=serving_name,
        served_entities=served_entities
    )
  print(f"Serving endpoint {serving_name} updated.")
        

def create_endpoint(serving_name, config, tags):
  """
    Create serving endpoint and wait for it to be ready

    Args:
        serving_name (str): Serving endpoint's name
        config (EndpointCoreConfigInput): Class containing required info for creating serving endpoint (Databricks SDK)
        tags(Array<EndpointTag>): List of EndpointTag classes
    
    Raises:
        requests.exceptions.HTPPError: Raises any HTTP error that might have occurred while creating the endpoint

  """
  w = WorkspaceClient()

  print(f"Creating new serving endpoint: {serving_name}")

  try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=serving_name,
        config=config,
        tags=tags
    )
    print(f"Serving endpoint '{serving_name}' created successfully.")
    print(f"Details: {endpoint}")
  except Exception as e:
    print(f"Error creating serving endpoint: {e}")
    raise

#########################################################################################################################################
#VECTOR SEARCH ENDPOINT 
########################################################################################################################################

def list_vs_endpoints(w):

    """
    List all vector search endpoints.

    Args:
        w (databricks.sdk.WorkspaceClient): Workspace-level Databricks REST API client object

    Returns:
        vsc.list_endpoints: All endpoints of specified vector search client

    """
    return list(w.vector_search_endpoints.list_endpoints())

def get_vs_endpoints(w, vector_search_endpoint):
    """
    Retrieve an existing vector search endpoint.

    Args:
        w (databricks.sdk.WorkspaceClient): Workspace-level Databricks REST API client object
        vector_search_endpoint (str): Name of vector search endpoint

    Returns:
        vsc.get_endpoint: Returns specified endpoint information

    """

    return w.vector_search_endpoints.get_endpoint(endpoint_name=vector_search_endpoint)

def create_vs_endpoint_and_wait(w, vector_search_endpoint, timeout=30):
    """
    Create a new vector search endpoint.

    Args:
        w (databricks.sdk.WorkspaceClient): Workspace-level Databricks REST API client object
        vector_search_endpoint (str): Name of vector search endpoint

    """
    from databricks.sdk.service.vectorsearch import EndpointType
    from datetime import timedelta

    w.vector_search_endpoints.create_endpoint_and_wait(name=vector_search_endpoint, endpoint_type=EndpointType.STANDARD, timeout=timedelta(minutes=timeout))


def create_or_select_vs_endpoint(w, vector_search_endpoint):
    """
    Create or select a vector search endpoint.

    Args:
        w (databricks.sdk.WorkspaceClient): Workspace-level Databricks REST API client object
        vector_search_endpoint (str): Name of vector search endpoint

    Raises:
        Exception: Raises any exception troughout function and details during a failed creation of Vector Search Endpoint

    """

    try:
        existing_endpoints = list_vs_endpoints(w)
        if (existing_endpoints and (any(ep.name == vector_search_endpoint for ep in existing_endpoints))):
            endpoint = get_vs_endpoints(w, vector_search_endpoint)
            print(f"Vector search endpoint '{vector_search_endpoint}' found and selected.")
            return endpoint
        else:
            print(f"Vector search endpoint '{vector_search_endpoint}' does not exist. Creating a new one.")
            create_vs_endpoint_and_wait(w, vector_search_endpoint)
            print(f"Vector search endpoint '{vector_search_endpoint}' created and selected.")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

##################################################################################################################################################################################################################################################################
#VECTOR SEARCH INDEX
##################################################################################################################################################################################################################################################################
import logging
import time

def list_indexes(vsc, vector_search_endpoint_name):
    """
    List existing indexes for a given vector search endpoint.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint

    Returns:
        vsc.list_indexes: Returns all indexes of the specified certain vector search endpoint

    """

    return vsc.list_indexes(name=vector_search_endpoint_name)

def check_index_exists(vsc, vector_search_endpoint_name, index_name):
    """
    Check if a specific index exists.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index to look for

    Returns:
        bool: True if any index is named index_name. False otherwise

    """

    existing_indexes = list_indexes(vsc, vector_search_endpoint_name)
    return any(index['name'] == index_name for index in existing_indexes.get("vector_indexes", []))

def describe_table_schema(catalog, schema, table):
    """
    Describe the schema of a table.

    Args:
        catalog (str): Name of catalog
        schema (str): Name of schema
        table (str): Name of table

    Returns:
        dict: Dictionary with the name of each column of the table and its respective data type.

    """

    schema_df = spark.sql(f"DESCRIBE {catalog}.{schema}.{table}")
    return {row['col_name']: row['data_type'] for row in schema_df.collect()}

def create_index(vsc, vector_search_endpoint_name, index_name, schema_dict, embedding_endpoint_name, retries=3):
    """
    Create a new index, retrying on internal errors.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index
        schema_dict (dict): Dictionary with the name of each column of the table and its respective data type.
        embedding_endpoint_name (str): Name of embedding endpoint
        retries (int): Number of retried

    Returns:
        bool: Returns True if function was executed within the number of retries. False otherwise.

    """

    while retries > 0:
        try:
            vsc.create_direct_access_index(
                endpoint_name=vector_search_endpoint_name,
                index_name=index_name,
                primary_key="id",
                embedding_dimension=3072,
                embedding_vector_column="chunk_embedding",
                schema=schema_dict,
                embedding_model_endpoint_name=embedding_endpoint_name
            )
            print(f"Index '{index_name}' created successfully.")

            return True
        except Exception as e:
            if hasattr(e, 'error_code'):
                if e.error_code == 'INTERNAL_ERROR':
                    logging.error(f"Exception: {e} \nRetrying...")
                    time.sleep(20)
                    retries -= 1
                elif e.error_code == 'RESOURCE_DOES_NOT_EXIST':
                    handle_index_creation_delay(vsc, vector_search_endpoint_name, index_name)
                    return True
    return False

def handle_index_creation_delay(vsc, vector_search_endpoint_name, index_name, timeout=1200):
    """
    Handle delays in index creation by polling until the index is created or timeout.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index
        timeout (int): Time out value in seconds

    Returns:
        bool: True if index was created successfully. False otherwise.

    Raises:
        Exception: Raises exception if function execution time exceeds timeout threshold.

    """

    print("Vector Search Index being created. Waiting...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=index_name).describe()['status']['detailed_state']
            print(f"Index '{index_name}' created successfully.")
            return True
        except Exception as e:
            if hasattr(e, 'error_code') and e.error_code == 'RESOURCE_DOES_NOT_EXIST':
                logging.error(f"Exception: {e} \nIndex not set up yet. Waiting...")
                time.sleep(60)
    raise TimeoutError(f"Timeout: Index creation exceeded time out of {timeout} seconds.")

def wait_for_index_online(vsc, vector_search_endpoint_name, index_name, timeout=600):
    """
    Wait until the index status becomes 'ONLINE_DIRECT_ACCESS'.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index
        timeout (int): Time out value in seconds

    Returns:
        bool: True if index is online. False otherwise

    Raises:
        Exception: Raises exception if function execution time exceeds timeout threshold.

    """

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            index_status = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=index_name).describe()['status']['detailed_state']
            if index_status == "ONLINE_DIRECT_ACCESS":
                return True
            time.sleep(10)
        except Exception as e:
            if hasattr(e, 'error_code') and e.error_code == 'INTERNAL_ERROR':
                logging.error(f"Exception: {e} \nRetrying...")
                time.sleep(20)
    raise TimeoutError(f"Timeout: Index status check exceeded time out of {timeout} seconds.")

def fetch_index(vsc, vector_search_endpoint_name, index_name, timeout=600):
    """
    Get index from vector search client

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index
        timeout (int): Time out value in seconds

    Returns:
        databricks.vector_search.index.VectorSearchIndex: Returns index if it is found.

    Raises:
        Exception: Raises exception if function execution time exceeds timeout threshold.

    """

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=index_name)
            return index
        except Exception as e:
            if hasattr(e, 'error_code') and e.error_code == 'INTERNAL_ERROR':
                logging.error(f"Exception: {e} \nRetrying...")
                time.sleep(20)
    raise TimeoutError(f"Timeout: Index status check exceeded time out of {timeout} seconds.")

def fill_index_with_data(catalog, schema, table_chunks, vsc, vector_search_endpoint_name, index_name, embedding_endpoint_name):
    """
    Fill the specified index with data from the given table.

    Args:
        catalog (str): Name of catalog
        schema (str): Name of schema
        table_chunks (str): Name of the table containing the chunks
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        index_name (str): Name of index
        embedding_endpoint_name (str): Name of embedding endpoint

    """

    df = spark.sql(f"SELECT * FROM {catalog}.{schema}.{table_chunks}")
    chunks_list = [row.asDict() for row in df.collect()]
    index = fetch_index(vsc, vector_search_endpoint_name, index_name)
    upsert_chunks_into_index(chunks_list, 0, index, embedding_endpoint_name)
    print(f"Index '{index_name}' filled in successfully.")

def update_vs_endpoint(vsc, vector_search_endpoint_name, embedding_endpoint_name, catalog, gld_schema, slv_schema, table_idx, table_chunks):
    """
    Update the vector search endpoint by creating or updating an index and filling it with data.

    Args:
        vsc (databricks.vector_search.client.VectorSearchClient): Vector search client object to access list_indexes() method.
        vector_search_endpoint (str): Name of vector search endpoint
        embedding_endpoint_name (str): Name of embedding endpoint
        catalog (str): Name of catalog
        gld_schema (str): Name of gold schema
        slv_schema (str): Name of silver schema
        talbe_idx (str): Name of index table
        table_chunks (str): Name of the table containing the chunks
    
    Raises:
        Exception: If index creation has been failed.

    """

    print(f"vector_search_endpoint_name: {vector_search_endpoint_name}, embedding_endpoint_name: {embedding_endpoint_name}")
    
    index_name = f"{catalog}.{gld_schema}.{table_idx}"
    print(f"index_name: {index_name}")

    if check_index_exists(vsc, vector_search_endpoint_name, index_name):
        print(f"Index '{index_name}' already exists.")
    else:
        print(f"Creating index '{index_name}'...")
        schema_dict = describe_table_schema(catalog, slv_schema, table_chunks)
        schema_dict['chunk_embedding'] = 'array<float>'
        if not create_index(vsc, vector_search_endpoint_name, index_name, schema_dict, embedding_endpoint_name):
            raise RuntimeError(f"Failed to create index '{index_name}'.")

    wait_for_index_online(vsc, vector_search_endpoint_name, index_name)
    fill_index_with_data(catalog, slv_schema, table_chunks, vsc, vector_search_endpoint_name, index_name, embedding_endpoint_name)