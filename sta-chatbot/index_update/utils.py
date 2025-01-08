import os
import sys
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from pypdf import PdfReader
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch



# List all files (can contain both already uploaded and new files)
def list_files(dbutils ,directory: str):
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

# Get text from new files and add it to text table

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
        return None

def trim_path(path):

    """
    Trims given path

    Args:
        path (str): Path to trim
        
    Returns:
        string: Returns specified path from the 5th place onwards, 
                 delimited by the OS specific separator (ex. '\' for Windows and '/' for Unix-based OS)

    """

    parts = path.split(os.sep)
    return os.sep.join(parts[5:])

def upsert_chunks_into_index(index, embedding_endpoint, chunks_list, current_chunk_it=0):
    """
    Upserting chunks into Vector search table index

    Args:
        index (databricks.vector_search.index.VectorSearchIndex): Table index object obtained from vector search client
        embedding_endpoint_name (str): Name of embedding endpoint
        chunk_list (List[dict]): List of dictionaries, each containing a row from the chunks' table
        current_chunk_it (int): Starting location of chunk in table. Starting iterator. If whole table, then 0.
        
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

    vs = DatabricksVectorSearch(
        index=index,
        embedding=DatabricksEmbeddings(endpoint=embedding_endpoint),
        text_column="text_chunk"
    )
    
    try: 
        vs.add_texts(
            texts=chunks_texts,
            ids=chunks_ids,
            metadatas=chunks_metadata
        )
    except Exception as e:
        print(f"Chunk could not be added to the index: {e}")
    
    if retry:
        upsert_chunks_into_index(index, embedding_endpoint, chunks_list, current_chunk_it=current_chunk_it)