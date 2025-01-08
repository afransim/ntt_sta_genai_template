import pytest
import logging
import requests
import json
from databricks.sdk import WorkspaceClient

@pytest.mark.integration
def get_embedding_size():

    """
    Calls vector search endpoint (GET) to retrieve embedding vector size

    Returns:
        Int: Embedding vector size from Vector Search Index endpoint

    Raises:
        requests.exceptions.RequestException: In case of any errors regarding connection with API endpoint
        Exception: For remaining unexpected errors

    """
    #Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    #Databricks workspace client
    logger.info("Connecting to Databricks Workspace client- accessing host and credentials to retrieve embedding size")
    workspace = WorkspaceClient()

    #Set up URL and Headers for request
    logger.info("Setting up endpoint URL and headers")
    url = f"{workspace.config.host}/api/2.0/vector-search/indexes/admin_govern_sta_des.gld_chatbot.ftr_ctti_docs_idx"
    headers = {'Authorization': f'Bearer {workspace.config.token}'}

    try:
        #GET request vector search index endpoint
        logger.info("Sending GET request to access embedding endpoint")
        response = requests.get(url=url, headers=headers)
        
        #response to JSON
        vector_search_response = response.json()
        logger.info(f"Jsonifying sample data: {vector_search_response}")

        #return embedding size
        logger.info("Getting the embedding size")
        return vector_search_response["direct_access_index_spec"]["embedding_vector_columns"][0]["embedding_dimension"]
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@pytest.mark.integration
def create_query_dict(num_results):

    """
    Function to create dummy data to call and test Vector Search Index endpoint.

    Args:
        num_results (int): Number of results to be asked from embedding endpoint

    Returns:
        Dict: Dummy dictionary as content for POST request
    """

    #Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    #Get embedding size
    logger.info("Getting the embedding vector size for creating the dummy data for the test")
    query_vector_size = int(get_embedding_size())

    #Create dummy unit vector for testing vector search index endpoint
    logger.info("Instantiating the dummy data for testing")
    query_vector = [1] * query_vector_size
    return {
        "num_results": num_results,
        "columns": ["chunk_embedding"],
        "query_vector": query_vector
    }

@pytest.fixture
@pytest.mark.integration
def post_data():
    """
    Data sampling function for testing (fixture).

    Args:
        request (list): List of fixture parameters, for testing

    Returns:
        Dict: Fixture parameters for testing
    """
    #Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    logger.info("Creating data sampling for testing")
    return create_query_dict(3)

@pytest.mark.integration
def test_vector_search_index_endpoint(post_data):

    """
    Calls Vector Search Index endpoint (POST) and tests response content

    Args:
        sample_data (list): Sample data received as fixture from pytest, set up previously
        in this file

    Raises:
        requests.exceptions.RequestException: In case of any errors regarding connection with API endpoint
        Exception: For remaining unexpected errors

    """
    #Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    #Databricks workspace client
    logger.info("Connecting to Databricks Workspace client- accessing host and credentials")
    workspace = WorkspaceClient()

    #Set up URL and Headers for request
    logger.info("Setting up endpoint URL and headers")
    url = f"{workspace.config.host}/api/2.0/vector-search/indexes/admin_govern_sta_des.gld_chatbot.ftr_ctti_docs_idx/query"
    headers = {'Authorization': f'Bearer {workspace.config.token}', 'Content-Type': 'application/json;charset=UTF-8', \
               'Accept': 'application/json, text/plain, */*'}

    #Dict to JSON for making POST request
    logger.info("Jsonifying sample data")
    data_json = json.dumps(post_data, allow_nan=True)

    try:
        #POST request for vector search index endpoint
        logger.info("Sending POST request to target URL, with respective headers and data")
        response = requests.post(url=url, headers=headers, data=data_json)

        #response to JSON
        logger.info("Capturing response into JSON format")
        vector_search_response = response.json()

        print(json.dumps(vector_search_response, indent=4))

        #Assert response contents and format
        logger.info("Asserting that the vector search index endpoint returns the expected fields and formatted correctly")
        assert ("column_count" in vector_search_response["manifest"]) and \
            ("columns" in vector_search_response["manifest"]), \
                "Manifest field in response not formatted correctly"

        assert ("row_count" in vector_search_response["result"]) and \
            ("data_array" in vector_search_response["result"]), \
                "Result field in response not formatted correctly"

        assert (len(vector_search_response["result"]["data_array"]) == vector_search_response["result"]["row_count"]) and \
            (len(vector_search_response["result"]["data_array"][0]) == vector_search_response["manifest"]["column_count"]) and \
            (len(vector_search_response["result"]["data_array"][0][0]) > 0) and \
            (isinstance(vector_search_response["result"]["data_array"][0][1], float)), \
                "Vector search data is not formatted correctly"
    
        logger.info("Catching possible RequestException errors and other exceptions")
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
