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
        vectorSearchResponse = response.json()
        logger.info(f"Jsonifying sample data: {vectorSearchResponse}")

        #return embedding size
        logger.info("Getting the embedding size")
        return vectorSearchResponse["direct_access_index_spec"]["embedding_vector_columns"][0]["embedding_dimension"]
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@pytest.fixture(params=[
    {"input": "Sample string to query the embedding model."},
    {"input": ""},
    {"input": "dfa;/134 12&   fda%5#  fgsf//''dfsgaw 23r 3 4532"}
])
@pytest.mark.integration
def sample_data(request):
    """
    Data sampling function for testing (fixture).

    Args:
        request (list): List of fixture parameters, for testing

    Returns:
        Dict: Fixture parameters for testing
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Returning the data sampling parametrization for testing")
    return request.param

@pytest.mark.integration
def test_embedding_endpoint(sample_data):

    """
    Calls embedding endpoint (POST) and asserts response content

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
    logger.info("Connecting to Databricks Workspace client- accessing host and credentials for embedding endpoint testing")
    workspace = WorkspaceClient()

    #Get embedding vector size
    logger.info("Retrieving embedding size")
    embedding_vector_size = get_embedding_size()

    #Set up URL and Headers for request
    logger.info("Setting up endpoint URL and headers")
    url = f"{workspace.config.host}/serving-endpoints/chatbot_embeddings/invocations"
    headers = {'Authorization': f'Bearer {workspace.config.token}', 'Content-Type': 'application/json'}

    #Dict to JSON for making POST request
    logger.info("Jsonifying sample data")
    data_json = json.dumps(sample_data, allow_nan=True)

    try:
        #POST request for embedding endpoint
        logger.info("Sending POST request to target URL, with respective headers and data")
        response = requests.post(url=url, headers=headers, data=data_json)

        #response to JSON
        logger.info("Capturing response into JSON format")
        vectorSearchResponse = response.json()

        #Assert response conents and format
        logger.info("Asserting that the embedding endpoint returns the expected fields and formatted correctly")
        assert ("object" in vectorSearchResponse) and \
                ("data" in vectorSearchResponse)and \
                ("object" in vectorSearchResponse["data"][0]) and \
                ("index" in vectorSearchResponse["data"][0]) and \
                ("embedding" in vectorSearchResponse["data"][0]) and \
                ("model" in vectorSearchResponse) and \
                ("usage" in vectorSearchResponse) and \
                ("prompt_tokens" in vectorSearchResponse["usage"]) and \
                ("total_tokens" in vectorSearchResponse["usage"]), "Embedding endpoint response not correctly formatted"

        logger.info(f"Asserting the embedding vector size: {vectorSearchResponse['data'][0]['embedding']} is equal to the one retrieved : {embedding_vector_size}")
        assert len(vectorSearchResponse["data"][0]["embedding"]) == embedding_vector_size, "Invalid embedding vector size"
    
        logger.info("Catching possible RequestException errors and other exceptions")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
