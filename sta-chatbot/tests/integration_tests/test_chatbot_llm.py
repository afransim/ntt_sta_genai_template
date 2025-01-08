import pytest
import logging
import requests
import json
from databricks.sdk import WorkspaceClient

#Pytest fixtures (data for testing)
@pytest.fixture(params=[
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
            {"role": "user", "content": "What is Databricks?"}
        ],
        "max_tokens": 128
    },
    {
        "messages": [
            {"role": "user", "content": "Good morning!"},
            {"role": "assistant", "content": "Good morning! How can I help you today?"},
            {"role": "user", "content": "Tell me about machine learning."}
        ],
        "max_tokens": 128
    }
])
@pytest.mark.integration
def sample_data(request):
    """
    Data sampling function for testing (fixture).

    Args:
        request (List[dict]): List of fixture parameters, for testing

    Returns:
        dict: Fixture parameters for testing
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Returning the data sampling parametrization for testing")
    return request.param

@pytest.mark.integration
def test_chat_llm_endpoint(sample_data):
    """
    Calls chatbot llm endpoint (POST) and tests response content

    Function's purpose is to test the endpoint of the chatbot llm
    and assert the response contents.

    Args:
        sample_data (List[dict]): Sample data received as fixture from pytest, set up previously
        in this file

    Raises:
        requests.exceptions.RequestException: In case of any errors regarding connection with API endpoint
        Exception: For remaining unexpected errors

    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    #Databricks workspace client
    logger.info("Connecting to Databricks Workspace client- accessing host and credentials")
    workspace = WorkspaceClient()

    #Set up URL and Headers for request
    logger.info("Setting up endpoint URL and headers")
    url = f"{workspace.config.host}/serving-endpoints/chatbot_llm/invocations"
    headers = {'Authorization': f'Bearer {workspace.config.token}', 'Content-Type': 'application/json'}

    #Dict to JSON for making POST request
    logger.info("Jsonifying sample data")
    data_json = json.dumps(sample_data, allow_nan=True)
    
    try:
        #POST request for chatbot llm endpoint
        logger.info("Sending POST request to target URL, with respective headers and data")
        response = requests.post(url=url, headers=headers, data=data_json)

        #response to JSON
        logger.info("Capturing response into JSON format")
        vectorSearchResponse = response.json()

        logger.info("Handling possible Request exceptions and other exceptions (HTTP error).")

        print(json.dumps(vectorSearchResponse, indent=4))

        logger.info("Asserting that the LLM endpoint responds with the expected fields and formatted correctly")
        #Assert response conents and format
        assert ("content_filter_results" in vectorSearchResponse["choices"][0]) and \
            ("message" in vectorSearchResponse["choices"][0]) and \
            ("model" in vectorSearchResponse) and \
            ("usage" in vectorSearchResponse) and \
            ("object" in vectorSearchResponse), "LLM API response is not formatted correctly"
        
        logger.info("Asserting the `content` field and role within the endpoint choices[0].mesage field")
        assert ("content" in vectorSearchResponse["choices"][0]["message"]) and \
            ("role" in vectorSearchResponse["choices"][0]["message"]), "LLM message is not formatted correctly"
        
        logger.info("Asserting that the LLM content is not empty")
        assert (len(vectorSearchResponse["choices"][0]["message"]["content"]) > 0), "LLM message content is empty"
    
        logger.info("Catching possible RequestException errors and other exceptions")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")