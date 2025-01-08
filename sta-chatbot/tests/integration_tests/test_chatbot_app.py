import pytest
import logging
import requests
import json
import re
from databricks.sdk import WorkspaceClient

@pytest.mark.integration
def contains_html_tags(text_to_check: str):
    """
    Check if argument string has HTML tags

    Args:
        textToCheck (str): String to check for HTML tags

    Returns:
        bool: True if string is HTML. Returns False otherwise.
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Checking if string parameter has any HTML tags")
    return bool(re.search(r'<[^>]+>', text_to_check))

#Pytest fixtures (data for testing)
@pytest.fixture(params=[
    {
        "messages": [
            {"content": "En quin format s’ha de facilitar l’accés a la informació pública?", "role": "user"},
            {"content": "En format reutilitzable.", "role": "assistant"},
            {"content": "Si us plau, expliqueu la resposta amb més detalls.", "role": "user"}
        ],
        "selected_documents": [
            "1920373 Llei 19-2013, de 9 de desembre, de transparència, accés a la informació pública i bon govern.pdf",
            "2017238 DECRET 8-2021, de 9 de febrer, sobre la transparència i el dret d'accés a la informació pública.pdf"
        ]
    }
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
def test_app_endpoint(sample_data):

    """
    Calls chatbot app endpoint (POST) and tests response content

    Function's purpose is to test the endpoint of the chatbot app
    and assert the response contents.

    Args:
        sample_data (list): Sample data received as fixture from pytest, set up previously
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
    url = f"{workspace.config.host}/serving-endpoints/chatbot_app/invocations"
    headers = {'Authorization': f'Bearer {workspace.config.token}', 'Content-Type': 'application/json'}

    #Dict to JSON for making POST request
    logger.info("Jsonifying sample data")
    data_json = json.dumps(sample_data, allow_nan=True)
    
    try:
        #POST request for chatbot app endpoint
        logger.info("Sending POST request to target URL, with respective headers and data")
        response = requests.post(url=url, headers=headers, data=data_json)

        #response to JSON
        logger.info("Capturing response into JSON format")
        vector_search_response = response.json()

        print(json.dumps(vector_search_response, indent=4))

        logger.info("Handling possible Request exceptions and other exceptions (HTTP error).")

        #Assert response conents and format
        logger.info("Asserting if endpoint response content is of type `str`")
        assert isinstance(vector_search_response, str), "Chatbot app answer should be string"

        logger.info("Asserting if endpoint response includes HTML tags")
        assert contains_html_tags(vector_search_response), "Chatbot app answer is not HTML"
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the endpoint: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")