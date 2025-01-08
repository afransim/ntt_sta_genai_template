import pytest
import logging
from unittest.mock import MagicMock, patch
from mlflow.tracking import MlflowClient

from evaluate_model.utils import get_model_run, get_latest_model_version, extract_only_answer_from_html

@pytest.mark.unit
@patch('evaluate_model.utils.MlflowClient')
def test_get_model_run(mock_client):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create a mock instance of MlflowClient
    logger.info("Mocking MlflowClient object")
    mock_client = mock_client.return_value
    
    # Mock the get_model_version response
    logger.info("Mocking model_version_details that results from get_model_version function from MlflowCient")
    mock_model_version_details = MagicMock()
    mock_model_version_details.run_id = 'test_run_id'
    mock_client.get_model_version.return_value = mock_model_version_details
    
    # Mock the get_run response
    logger.info("Mocking run details response content")
    mock_run_details = MagicMock()
    mock_client.get_run.return_value = mock_run_details
    
    # Call the function under test
    logger.info("Running the get_model_run from MlflowClient")
    model_name = 'test_model'
    model_version = 1
    run_id = get_model_run(model_name, model_version)
    
    # Assert that the correct run_id is returned
    logger.info(f"Assert if the resulting run_id: {run_id} is as expected: test_run_id")
    assert run_id == 'test_run_id'
    
    # Ensure that get_model_version was called with the correct arguments
    logger.info("Asserting if get_model_version function and get_run were called at least once")
    mock_client.get_model_version.assert_called_once_with(name=model_name, version=model_version)
    
    # Ensure that get_run was called with the correct run_id
    mock_client.get_run.assert_called_once_with('test_run_id')

@patch.object(MlflowClient, 'search_model_versions')
@pytest.mark.unit
def test_get_latest_model_version(mock_search_model_versions):
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Mocking the serach_model_versions function from MlflowClient")
    mock_search_model_versions.return_value = [
        MagicMock(version='1'),
        MagicMock(version='2'),
        MagicMock(version='3')
        ]
    
    expected_version = 3

    # Call the function under test
    logger.info("Running the get_latest_model_version function from MlflowClient")
    model_name = 'rag_test_model'
    actual_version = get_latest_model_version(model_name)

    # Assert that the correct latest version is returned
    logger.info(f"Asserting if the resulting version: {actual_version} is as expected: {expected_version}")
    assert actual_version == expected_version

@pytest.mark.parametrize("html_string, expected", [
    ("<html><body><p>Answer with <b>bold</b> text.</p></body></html>", "Answer with <b>bold</b> text."),
    ("<html><body><p>Answer with newline\nand carriage return\r</p></body></html>", "Answer with newline\nand carriage return\r"),
    ("<html><body><div>Other content</div></body></html>", "<html><body><div>Other content</div></body></html>"),
    ("<html><body>Only text, no tags.</body></html>", "<html><body>Only text, no tags.</body></html>") 
])
@pytest.mark.unit
def test_extract_only_answer_from_html(html_string, expected):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Running the extract_only_answer_from_html function")
    result = extract_only_answer_from_html(html_string)

    logger.info(f"Asserting if resulting string: {result} is as expected: {expected}")
    assert result == expected