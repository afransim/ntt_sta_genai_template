import pytest
import logging
from unittest.mock import patch, MagicMock
from mlflow.tracking import MlflowClient

from evaluate_model.utils import get_latest_model_version

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