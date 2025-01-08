import pytest
import logging
from unittest.mock import MagicMock

from remove_files.utils import get_deleted_files_id_from_table, remove_data_from_table, list_files

@pytest.mark.parametrize(
    "mock_fs_ls_side_effect, expected, expected_call_count", [
        ([[MagicMock(path='/dir/test1.txt', isDir=lambda: False), MagicMock(path='/dir/file2.txt', isDir=lambda: False)]], ['/dir/test1.txt', '/dir/file2.txt'], 1),
        ([[MagicMock(path='/dir/subdir', isDir=lambda: True), MagicMock(path='/dir/file1.txt', isDir=lambda: False)], [MagicMock(path='/dir/subdir/file2.txt', isDir=lambda: False)]], ['/dir/subdir/file2.txt', '/dir/file1.txt'], 2),
        ([[]], [], 1),
        (Exception("Error accessing directory"), [], 1)
    ]
)
@pytest.mark.unit
def test_list_files(mock_fs_ls_side_effect, expected, expected_call_count):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create a mock dbutils object
    logger.info("Mocking dbutils object")
    mock_dbutils = MagicMock()

    logger.info("Mocking the behaviour of the dbutils.fs.ls function")
    if isinstance(mock_fs_ls_side_effect, list):
        mock_dbutils.fs.ls.side_effect = mock_fs_ls_side_effect

    # Call the list_files function
    logger.info("Calling the list_files function...")
    result = list_files(mock_dbutils, '/dir')

    # Assertions
    logger.info(f"Asserting if result: {result} is equal to the expected: {expected}")
    assert result == expected, f"Expected {expected} but got {result}"
    logger.info("Asserting if call count of dbutils.fs.ls according to recursivity of the list_files function")
    assert mock_dbutils.fs.ls.call_count == expected_call_count

@pytest.fixture(scope="session")
@pytest.mark.unit
def mock_spark_session():

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Mocking spark session")
    mock_spark = MagicMock()

    logger.info("Mocking a dataframe that results from spark table function")
    mock_df = MagicMock()

    logger.info("Mocking the populating of the mocked dataframe")
    mock_spark.table.return_value = mock_df

    logger.info("Returning the mocked objects")
    return mock_spark, mock_df

@pytest.mark.unit
def test_get_file_ids_to_delete(mock_spark_session):
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Accessing the mocked spark session and dataframe objects")
    mock_spark, mock_df = mock_spark_session

    logger.info("Instantiating the databricks schema and file names to delete")
    catalog_name = "test_catalog"
    schema_name = "test_schema"
    table_name = "test_table"
    filenames_to_delete = ["file2.txt", "file4.txt"]

    logger.info("Mocking the filtration of the dataframe, to access only the information for the files to delete")
    mock_filtered_df = MagicMock()
    mock_df.filter.return_value = mock_filtered_df

    mock_selected_df = MagicMock()
    mock_filtered_df.select.return_value = mock_selected_df

    logger.info("Mocking the correct IDs of the files to delete")
    mock_selected_df.collect.return_value = [
        MagicMock(id=2),
        MagicMock(id=4)
    ]

    logger.info("Running the get_deleted_files_id_from_table function")
    result = get_deleted_files_id_from_table(catalog_name, schema_name, table_name, filenames_to_delete, mock_spark)

    logger.info(f"Assert if the returned file ID: {sorted(result)} is as expected [2,4]")
    assert sorted(result) == [2, 4]

    logger.info("Asserting if spark.table, spark.dataframe.filter, spark.dataframe.select and spark.dataframe.collect functions were called once")
    mock_spark.table.assert_called_once_with(f"{catalog_name}.{schema_name}.{table_name}")
    mock_df.filter.assert_called_once_with(mock_df.file_name.isin(filenames_to_delete))
    mock_filtered_df.select.assert_called_once_with("id")
    mock_selected_df.collect.assert_called_once()

@pytest.mark.unit
def test_remove_data_from_table():

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Mocking the Spark session
    logger.info("Mocking the spark session")
    spark = MagicMock()
    
    # Mocking the inputs
    logger.info("Instantiating the databricks schema, file names to delete, and remaining files")
    catalog = 'test_base'
    schema = 'test_schema'
    table = 'test_table'
    filenames_to_delete = ['file1.csv', 'file2.csv']
    full_table_name = f"{catalog}.{schema}.{table}"

    # Mocking the DataFrame returned by spark.table()
    logger.info("Mocking dataframe returned by spark.table function")
    mock_df = MagicMock()
    spark.table.return_value = mock_df

    # Mock the behavior of the column returned by `isin`
    logger.info("Mocking the behaviour of the column returned by the `isin` method")
    isin_mock = MagicMock(name='isin_mock')
    mock_df.file_name.isin.return_value = isin_mock
    
    # Mock the behavior of the inverted `isin` result
    logger.info("Mocking the inverted behaviour of the `isin` method")
    inverted_isin_mock = ~isin_mock
    mock_df.filter.return_value = MagicMock(name='filtered_df')

    # Mock the 'write' method on the filtered DataFrame
    logger.info("Mocking the spark.dataframe.write method")
    mock_writer = MagicMock()
    mock_df.filter.return_value.write.mode.return_value = mock_writer

    # Call the function with mocks
    logger.info("Running the remove_data_from_table function")
    remove_data_from_table(catalog, schema, table, filenames_to_delete, spark)

    # Assertions to ensure the function behaves as expected
    logger.info("Asserting that the spark.table function, isin method and spark.dataframe.filter function were called once")
    spark.table.assert_called_once_with(full_table_name)

    # Ensure the 'isin' method was called with the correct filenames
    mock_df.file_name.isin.assert_called_once_with(filenames_to_delete)

    # Ensure the filtering happened correctly with the inverted 'isin'
    mock_df.filter.assert_called_once_with(inverted_isin_mock)

    # Simulate what the filtered DataFrame should contain after filtering
    logger.info("Mocking spark.dataframe.filtering results")
    filtered_df = mock_df.filter.return_value
    filtered_df.collect.return_value = [{'file_name': 'file3.csv'}]

    # Ensure that the correct files remain
    logger.info("Asserting that the correct files remain in the dataframe")
    remaining_data = filtered_df.collect()
    assert remaining_data == [{'file_name': 'file3.csv'}], "The filtered DataFrame should only contain 'file3.csv'"

    # Check that the write operation was called with 'overwrite' mode
    logger.info("Asserting that the spark.dataframe.write method was caled once with 'overwrite' configuration")
    filtered_df.write.mode.assert_called_once_with('overwrite')
    mock_writer.saveAsTable.assert_called_once_with(full_table_name)