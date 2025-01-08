import pytest
import logging
from unittest.mock import MagicMock, patch 

import build_document_index.utils

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
    result = build_document_index.utils.list_files(mock_dbutils, '/dir')

    # Assertions
    logger.info(f"Asserting if result: {result} is equal to the expected: {expected}")
    assert result == expected, f"Expected {expected} but got {result}"
    logger.info("Asserting if call count of dbutils.fs.ls according to recursivity of the list_files function")
    assert mock_dbutils.fs.ls.call_count == expected_call_count

@patch('build_document_index.utils.TextLoader')
@pytest.mark.unit
def test_extract_text_with_text_loader(mock_text_loader):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create a mock instance of TextLoader
    logger.info("Mocking TextLoader object")
    mock_loader_instance = MagicMock()
    mock_text_loader.return_value = mock_loader_instance

    # Mock the load method to return a list with a mock object
    logger.info("Mocking the contents read by the TextLoader object")
    mock_loader_instance.load.return_value = [MagicMock(page_content="Sample text content")]

    logger.info("Running extract_text_with_text_loader function")
    result = build_document_index.utils.extract_text_with_text_loader("fake_file_path.txt")

    logger.info(f"Asserting if result content: {result} is as expected: Sample text content")
    assert result == "Sample text content"

    logger.info("Asserting the number of calls for TextLoader object and load function")
    mock_text_loader.assert_called_once_with("fake_file_path.txt")
    mock_loader_instance.load.assert_called_once()

@patch('build_document_index.utils.Docx2txtLoader')
@pytest.mark.unit
def test_extract_text_with_microsoft_word_loader(mock_docx2txt_loader):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create a mock instance of Docx2txtLoader
    logger.info("Mocking Docx2txtLoader object")
    mock_loader_instance = MagicMock()
    mock_docx2txt_loader.return_value = mock_loader_instance

    # Mock the load method to return a list with a mock object
    logger.info("Mock the load method from the Docx2txtLoader object")
    mock_loader_instance.load.return_value = [MagicMock(page_content="Sample text content")]

    # Call the function under test
    logger.info("Running extract_text_with_microsoft_word_loader function")
    result = build_document_index.utils.extract_text_with_microsoft_word_loader("fake_file_path.docx")
    
    # Assertions to verify the behavior
    logger.info(f"Asserting if the result: {result} is as expected: Sample Text Content")
    assert result == "Sample text content"

    logger.info("Asserting the number of calls of the Docx2txtLoader object and respective load function")
    mock_docx2txt_loader.assert_called_once_with("fake_file_path.docx")
    mock_loader_instance.load.assert_called_once()

@patch('build_document_index.utils.PdfReader')
@pytest.mark.unit
def test_extract_text_with_pdf_reader(mock_pdf_reader):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create a mock instance of PdfReader
    logger.info("Mocking the PdfReader object")
    mock_reader_instance = MagicMock()
    mock_pdf_reader.return_value = mock_reader_instance

    # Create mock pages with the extract_text method
    logger.info("Mocking the page content returned by the extract_text method")
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 text. "

    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 text. "

    mock_reader_instance.pages = [mock_page1, mock_page2]

    # Call the function under test
    logger.info("Running the extract_text function")
    result = build_document_index.utils.extract_text("fake_file_path.pdf")
    
    # Assertions to verify the behavior
    logger.info(f"Asserting if the result: {result} is as expected: Page 1 text. Page 2 text. ") 
    assert result == "Page 1 text. Page 2 text. "
    mock_pdf_reader.assert_called_once_with("fake_file_path.pdf")

    logger.info("Asserting the number of calls for the extract_text function")
    assert mock_page1.extract_text.called
    assert mock_page2.extract_text.called

@patch('build_document_index.utils.PdfReader')
@patch('build_document_index.utils.TextLoader')
@patch('build_document_index.utils.Docx2txtLoader')
@pytest.mark.unit
def test_extract_text(mock_docx_loader, mock_text_loader, mock_pdf_reader):

    # Create a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create mock instances for each loader
    logger.info("Mocking each of the text loader instances")
    mock_pdf_reader_instance = MagicMock()
    mock_text_loader_instance = MagicMock()
    mock_docx_loader_instance = MagicMock()
    
    # Configure mocks
    logger.info("Mocking the text loader functions' results")
    mock_pdf_reader.return_value = mock_pdf_reader_instance
    mock_text_loader.return_value = mock_text_loader_instance
    mock_docx_loader.return_value = mock_docx_loader_instance

    # Mocking the behavior of each loader function
    logger.info("Mocking the behaviour of each function")
    mock_pdf_reader_instance.pages = [MagicMock(extract_text=MagicMock(return_value="PDF content"))]
    mock_text_loader_instance.load.return_value = [MagicMock(page_content="Text content")]
    mock_docx_loader_instance.load.return_value = [MagicMock(page_content="Docx content")]

    # Test cases
    logger.info("Asserting the results of the text loader functions")
    assert build_document_index.utils.extract_text("file.pdf") == "PDF content"
    assert build_document_index.utils.extract_text("file.md") == "Text content"
    assert build_document_index.utils.extract_text("file.docx") == "Docx content"
    assert build_document_index.utils.extract_text("file.txt") == "Text content"
    assert build_document_index.utils.extract_text("file.py") is None

    # Assertions to verify the mocks were called correctly
    logger.info("Asserting the number of calls of each text loader function for their respective file types")
    mock_pdf_reader.assert_called_once_with("file.pdf")
    mock_text_loader.assert_any_call("file.md")
    mock_docx_loader.assert_called_once_with("file.docx")