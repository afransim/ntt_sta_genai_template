import json
from typing import List
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import col

def list_files(dbutils, directory: str):
    """
    Recursively list all files in a specified directory using Databricks utilities.

    Args:
        dbutils (object): Databricks utilities object used to interact with the file system.
        directory (str): Path to the directory where files should be listed.

    Returns:
        List[str]: A list of file paths within the directory, including those in subdirectories.
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


def get_deleted_files_id_from_table(
    catalog: str,
    database: str,
    table: str,
    filenames_to_delete: List[str],
    spark
) -> List[str]:
    """
    Retrieve the IDs of files marked for deletion from a specified table in a Spark catalog.

    Args:
        catalog (str): Name of the catalog containing the table.
        database (str): Name of the database where the table is located.
        table (str): Name of the table containing file information.
        filenames_to_delete (List[str]): List of filenames to search for in the table.
        spark (object): Spark session used to query the table.

    Returns:
        List[str]: List of file IDs that correspond to the files to be deleted.
    """
    table_path = f"{catalog}.{database}.{table}"
    df = spark.table(table_path)
    df = df.filter(df.file_name.isin(filenames_to_delete))
    df = df.select("id")
    result = [row.id for row in df.collect()]
    return result



def remove_data_from_table(
    catalog: str,
    database: str,
    table: str,
    filenames_to_delete: List[str],
    spark
):
    """
    Remove specified files from a table in a Spark catalog and overwrite the table with the remaining files.

    Args:
        catalog (str): Name of the catalog containing the table.
        database (str): Name of the database where the table is located.
        table (str): Name of the table containing file information.
        filenames_to_delete (List[str]): List of filenames to delete from the table.
        spark (object): Spark session used to query and update the table.

    Returns:
        None
    """
    table_full_name = f"{catalog}.{database}.{table}"
    df = spark.table(table_full_name)

    remaining_files_df = df.filter(~df.file_name.isin(filenames_to_delete))
    remaining_files_df.write.mode("overwrite").saveAsTable(table_full_name)



def remove_data_from_index(
    vector_search_endpoint: str,
    catalog: str,
    database: str,
    table: str,
    ids_to_delete: List[str]
):
    """
    Remove specified entries from a vector search index by file IDs.

    Args:
        vector_search_endpoint (str): Endpoint of the vector search service.
        catalog (str): Name of the catalog containing the table.
        database (str): Name of the database where the table is located.
        table (str): Name of the table containing file information.
        ids_to_delete (List[str]): List of file IDs to delete from the index.

    Returns:
        None
    """
    if ids_to_delete != []:
        vsc = VectorSearchClient()
        index_name = f"{catalog}.{database}.{table}"
        index = vsc.get_index(endpoint_name=vector_search_endpoint, index_name=index_name)
        index.delete(ids_to_delete)


def delete_index_table(
    endpoint_name: str,
    catalog: str,
    database: str,
    table: str,
):
    """
    Delete the entire vector search index associated with a specified table.

    Args:
        endpoint_name (str): Name of the vector search endpoint.
        catalog (str): Name of the catalog containing the table.
        database (str): Name of the database where the table is located.
        table (str): Name of the table associated with the index to be deleted.

    Returns:
        None
    """
    vsc = VectorSearchClient()
    index_name = f"{catalog}.{database}.{table}"
    vsc.delete_index(endpoint_name=endpoint_name, index_name=index_name)
