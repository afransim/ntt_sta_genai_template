from databricks.sdk.runtime import spark

def create_volume(catalog: str,schema: str,volume: str, managed_location:str, external_location: str):
    '''
    Checks if volume under catalog and schema exists, if not, it is created
    
    Args:
        catalog (str): Azure Databricks Unity Catalog catalog 
        schema (str): Azure Databricks Unity Catalog schema 
        volume (str): Azure Databricks Unity Catalog volume 
        managed_location (str): Azure storage managed location
        external_location (str): Azure storage external location
    '''
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} MANAGED LOCATION '{managed_location}/catalogs/{catalog}'")
    spark.sql(f"CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog}.{schema}.{volume} LOCATION '{external_location}/catalogs/{catalog}/{schema}/volumes/{volume}'")