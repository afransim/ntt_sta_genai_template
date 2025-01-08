import mlflow
from databricks.sdk import WorkspaceClient

def get_latest_model_version(model_name):
    
    """
    Get the latest model version number within a certain catalog and schema

    Args:
        catalog (str): Name of model's catalog
        schema (str): Name of model's schema
        model_name (int): Name of model in catalog.schema.model_name

    Returns:
        int: Value of the latest version of the specified model
    """

    mlflow.set_registry_uri("databricks-uc")
    mlflow_client = mlflow.MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def endpoint_exists(serving_name):
  """
    Check if an endpoint with the serving_endpoint_name exists

    Args:
        serving_name (str): Serving endpoint's name

    Returns:
        bool: True if serving endpoints exists. False otherwise. 

  """
  w = WorkspaceClient()
  try:
    w.serving_endpoints.get(name=serving_name)
    return True
  except Exception as e:
      return False

def create_endpoint(serving_name, config, tags):
  """
    Create serving endpoint and wait for it to be ready

    Args:
        serving_name (str): Serving endpoint's name
        config (EndpointCoreConfigInput): Class containing required info for creating serving endpoint (Databricks SDK)
        tags(Array<EndpointTag>): List of EndpointTag classes
    
    Raises:
        requests.exceptions.HTPPError: Raises any HTTP error that might have occurred while creating the endpoint

  """
  w = WorkspaceClient()

  print(f"Creating new serving endpoint: {serving_name}")

  try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=serving_name,
        config=config,
        tags=tags
    )
    print(f"Serving endpoint '{serving_name}' created successfully.")
    print(f"Details: {endpoint}")
  except Exception as e:
    print(f"Error creating serving endpoint: {e}")
    raise
  
def update_endpoint(serving_name, served_entities):

  """
    Update endpoint

    Args:
        serving_name (str): Serving endpoint's name
        served_entities (Array<ServedEntityInput>): Array containing served entities classes
      
    Raises:
        HTTPError: Raises any HTTP error while calling the endpoint

  """

  w=WorkspaceClient()

  print(f"Updating config for serving endpoint: {serving_name}")
  w.serving_endpoints.update_config_and_wait(
        name=serving_name,
        served_entities=served_entities
    )
  print(f"Serving endpoint {serving_name} updated.")
