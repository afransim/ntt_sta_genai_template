# Databricks notebook source
##################################################################################
# Deploy Application Notebook
#
# In this notebook, we will deploy the custom model registered with MLflow in the prior notebook and deploy it to Databricks model serving (AWS|Azure). Databricks model serving provides containerized deployment options for registered models thought which authenticated applications can interact with the model via a REST API. This provides MLOps teams an easy way to deploy, manage and integrate their models with various applications.
#
# Parameters:
# * bundle_root (required):            - Databricks bundle root path.
# * catalog (automatic):               - Catalog where the model is stored and where to store the inference table.
# * gld_schema (automatic)             - Schema where the model is stored and where to store the inference table.
# * model_name (automatic):            - Name of the model.
# * use_case (automatic):              - Name of the use case.
##################################################################################
# In this notebook, we will deploy the custom model registered with MLflow in the prior notebook and deploy it to Databricks model serving [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/).  Databricks model serving provides containerized deployment options for registered models thought which authenticated applications can interact with the model via a REST API. 
# COMMAND ----------
# DBTITLE 1,Imports and Retrieve parameters from previous tasks

root = dbutils.widgets.get("bundle_root")
catalog=dbutils.jobs.taskValues.get(taskKey = "AssembleApplication", key = "catalog", default="alt_des")
use_case=dbutils.jobs.taskValues.get(taskKey = "AssembleApplication", key = "use_case", default="chatbot")
model_name=dbutils.jobs.taskValues.get(taskKey = "AssembleApplication", key = "model_name", default="chatbot_model")
gld_schema=dbutils.jobs.taskValues.get(taskKey = "AssembleApplication", key = "gld_schema")
bundle_root = dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "bundle_root", default="default")
finops_tag= dbutils.jobs.taskValues.get(taskKey="BuildDocumentIndex", key="finops_tag", default="finops_tag")
serving_name = use_case + '_app'

# COMMAND ----------
# DBTITLE 1, Import of relevant helper functions
import sys

sys.path.append(bundle_root)

from deploy_application.utils import get_latest_model_version, endpoint_exists, create_endpoint, update_endpoint

# COMMAND ----------
# DBTITLE 1, Retrieve the latest model version for deployment
from mlflow.tracking import MlflowClient

MODEL_NAME_FQN = f"{catalog}.{gld_schema}.{model_name}"
model_version = get_latest_model_version(MODEL_NAME_FQN)
print("model name: ",MODEL_NAME_FQN, "\nmodel version: ", model_version) 

client = MlflowClient(registry_uri="databricks-uc")
model = client.get_model_version(MODEL_NAME_FQN, model_version)
alias = "champion"
if alias not in model.aliases:
    client.set_registered_model_alias(
        name = MODEL_NAME_FQN,
        alias = alias,
        version = model_version,
    )

# COMMAND ----------
# DBTITLE 1, Deploy Model Serving Endpoint
# Models may typically be deployed to model serving endpoints using either the Databricks workspace user-interface or a REST API.  Because our model depends on the deployment of a sensitive environment variable, we will need to leverage a relatively new model serving feature that is currently only available via the REST API.
from databricks.sdk.service.serving import AutoCaptureConfigInput    
from databricks.sdk.service.serving import ServedEntityInput
from databricks.sdk.service.serving import EndpointCoreConfigInput
from databricks.sdk.service.serving import EndpointTag
from mlflow.utils.databricks_utils import get_databricks_host_creds

serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()



served_entities = [
    ServedEntityInput(
        name="current",
        entity_name=MODEL_NAME_FQN,
        entity_version=model_version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "DATABRICKS_TOKEN": creds.token,
            'DATABRICKS_HOST': f"https://{serving_host}",
            'MLFLOW_TRACKING_URI': 'databricks'
        }
    )
]


auto_capture_config = AutoCaptureConfigInput(
    enabled=True,
    catalog_name=catalog,
    schema_name=gld_schema,
    table_name_prefix=serving_name
)

config = EndpointCoreConfigInput(
    served_entities=served_entities,
    auto_capture_config=auto_capture_config
)

tags = [EndpointTag(key="finops-projecte", value=finops_tag)]

# COMMAND ----------
# DBTITLE 1,Use the defined function to create or update the endpoint
# Kick off endpoint creation/update
if not endpoint_exists(serving_name):
  create_endpoint(serving_name, config, tags)
else:
  update_endpoint(serving_name, served_entities)
