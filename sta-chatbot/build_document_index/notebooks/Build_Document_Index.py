# Databricks notebook source
##################################################################################
# The purpose of this notebook is to access and prepare our data for use with chatbot. 
#
# This notebook has the following parameters:
#
# * env (required) - Environment the notebook is running in.
# * catalog (required) - Catalog name to read raw files and create tables.
# * use_case (required) - Use case name to instantiate gold, silver and bronze schemas.
# * hierarchy_volume (required) - Name of Volume where files are stored.
# * hierarchy_volume_core_folder (required) - Name of folder, within Volume, that stores the core raw files.
# * hierarchy_volume_temporary_folder (required) - Name of folder, within Volume, that stores the temporary raw files.
# * table_functional_name (required) - Table name to instantiate required tables.
# * vector_search_endpoint_name (required) - Name of vector search endpoint.
# * llm_endpoint_name (required) - Name of model endpoint.
# * embedding_endpoint_name (required) - Name of embedding endpoint.
# * bundle_root (required) - Databricks bundle root path.
# * model_name (required) - Should describe the GenAI task
##################################################################################
# COMMAND ----------
# DBTITLE 1,Import Required Functions
import os
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
from pyspark.sql.functions import substring_index
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from mlflow.utils.databricks_utils import get_databricks_host_creds
from databricks.sdk import WorkspaceClient

dbutils.widgets.dropdown("env", "des", ["des", "pre", "pro"], label="Environment Name")
dbutils.widgets.text("catalog", "alt_des", label="Catalog")              
dbutils.widgets.text("use_case", "chatbot", label="Use Case")
dbutils.widgets.text("hierarchy_volume", "ctti_rag_data", label="Hierarchy Volume")              
dbutils.widgets.text("hierarchy_volume_core_folder", "core_files", label="Hierarchy Volume Core Folder")
dbutils.widgets.text("hierarchy_volume_user_temporary_folder", "user_files/temporary_files", label="Hierarchy Volume User Temporary Folder")
dbutils.widgets.text("table_functional_name", "CTTI_DOCS", label="Table functional name for text, tracking, chunks and index tables")
dbutils.widgets.text("vector_search_endpoint_name", "sta-chatbot", label="Vector Search Endpoint Name")
dbutils.widgets.text("llm_endpoint_name", "chatbot_llm", label="LLM Endpoint Name")
dbutils.widgets.text("embedding_endpoint_name", "chatbot_embeddings", label="Embedding Endpoint Name")
dbutils.widgets.text("des_managed_location", "", label="des_managed_location")
dbutils.widgets.text("pre_managed_location", "", label="pre_managed_location")
dbutils.widgets.text("pro_managed_location", "", label="pro_managed_location")
dbutils.widgets.text("des_external_location", "", label="des_external_location")
dbutils.widgets.text("pre_external_location", "", label="pre_external_location")
dbutils.widgets.text("pro_external_location", "", label="pro_external_location")
dbutils.widgets.text("openai_api_base", "", label="Azure Open AI endpoint")
dbutils.widgets.text("bundle_root", "default", label="bundle_root")
dbutils.widgets.text("finops_tag", "finops_tag", label="Finops Tag")

env = dbutils.widgets.get("env")
assert env in ["des", "pre", "pro"], f"Env name validation failed: {env} should be des, pre or pro."

catalog = dbutils.widgets.get("catalog")
valid_domains = ["admin", "agr", "cct", "clt", "eco", "edu", "ell", "fph", "ind", "inf", "jud", "mam", "mob", "slt", "sem", "sso", "sci", "societat", "tec", "ter", "trb", "nco", "alt"]
assert catalog.split("_")[-1] in ['des', 'pre', 'pro'], f"Catalog name validation failed: {catalog.split('_')[-1]} environment is not valid"
assert catalog.split("_")[0] in valid_domains, f"Catalog name validation failed: {catalog.split('_')[0]} domain is not valid"

assert catalog.split("_")[-1]== env, f"Catalog name ({catalog}) must be correlated with the enviroment in use ({env})."

use_case = dbutils.widgets.get("use_case")
hierarchy_volume = dbutils.widgets.get("hierarchy_volume")
hierarchy_volume_core_folder = dbutils.widgets.get("hierarchy_volume_core_folder")
hierarchy_volume_user_temporary_folder = dbutils.widgets.get("hierarchy_volume_user_temporary_folder")
table_functional_name = dbutils.widgets.get("table_functional_name")
vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
llm_endpoint_name = dbutils.widgets.get("llm_endpoint_name")
embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")
des_managed_location = dbutils.widgets.get("des_managed_location")
pre_managed_location = dbutils.widgets.get("pre_managed_location")
pro_managed_location = dbutils.widgets.get("pro_managed_location")
des_external_location = dbutils.widgets.get("des_external_location")
pre_external_location = dbutils.widgets.get("pre_external_location")
pro_external_location = dbutils.widgets.get("pro_external_location")
openai_api_base = dbutils.widgets.get("openai_api_base")
bundle_root = dbutils.widgets.get("bundle_root")
finops_tag = dbutils.widgets.get("finops_tag")
 
# COMMAND ----------
# DBTITLE 1, Set necessary locations
location_map = {
    'des': (des_managed_location, des_external_location),
    'pre': (pre_managed_location, pre_external_location),
    'pro': (pro_managed_location, pro_external_location)
}

# Set default to 'des' if env is not 'pre' or 'pro'
managed_location, external_location = location_map.get(env, location_map['des'])

# COMMAND ----------
# DBTITLE 1, Import relevant helper functions
import sys
sys.path.append(bundle_root)

from build_document_index.utils import endpoint_exists, create_endpoint, update_endpoint
# COMMAND ----------
# DBTITLE 1, Define schema, tables and volume names
brz_schema = "brz_"+use_case
slv_schema = "slv_"+use_case
gld_schema = "gld_"+use_case
table_text = "ftr_"+table_functional_name+"_txt"
table_track = "aux_"+table_functional_name+"_track"
table_chunks = "ftr_"+table_functional_name+"_chunks"
table_idx = "ftr_"+table_functional_name+"_idx"
flat_volume = hierarchy_volume+"_flat"


openai_api_key_reference = "{{secrets/pgc-dbs-sco-chb-01/wsp-sco-tok-oai-01}}"
openai_api_base_reference = openai_api_base


# COMMAND ----------
DB_HIERARCHY_VOLUME_CORE_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{hierarchy_volume}/{hierarchy_volume_core_folder}"
DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{hierarchy_volume}/{hierarchy_volume_user_temporary_folder}"
DB_FLAT_VOLUME_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{flat_volume}"
TMP_FILE_PATH = "/tmp/tmp_file"

# COMMAND ----------
# DBTITLE 1,List files in Volume
from build_document_index.utils import list_files, flatten_directory_copy

volume_path =f"/Volumes/{catalog}/{brz_schema}/{hierarchy_volume}/{hierarchy_volume_core_folder}"
file_paths = list_files(dbutils, volume_path)

# Extract file names from paths
df = spark.createDataFrame(file_paths, "string").select(substring_index("value", "/", -1).alias("file_name"))

# Show dataframe
df.show()

# COMMAND ----------
# DBTITLE 1, Check for files not yet processed
# Get the list of already processed allowed type of files from the table
processed_files = spark.sql(f"SELECT DISTINCT file_name FROM  `{catalog}`.`{brz_schema}`.`{table_track}`").collect()
processed_files = set(row["file_name"] for row in processed_files)

# Find all files in the folders and subfolders
all_files = []
for hierarchy_volume_core_folder, dirs, files in os.walk(volume_path):
    for file in files:
        all_files.append(os.path.join(hierarchy_volume_core_folder, file))

# COMMAND ----------
# DBTITLE 1,Extract and save text from pdf
# Function to read allowed type of files and extract text
from build_document_index.utils import extract_text

table_full_name = f"{catalog}.{brz_schema}.{table_text}"

# Check if the table exists or is empty
if not spark.catalog.tableExists(table_full_name) or spark.table(table_full_name).isEmpty():
    doc_data = []
    
    for file in all_files:
        file_path = os.path.join(hierarchy_volume_core_folder, file)
        file_name = os.path.basename(file_path)
        folder_path = 'dbfs:' + os.path.dirname(file_path)
        try:
            text = extract_text(file_path)
            doc_data.append({"file_name": file_name, "extracted_text": text, "folder_name": folder_path, "document_type": "core"})
        except Exception as e:
            print(f"File {file_name} could not be processed. Error: {e}")

    if doc_data:
        # Saving extracted text to table_text 
        try:
            files_df = spark.createDataFrame(doc_data)
            files_df.write.mode('overwrite').saveAsTable(table_full_name)
        except Exception as e:
            print(f"There was an error saving the extracted text. Error: {e}")

flatten_directory_copy(DB_HIERARCHY_VOLUME_CORE_DESTINATION, DB_FLAT_VOLUME_DESTINATION)

# Display the table
display(spark.table(table_full_name).limit(5))

# COMMAND ----------
# DBTITLE 1,Extract and save chunks from text
# Define the table name for chunked data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from build_document_index.utils import get_chunks_with_metadata

table_chunks_full_name = f"{catalog}.{slv_schema}.{table_chunks}"

# Check if the table_chunks exists or is empty
if not spark.catalog.tableExists(table_chunks_full_name) or spark.table(table_chunks_full_name).isEmpty():

    # Parameters for the splitter:
    chunk_size = 1000  # Example chunk size
    chunk_overlap = 200  # Example chunk overlap

    # Initialize the splitter
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Define the UDF for splitting text into chunks with additional metadata
    def get_chunks_with_metadata(text):
        chunks = recursive_splitter.split_text(text)
        return [(chunk, len(chunk), i, len(chunks)) for i, chunk in enumerate(chunks)]

    # Register the UDF with Spark
    split_udf = F.udf(get_chunks_with_metadata, ArrayType(StructType([
        StructField("chunk", StringType(), False),
        StructField("chunk_length_in_chars", IntegerType(), False),
        StructField("chunk_location_in_doc", IntegerType(), False),
        StructField("total_chunks_in_doc", IntegerType(), False)
    ])))

    # Read the data from the first table
    df = spark.table(table_full_name)

    # Apply the text splitting UDF to create text chunks with metadata
    chunked_df = df.withColumn("chunks_metadata", F.explode(split_udf(F.col("extracted_text"))))

    # Select and flatten the desired columns
    final_chunked_df = chunked_df.select(
        F.col("folder_name"),
        F.col("file_name"),
        F.col("document_type"),
        F.col("chunks_metadata.chunk").alias("text_chunk"),
        F.col("chunks_metadata.chunk_length_in_chars"),
        F.col("chunks_metadata.chunk_location_in_doc"),
        F.col("chunks_metadata.total_chunks_in_doc")
    )

    # Save the chunked text DataFrame to a new table
    final_chunked_df.write.mode('overwrite').saveAsTable(table_chunks_full_name)

# Display the chunked text table
display(spark.table(table_chunks_full_name).limit(2))

# COMMAND ----------
# DBTITLE 1,Update docs_track table so same files aren't processed again
if 'files_df' in locals():
    files_df.createOrReplaceTempView("temp_table")  # Create a temporary table from the DataFrame

    # Insert only the rows that do not exist in the target table
    spark.sql(f"""
        INSERT INTO `{catalog}`.`{brz_schema}`.`{table_track}`
        SELECT file_name, folder_name, document_type, 'READY' AS status 
        FROM temp_table
        WHERE NOT EXISTS (
            SELECT 1 FROM `{catalog}`.`{brz_schema}`.`{table_track}`
            WHERE temp_table.file_name = `{catalog}`.`{brz_schema}`.`{table_track}`.file_name AND
                temp_table.folder_name = `{catalog}`.`{brz_schema}`.`{table_track}`.folder_name AND
                temp_table.document_type = `{catalog}`.`{brz_schema}`.`{table_track}`.document_type
        )
    """)
else:
    print("No files were found to add to index.")

# COMMAND ----------
# DBTITLE 1,Create embedding model endpoint
from databricks.sdk.service.serving import ServedEntityInput, ExternalModel, OpenAiConfig, ExternalModelProvider, EndpointCoreConfigInput, EndpointTag

embedding_openAiConfig = OpenAiConfig(
    openai_api_type="azure",
    openai_api_base=f"{openai_api_base_reference}",
    openai_api_key=f"{openai_api_key_reference}",
    openai_deployment_name="text-embedding-3-large",
    openai_api_version="2024-02-01",
)

embedding_external_model_config=ExternalModel(
    name="text-embedding-3-large",
    provider=ExternalModelProvider("openai"),
    task="llm/v1/embeddings",
    openai_config=embedding_openAiConfig
)
embedding_served_entities = [ServedEntityInput(external_model=embedding_external_model_config)]


embedding_config = EndpointCoreConfigInput(
    served_entities=embedding_served_entities
)

embedding_tags = [EndpointTag(key="finops-projecte", value=finops_tag)]

# Check if the desired endpoint exists
if endpoint_exists(embedding_endpoint_name):
    print(f"Endpoint '{embedding_endpoint_name}' already exists.")
    print("Updating embedding endpoint...")

    update_endpoint(embedding_endpoint_name, embedding_served_entities)
else:    
    create_endpoint(embedding_endpoint_name, embedding_config, embedding_tags)

# COMMAND ----------
# DBTITLE 1,Create Vector Search Endpoint
from build_document_index.utils import create_or_select_vs_endpoint

w = WorkspaceClient()
vector_search_endpoint = vector_search_endpoint_name

create_or_select_vs_endpoint(w, vector_search_endpoint)

# COMMAND ----------
# DBTITLE 1, Populate Vector Search Index
import sys
from build_document_index.utils import update_vs_endpoint

vsc = VectorSearchClient()
update_vs_endpoint(vsc, vector_search_endpoint_name, embedding_endpoint_name, catalog, gld_schema, slv_schema, table_idx, table_chunks)

# COMMAND ----------
# DBTITLE 1,Create model gptt4 endpoint

llm_openAiConfig = OpenAiConfig(
    openai_api_type="azure",
    openai_api_base=f"{openai_api_base_reference}",
    openai_api_key=f"{openai_api_key_reference}",
    openai_deployment_name="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

llm_external_model_config=ExternalModel(
    name="gpt-4o",
    provider=ExternalModelProvider("openai"),
    task="llm/v1/chat",
    openai_config=llm_openAiConfig
)

llm_served_entities = [ServedEntityInput(external_model=llm_external_model_config)]

llm_config = EndpointCoreConfigInput(
    served_entities=llm_served_entities
)

from databricks.sdk.service.serving import EndpointTag
llm_tags = [EndpointTag(key="finops-projecte", value=finops_tag)]


# Check if the desired endpoint exists
if endpoint_exists(llm_endpoint_name):
    print(f"Endpoint '{llm_endpoint_name}' already exists.")
    print("Updating LLM endpoint...")

    update_endpoint(llm_endpoint_name, llm_served_entities)
else:    
    create_endpoint(llm_endpoint_name, llm_config, llm_tags)

# COMMAND ----------
# DBTITLE 1, Persisting variables needed for subsequent tasks
dbutils.jobs.taskValues.set("catalog", catalog)
dbutils.jobs.taskValues.set("use_case", use_case)
dbutils.jobs.taskValues.set("brz_schema", brz_schema)
dbutils.jobs.taskValues.set("slv_schema", slv_schema)
dbutils.jobs.taskValues.set("gld_schema", gld_schema)
dbutils.jobs.taskValues.set("hierarchy_volume", hierarchy_volume)
dbutils.jobs.taskValues.set("hierarchy_volume_core_folder", hierarchy_volume_core_folder)
dbutils.jobs.taskValues.set("flat_volume", flat_volume)
dbutils.jobs.taskValues.set("table_text", table_text)
dbutils.jobs.taskValues.set("table_track", table_track)
dbutils.jobs.taskValues.set("table_chunks", table_chunks)
dbutils.jobs.taskValues.set("table_idx", table_idx)
dbutils.jobs.taskValues.set("vector_search_endpoint_name", vector_search_endpoint_name)
dbutils.jobs.taskValues.set("llm_endpoint_name", llm_endpoint_name)
dbutils.jobs.taskValues.set("embedding_endpoint_name", embedding_endpoint_name)
dbutils.jobs.taskValues.set("bundle_root", bundle_root)
dbutils.jobs.taskValues.set("finops_tag", finops_tag)