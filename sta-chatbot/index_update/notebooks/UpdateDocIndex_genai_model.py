# Databricks notebook source
##################################################################################
# The purpose of this notebook is to access and update document index with temporary file information. 
#
# This notebook has the following parameters:
#
# * env (required)                         - Environment the notebook is running in.
# * catalog (required)                     - Catalog name to read raw files and create tables.
# * use_case (required)                    - Use case name to instantiate gold, silver and bronze schemas.
# * hierarchy_volume (required)            - Name of Volume where files are stored.
# * table_functional_name (required)       - Table name to instantiate required tables.
# * vector_search_endpoint_name (required) - Name of vector search endpoint.
# * llm_endpoint_name (required)           - Name of model endpoint.
# * embedding_endpoint_name (required)     - Name of embedding endpoint.
# * bundle_root (required):                - Databricks bundle root path.
# * document_type (required)               - Type of document to be updated into index (tmp/core/pmt)
# COMMAND ----------
# DBTITLE 1, Install libraries

%pip install langchain-text-splitters==0.2.2 databricks-vectorsearch==0.39 langchain-community==0.2.10 pypdf==4.2.0 mlflow #NOSONAR

dbutils.library.restartPython()

# COMMAND ----------
# DBTITLE 1, Setting up start time
from datetime import datetime
start_time = datetime.now()

# COMMAND ----------
# DBTITLE 1, Library and helper functions import
import os
from databricks.vector_search.client import VectorSearchClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyspark.sql import functions as F
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
import sys 

root = dbutils.widgets.get("bundle_root")
sys.path.append(root)

import index_update.utils

# COMMAND ----------
# DBTITLE 1, Setting up widgets to retrieve notebook arguments
dbutils.widgets.dropdown("env", "des", ["des", "pre", "pro"], label="Environment Name")

dbutils.widgets.text("catalog", "alt_des", label="Catalog")
dbutils.widgets.text("use_case", "chatbot", label="Use Case")               
dbutils.widgets.text("table_functional_name", "CTTI_DOCS", label="Table functional name for text, tracking, chunks and index tables")

dbutils.widgets.text("vector_search_endpoint_name", "sta-chatbot", label="Vector Search Endpoint Name")
dbutils.widgets.text("embedding_endpoint_name", "chatbot_embeddings", label="Embedding Endpoint Name")

dbutils.widgets.dropdown("document_type", "tmp", ["tmp", "core"], label="Document Type")
dbutils.widgets.text("hierarchy_volume", "ctti_rag_data", label="Name of the root storage folder")

dbutils.widgets.text("model_name", "chatbot_model", label="LLM name")

env = dbutils.widgets.get("env")
assert env in ["des", "pre", "pro"], f"Env name validation failed: {env} should be des, pre or pro."

catalog = dbutils.widgets.get("catalog")
valid_domains = ["admin", "agr", "cct", "clt", "eco", "edu", "ell", "fph", "ind", "inf", "jud", "mam", "mob", "slt", "sem", "sso", "sci", "societat", "tec", "ter", "trb", "nco", "alt"]
assert catalog.split("_")[-1] in ['des', 'pre', 'pro'], f"Catalog name validation failed: {catalog.split('_')[-1]} environment is not valid"
assert catalog.split("_")[0] in valid_domains, f"Catalog name validation failed: {catalog.split('_')[0]} domain is not valid"

use_case = dbutils.widgets.get("use_case")
schema_brz = "brz_"+use_case
schema_slv = "slv_"+use_case
schema_gld = "gld_"+use_case

table_functional_name = dbutils.widgets.get("table_functional_name")

vector_search_endpoint_name=dbutils.widgets.get("vector_search_endpoint_name")
embedding_endpoint_name=dbutils.widgets.get("embedding_endpoint_name")

model_name = dbutils.widgets.get("model_name")
assert model_name.split("_")[-1]=="model", f"Model name validation failed: {model_name.split('_')[-1]}. Should end with _model."

document_type = dbutils.widgets.get("document_type")

hierarchy_volume = dbutils.widgets.get("hierarchy_volume")

# COMMAND ----------

# DBTITLE 1, Define table and volume names
table_text = "ftr_"+table_functional_name+"_txt"
table_track = "aux_"+table_functional_name+"_track"
table_chunks = "ftr_"+table_functional_name+"_chunks"
table_idx = "ftr_"+table_functional_name+"_idx"

flat_volume = hierarchy_volume+"_flat"

# COMMAND ----------

# DBTITLE 1, Detecting volume used according to document type
try:
    if document_type == "core":
        volume = f"{hierarchy_volume}/core_files"
    else: # "tmp" or other
        volume = f"{hierarchy_volume}/user_files/temporary_files"
except Exception as e:
        volume = f"{hierarchy_volume}/core_files"
        document_type="core"
print(f"Used volume: {volume}")

# COMMAND ----------

# DBTITLE 1, Get already uploaded files
df_table = spark.table(f"{catalog}.{schema_brz}.{table_track}")
df_table = df_table.withColumn("path", concat(col("folder_name"), lit("/"), col("file_name")))
df_table = df_table.filter((df_table.status == "UPDATING") | (df_table.status == "UPDATED") | (df_table.status == 'READY')).select("path")
df_table.collect()

# COMMAND ----------

# DBTITLE 1, Listing all files in volume
rdd_schema = StructType([StructField("path", StringType(), True)])

doc_volume_path = f"/Volumes/{catalog}/{schema_brz}/{volume}/"
file_paths = [(t,) for t in index_update.utils.list_files(dbutils, doc_volume_path)]
df_volume = spark.createDataFrame(file_paths, schema=rdd_schema)
df_volume.collect()

# COMMAND ----------

# DBTITLE 1, Listing files that need to be added to index
df = df_volume.subtract(df_table)

filepaths_to_add = [row.path for row in df.collect()]
filenames_to_add = [path.split("/")[-1] for path in filepaths_to_add]

print(filepaths_to_add)

if df.count() == 0:
    dbutils.notebook.exit("No files to add to the index")


# COMMAND ----------

# DBTITLE 1, Update index with docs in doc_df
filepaths_added = []
doc_df = None
if spark.catalog.tableExists(f"{catalog}.{schema_brz}.{table_text}") and not spark.table(f"{catalog}.{schema_brz}.{table_text}").isEmpty() and not df.isEmpty() and document_type:
    doc_data = []
    for row in df.toLocalIterator():
        file_path = row['path']
        file_name = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        file_path = file_path.replace("dbfs:", "")
        try:
            text = index_update.utils.extract_text(file_path)
            doc_data.append({"file_name": file_name, "extracted_text": text, "folder_name": folder_path, "document_type": document_type})
            filepaths_added.append(file_path)
        except Exception as e:
            print(f"The file {file_name} could not be processed: {e}")
    try:
        doc_df = spark.createDataFrame(doc_data)
        doc_df.write.mode('append').saveAsTable(f"{catalog}.{schema_brz}.{table_text}")
        display(doc_df)
    except Exception as e:
        print(f"Doc data could not be saved. Error: {e}")

# COMMAND ----------

# DBTITLE 1, List new files and add them to table_track view
if doc_df and not doc_df.isEmpty():
    doc_df.createOrReplaceTempView("temp_table")  # Create a temporary table from the DataFrame

    spark.sql(f"""
    INSERT INTO {catalog}.{schema_brz}.{table_track} 
    SELECT file_name, folder_name, document_type, 'UPDATING' AS status 
    FROM temp_table
    WHERE NOT EXISTS (
        SELECT 1 FROM {catalog}.{schema_brz}.{table_track}
        WHERE temp_table.file_name = {catalog}.{schema_brz}.{table_track}.file_name AND
              temp_table.folder_name = {catalog}.{schema_brz}.{table_track}.folder_name AND
              temp_table.document_type = {catalog}.{schema_brz}.{table_track}.document_type
        )
    """)
else:
    print("doc_df is None or doc_df is empty")

# COMMAND ----------

# DBTITLE 1, Get chunks from new files and add it to chunks table
# Check if the table exists
if spark.catalog.tableExists(f"{catalog}.{schema_slv}.{table_chunks}") and not doc_df.isEmpty():

    # Parameters for the splitter
    chunk_size = 1000  
    chunk_overlap = 200

    # Define the UDF for splitting text into chunks with additional metadata
    def get_chunks_with_metadata(text):
        # Initialize the splitter
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = recursive_splitter.split_text(text)
        return [(chunk, len(chunk), i, len(chunks)) for i, chunk in enumerate(chunks)]

    # Register the UDF with Spark
    split_udf = F.udf(get_chunks_with_metadata, ArrayType(StructType([
        StructField("chunk", StringType(), False),
        StructField("chunk_length_in_chars", IntegerType(), False),
        StructField("chunk_location_in_doc", IntegerType(), False),
        StructField("total_chunks_in_doc", IntegerType(), False)
    ])))

    # Apply the text splitting UDF to create text chunks with metadata
    chunked_df = doc_df.withColumn("chunks_metadata", F.explode(split_udf(F.col("extracted_text"))))

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
    final_chunked_df.write.mode('append').saveAsTable(f"{catalog}.{schema_slv}.{table_chunks}")

# COMMAND ----------

# DBTITLE 1, Retrieving index and upserting chunks to it
vsc = VectorSearchClient()

index_name =f"{catalog}.{schema_gld}.{table_idx}"
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=index_name)

df = spark.sql(f"SELECT * FROM {catalog}.{schema_slv}.{table_chunks}")
df = df.filter(df.file_name.isin(filenames_to_add))
chunks_list = [row.asDict() for row in df.collect()]

print(f"Filling index '{index_name}' with data...")

index_update.utils.upsert_chunks_into_index(index, embedding_endpoint_name, chunks_list, 0)

print(f"Index '{index_name}' filled in successfully.")

# COMMAND ----------

# DBTITLE 1, List new files and add them to table_track
if doc_df and not doc_df.isEmpty():

    spark.sql(f"""
        UPDATE {catalog}.{schema_brz}.{table_track}
        SET status = 'UPDATED'
        WHERE file_name IN (SELECT file_name FROM temp_table)
          AND folder_name IN (SELECT folder_name FROM temp_table)
          AND document_type IN (SELECT document_type FROM temp_table)
          AND status = 'UPDATING'
    """)

# COMMAND ----------

# DBTITLE 1, Copy new files into flat volume
for filepath in filepaths_added:
    filename = os.path.basename(filepath)
    # Destination
    flat_volume_filepath = f"dbfs:/Volumes/{catalog}/{schema_brz}/{flat_volume}/{filename}"
    dbutils.fs.cp(filepath, flat_volume_filepath)
    print(f"{flat_volume_filepath} successfully copied")

# COMMAND ----------

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
display(duration)
