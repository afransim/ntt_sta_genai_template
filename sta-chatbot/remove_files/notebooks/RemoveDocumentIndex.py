# Databricks notebook source
##################################################################################
# The purpose of this notebook is to access and delete temporary files from vector index, volumes and tables. 
#
# This notebook has the following parameters:
#
# * catalog (required)                                    - Catalog name to read raw files and create tables.
# * use_case (required)                                   - Use case name to instantiate gold, silver and bronze schemas.
# * hierarchy_volume (required)                           - Name of Volume where files are stored.
# * hierarchy_volume_user_temporary_folder (required)     - Name of Volume directory where tmp files are stored.
# * table_functional_name (required)                      - Table name to instantiate required tables.
# * flat_volume (required)                                - Name of Volume that stores files in flat directory
# * vector_search_endpoint_name (required)                - Name of vector search endpoint.
# * embedding_endpoint_name (required)                    - Name of embedding endpoint.
# * bundle_root (required):                               - Databricks bundle root path.

# COMMAND ----------
# DBTITLE 1, Relevant library and helper function imports
import sys
import os

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.types import StringType, StructType, StructField

root = dbutils.widgets.get("bundle_root")
sys.path.append(root)

import remove_files.utils as utils

# COMMAND ----------
# DBTITLE 1, Instantiate necessary dbutils
dbutils.widgets.text("vector_search_endpoint_name", "", label="VectorSearch Endpoint Name")
dbutils.widgets.text("embedding_endpoint_name","",label="Embedding Endpoint Name")
dbutils.widgets.text("catalog","alt_des",label="Catalog")
dbutils.widgets.text("use_case","chatbot",label="Use Case")
dbutils.widgets.text("hierarchy_volume","",label="Hierarchy Volume")
dbutils.widgets.text("hierarchy_volume_user_temporary_folder","",label="Hierarchy Volume User Temporary Folder")
dbutils.widgets.text("flat_volume","",label="Flat Volume")
dbutils.widgets.text("table_functional_name","",label="Table Text")

catalog = dbutils.widgets.get("catalog")
valid_domains = ["admin", "agr", "cct", "clt", "eco", "edu", "ell", "fph", "ind", "inf", "jud", "mam", "mob", "slt", "sem", "sso", "sci", "societat", "tec", "ter", "trb", "nco", "alt"]
assert catalog.split("_")[-1] in ['des', 'pre', 'pro'], f"Catalog name validation failed: {catalog.split('_')[-1]} environment is not valid"
assert catalog.split("_")[0] in valid_domains, f"Catalog name validation failed: {catalog.split('_')[0]} domain is not valid"

vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint_name")
embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

hierarchy_volume = dbutils.widgets.get("hierarchy_volume")
hierarchy_volume_user_temporary_folder = dbutils.widgets.get("hierarchy_volume_user_temporary_folder")
flat_volume = dbutils.widgets.get("flat_volume")
table_functional_name = dbutils.widgets.get("table_functional_name")

use_case = dbutils.widgets.get("use_case")
schema_brz = "brz_"+use_case
schema_slv = "slv_"+use_case
schema_gld = "gld_"+use_case

table_text = "ftr_"+table_functional_name+"_txt"
table_track = "aux_"+table_functional_name+"_track"
table_chunks = "ftr_"+table_functional_name+"_chunks"
table_idx = "ftr_"+table_functional_name+"_idx"

# COMMAND ----------
# DBTITLE 1, Remove files from temporary user files folder 
folderpath = f"dbfs:/Volumes/{catalog}/{schema_brz}/{hierarchy_volume}/{hierarchy_volume_user_temporary_folder}"
filepaths_to_delete = dbutils.fs.ls(folderpath)
if len(filepaths_to_delete) != 0:
    for file in filepaths_to_delete:
        filepath = file.path
        dbutils.fs.rm(filepath, recurse=False)
        print(f"{filepath} successfully deleted")
else:
    print("No files to be deleted were found.")

# COMMAND ----------
# DBTITLE 1, Get all uploaded files
df_table = spark.table(f"{catalog}.{schema_brz}.{table_track}")
df_table = df_table.withColumn("path", concat(col("folder_name"), lit("/"), col("file_name"))).select("path")
df_table.collect()

# COMMAND ----------
# DBTITLE 1, Listing files uploaded to volume
rdd_schema = StructType([StructField("path", StringType(), True)])

volume_path =f"/Volumes/{catalog}/{schema_brz}/{hierarchy_volume}/"
file_paths = [(t,) for t in utils.list_files(dbutils, volume_path)]
df_volume = spark.createDataFrame(file_paths, schema=rdd_schema)
df_volume.collect()

# List all deleted files
df_deleted_files = df_table.subtract(df_volume)
filepaths_deleted = [row.path for row in df_deleted_files.collect()]
filepaths_deleted = [path.split("/")[-1] for path in filepaths_deleted]
print(filepaths_deleted)

# COMMAND ----------
# DBTITLE 1, Listing deleted files ids
ids_to_delete = utils.get_deleted_files_id_from_table(catalog, schema_slv, table_chunks, filepaths_deleted, spark=spark)
print(ids_to_delete)

# COMMAND ----------
# Remove files from chunk table
utils.remove_data_from_table(catalog, schema_slv, table_chunks, filepaths_deleted, spark=spark)

# Remove files from text table
utils.remove_data_from_table(catalog, schema_brz, table_text, filepaths_deleted, spark=spark)

# COMMAND ----------
# DBTITLE 1, Update index table
vsc = VectorSearchClient()
index_name = f"{catalog}.{schema_gld}.{table_idx}"
index = vsc.get_index(endpoint_name=vector_search_endpoint, index_name=index_name)
if ids_to_delete:
    index.delete(ids_to_delete)

# COMMAND ----------
# DBTITLE 1, Remove files from track table
utils.remove_data_from_table(catalog, schema_brz, table_track, filepaths_deleted, spark=spark)

# COMMAND ----------
# DBTITLE 1, Remove file in flat folder volume
if len(filepaths_deleted) != 0:
    for filepath in filepaths_deleted:
        filename = os.path.basename(filepath)
        flat_volume_filepath = f"dbfs:/Volumes/{catalog}/{schema_brz}/{flat_volume}/{filename}"
        dbutils.fs.rm(flat_volume_filepath, recurse=False)
        print(f"{flat_volume_filepath} successfully deleted")
else:
    print("No files to be deleted were found.")