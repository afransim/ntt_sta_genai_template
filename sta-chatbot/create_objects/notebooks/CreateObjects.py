# Databricks notebook source
##################################################################################
# The purpose of this notebook is to create the objects necessary for the chabot deployment. 
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
# * des_managed_location (required) - Dev managed storage location that stores catalogs, schemas and tables.
# * pre_managed_location (required) - Pre managed storage location that stores catalogs schemas and tables.
# * pro_managed_location (required) - Pro managed storage location that stores catalogs schemas and tables.
# * des_external_location (required) - Dev external storage location that stores volumes.
# * pre_external_location (required) - Pre external storage location that stores volumes.
# * pro_external_location (required) - Pro external storage location that stores volumes.
# * bundle_root (required) - Databricks bundle root path.
##################################################################################
# COMMAND ----------
# DBTITLE 1,Import Required Functions
import os

dbutils.widgets.dropdown("env", "des", ["des", "pre", "pro"], label="Environment Name")
dbutils.widgets.text("catalog", "alt_des", label="Catalog")              
dbutils.widgets.text("use_case", "chatbot", label="Use Case")
dbutils.widgets.text("hierarchy_volume", "ctti_rag_data", label="Hierarchy Volume")              
dbutils.widgets.text("hierarchy_volume_core_folder", "core_files", label="Hierarchy Volume Core Folder")
dbutils.widgets.text("hierarchy_volume_user_temporary_folder", "user_files/temporary_files", label="Hierarchy Volume User Temporary Folder")
dbutils.widgets.text("table_functional_name", "CTTI_DOCS", label="Table functional name for text, tracking, chunks and index tables")
dbutils.widgets.text("des_managed_location", "", label="des_managed_location")
dbutils.widgets.text("pre_managed_location", "", label="pre_managed_location")
dbutils.widgets.text("pro_managed_location", "", label="pro_managed_location")
dbutils.widgets.text("des_external_location", "", label="des_external_location")
dbutils.widgets.text("pre_external_location", "", label="pre_external_location")
dbutils.widgets.text("pro_external_location", "", label="pro_external_location")
dbutils.widgets.text("bundle_root", "default", label="bundle_root")

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
des_managed_location = dbutils.widgets.get("des_managed_location")
pre_managed_location = dbutils.widgets.get("pre_managed_location")
pro_managed_location = dbutils.widgets.get("pro_managed_location")
des_external_location = dbutils.widgets.get("des_external_location")
pre_external_location = dbutils.widgets.get("pre_external_location")
pro_external_location = dbutils.widgets.get("pro_external_location")
bundle_root = dbutils.widgets.get("bundle_root")
 
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

from create_objects.utils import create_volume

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


# COMMAND ----------
DB_HIERARCHY_VOLUME_CORE_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{hierarchy_volume}/{hierarchy_volume_core_folder}"
DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{hierarchy_volume}/{hierarchy_volume_user_temporary_folder}"
DB_FLAT_VOLUME_DESTINATION = f"/Volumes/{catalog}/{brz_schema}/{flat_volume}"
TMP_FILE_PATH = "/tmp/tmp_file"

# COMMAND ----------
# DBTITLE 1, Create schemas
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{brz_schema} MANAGED LOCATION '{managed_location}/catalogs/{catalog}'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{slv_schema} MANAGED LOCATION '{managed_location}/catalogs/{catalog}'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{gld_schema} MANAGED LOCATION '{managed_location}/catalogs/{catalog}'")

# COMMAND ----------
# DBTITLE 1, Create volumes and main subfolders
# Hierarchy volume
create_volume(catalog, brz_schema, hierarchy_volume, managed_location, external_location)
if not os.path.exists(DB_HIERARCHY_VOLUME_CORE_DESTINATION):
    print(f"{DB_HIERARCHY_VOLUME_CORE_DESTINATION} not found. Creating...")
    dbutils.fs.mkdirs(DB_HIERARCHY_VOLUME_CORE_DESTINATION)
    print(f"{DB_HIERARCHY_VOLUME_CORE_DESTINATION} was successfully created")
else:
    print(f"{DB_HIERARCHY_VOLUME_CORE_DESTINATION} already existing")

if not os.path.exists(DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION):
    print(f"{DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION} not found. Creating...")
    dbutils.fs.mkdirs(DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION)
    print(f"{DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION} was successfully created")
else:
    print(f"{DB_HIERARCHY_VOLUME_USER_TMP_DESTINATION} already existing")

# Flat volume
create_volume(catalog, brz_schema, flat_volume, managed_location, external_location)

# COMMAND ----------
# Create table to hold extracted text (if doesn't exist yet)
spark.sql(f"""CREATE TABLE IF NOT EXISTS `{catalog}`.`{brz_schema}`.`{table_text}` (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY, 
    file_name STRING,
    extracted_text STRING,
    folder_name STRING,
    document_type STRING
    ) tblproperties (delta.enableChangeDataFeed = true)""")

# Create table to track which files we've already processed (if doesn't exist yet)
spark.sql(f"""CREATE TABLE IF NOT EXISTS `{catalog}`.`{brz_schema}`.`{table_track}` (
    file_name STRING,
    folder_name STRING,
    document_type STRING,
    status STRING
    ) tblproperties (delta.enableChangeDataFeed = true)""")

# Create table to store the text chunks (if doesn't exist yet)
spark.sql(f"""CREATE TABLE IF NOT EXISTS `{catalog}`.`{slv_schema}`.`{table_chunks}` (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY, 
    file_name STRING,
    text_chunk STRING,
    folder_name STRING,
    document_type STRING,
    chunk_location_in_doc INTEGER,
    total_chunks_in_doc INTEGER,
    chunk_length_in_chars INTEGER
    ) tblproperties (delta.enableChangeDataFeed = true)""")


