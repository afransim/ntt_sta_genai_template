# How to use the template code for GenAI use cases
## Table of contents
* [Pre-requisites](#pre-requisites)
* [Steps to customize the template](#steps-to-customize-the-template)
* [Project variables](#project-variables)
***

### Pre-requisites
* python should be installed
***

### Steps to customize the template
1. Fork the template repository
1. Fill the `cookiecutter.json` with the relevant information on the project variables, that will be used to name files and folders in the following step. Check [below](#project-variables) for more information on the project variables
1. Using the command line from the repository root, run the `init_project.py` script.
1. Once the variable names have been replaced and folders reorganized, confirm that the folder structure is the following:

    ```
    ├── .\github
        └── workflows
            └── bundle-cd-des.yml
    ├── [directory_name]
        ├── databricks.yml
        └── resources
            ├── chatbot-rag-resource.yml
            ├── create-objects-resource.yml
            ├── evaluate-model-resource.yml 
            ├── index-update-resource.yml
            └── remove-files-resource.yml
        ├── assemble_application
        ├── build_document_index
        ├── deploy_application
        ├── evaluate_model
        ├── index_update
        ├── remove_files
        └── tests
    ```
1. Push the repository content to the DES branch in GitHub

Once the repository is pushed to the DES branch, the CI/CD workflows will deploy the code to the DES Databricks workspace.
***

### Project variables
* **directory_name**: _string_, name of the folder containing the databricks asset bundle code
* **bundle_name**: _string_, name to be used upon bundle deploy
* **experiment_name**: _string_, name to be used to keep track of the Chatbot model changes and evaluation
* **des_workspace**: _string_, link to the desarrollo databricks workspace
* **pre_workspace**: _string_, link to the preproducción databricks workspace
* **pro_workspace**:_string_,  link to the producción databricks workspace
* **des_managed_location**:_string_, managed storage objects’ location for DES environment 
* **pre_managed_location**:_string_, managed storage objects’ location for PRE environment 
* **pro_managed_location**:_string_, managed storage objects’ location for PRO environment 
* **des_external_location**:_string_, external storage objects’ location for DES environment 
* **pre_external_location**:_string_, external storage objects’ location for PRE environment 
* **pro_external_location**:_string_, external storage objects’ location for PRO environment
* **group_name**, _string_, name of group of users on Databricks
* **finops_tag**. _string_, approved FinOps tag for the use case (implemented in clusters and serving endpoints)
* **vector_search_endpoint_name**, _string_, name of vector search endpoint
* **llm_endpoint_name**, _string_, name of llm model endpoint
* **embedding_endpoint_name**, _string_, name of embedding model endpoint
* **catalog**: _string_, catalog where core volumes, files and tables will be instantiated
* **use_case**: _string_, used for instantiating gold, silver and bronze schemas (gld_use_case, slv_use_case and brz_use_case, respectively)
* **hierarchy_volume**, _string_, name of databricks volume where files are stored
* **hierarchy_volumne_core_folder**, _string_, name of the older, within the **hierarchy_volume**, that stores the core raw files
* **hierarchy_volume_user_temporary_folder**, _string_, name of folder, within Volume, that stores the temporary raw files
* **flat_volumne**, _string_, name of Volume that stores files in flat directory
* **table_functional_name**, _string_, table name to instantiate required tables
* **model_name**: _string_, name to be used for the GenAI model; should describe the GenAI task
* **document_type**, _string_, definition of document type as core (core) or temporary (tmp)
* **file_arrival_directory**, _string_, full path to temporary files folder
* **min_time_between_triggers_seconds**, _integer_, minimum number of seconds betweens file arrival triggers for the index update workflow 
* **remove_files_workflow_schedule**, _string_, [cron expression](https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm) that will determine the periodicity with which the remove files workflow will be executed
* **timezone_id**: _string_, timezone in which the remove files workflows should be executed
* **notification_email**: _string_, email to which the alerts will be sent when the workflow succeeds or fails
* **chunk_size**: _string_, total size of each chunk
* **chunk_overlap**: _string_, total size of each chunk that can be overlapping with other chunks