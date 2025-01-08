# sta-chatbot
This repository contains the code to deploy a [Databricks Asset Bundle](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/), containing a set of workflows for building and deploying a GenAI model to address the chatbot use case. 

## Table of contents
* [sta-chatbot Folder Content](#sta-chatbot-Folder-Content)
* [Bundle workflows Description](#bundle-workflows-description)
* [DAB Configuration](#dab-configuration)
    * [Model Workflow](#model-workflow)
    * [Batch Inference Workflow](#batch-inference-workflow)
* [Testing](#testing)
    * [Unit Tests](#unit-tests)
    * [Integration Tests](#integration-tests)
    
***

## sta-chatbot Folder Content
|Component|Description|
|---|---|
|Resources folder|Folder containing yml files that define the different tasks of the bundle|
|Task folders|Each folder containing a folder (`notebooks`) with a databricks notebook to define the task and a python module with the necessary helper functions|
|Tests folder|Folder containg integration and unit tests' modules|

```
    sta-chatbot     <- Template directory. Contains python code, notebooks and GenAI resources related to one GenAI project.
        │
        ├── databricks.yml                      <- Root bundle file for the GenAI project that can be loaded by databricks CLI 
        |                                          bundles. It defines the bundle name, workspace URL and resource config components to be included.
        │
        ├── pytest.ini                          <- Integration and unit tests' configuration file.
        │
        ├── requirements.txt                    <- Text file containing the necessary libraries.
        │                    
        ├── resources                           <- Databricks workflows configuration definitions expressed as .yml code.
        │   │
        │   ├── chatbot-rag-resource.yml                <- GenAI resource config definition for the chatbot rag building, assembling and deployment.
        │   │
        │   ├── evaluate-model-resource.yml             <- GenAI resource config definition for model evaluation.
        │   │
        │   ├── index-update-resource.yml               <- GenAI resource config definition for updating the vector index for the current model upon upload of a new document.
        │   │
        │   └── remove-files-resource.yml               <- GenAI resource config definition for removing temporary file information from vector index, volumes and tables.
        │
        ├── assemble_application                <- Folder containing source code for the task that assembles the RAG application.
        │   │
        │   └── notebooks                               <- Assemble application notebook folder.
        │       │
        │       ├── Assemble_Application.py                     <- Assemble application notebook.
        │       │
        │       └── chain_history.py                            <- Python module that defines the logic to implement the RAG/LLM.
        │
        ├── build_document_index                <- Folder containing source code for build document index task.
        │   │                
        │   ├── utils.py                                <- Helper functions for the Build document index task.
        │   │
        │   └── notebooks                               <- Build document index notebook folder.
        │       │
        │       └── Build_Document_Index.py                     <- Build document index notebook.
        │
        ├── create_objects                  <- Folder containing source code for the create objects task.
        │   │
        │   ├── utils.py                                <- Helper functions for the create objects task.
        │   │
        │   └── notebooks                               <- Create objects notebook folder.
        │       │
        │       └── CreateObjects.py                       <- Create objects notebook.
        │
        ├── deploy_application                  <- Folder containing source code for the Deploy application task.
        │   │
        │   ├── utils.py                                <- Helper functions for the Deploy application task.
        │   │
        │   └── notebooks                               <- Deploy application notebook folder.
        │       │
        │       └── Deploy_Application.py                       <- Deploy application notebook.
        │
        ├── index_update                        <- Folder containing source code for the Update document index task.
        │   │
        │   ├── utils.py                                <- Helper functions for the Update document index task.
        │   │
        │   └── notebooks                               <- Update document index task notebook folder.
        │       │
        │       └── UpdateDocIndex_genai_model.py               <- Update document index task notebook.
        │
        ├── remove_files                        <- Folder containing source code for the Remove files task
        │   │
        │   ├── utils.py                                <- Helper functions for the Remofe files task.
        │   │
        │   └── notebooks                               <- Remove files notebook folder.
        │       │
        │       └── RemoveDocumentIndex.py                      <- Remove files task notebook.
        │
        └── tests                               <- Folder containing source code for unit and integration testing.
            │
            ├── __init__.py                             <- Initialisation file.
            │
            ├── integration_tests                       <- Integration tests folder.
            │   │
            │   ├── __init__.py                                 <- Initialisation file.
            │   │
            │   ├── test_chatbot_app.py                         <- Chatbot app endpoint testing.
            │   │
            │   ├── test_chatbot_llmp.py                        <- Chatbot llm endpoint testing.
            │   │
            │   ├── test_embeddings.py                          <- Embedding endpoint testing.
            │   │
            │   └── test_vector_search_index.py                 <- Vector search index endpoint testing.
            │
            └── unit_tests                              <- Unit tests folder.
                │
                ├── __init__.py                             <- Initialisation file.
                │
                ├── test_build_document_index.py            <- Build document index task - helper methods testing.
                │
                ├── test_deploy_application.py              <- Deploy application task - helper methods testing.
                │
                ├── test_evaluate_model.py                  <- Evaluate model task - helper methods testing.
                │
                ├── test_remove_deleted_file_info.py        <- Remove Deleted Files - helper methods testing.
                │
                └── test_update_index.py                    <- Update index - helper methods testing.
```

## Bundle Workflows Description
There are four main workflows already implemented in this bundle, defined in the `resources` folder:
1. **Chatbot rag** ([chatbot rag resource](./resources/chatbot-rag-resource.yml)): this workflow will build, assemble and validate the chatbot_model model, setting up a vector search endpoint and index based on the core files, and deploying an embedding endpoint as well. It also creates the necessary tables (text, chuncks, track and index) to maintain the chatbot_model and update it with revelant temporary data.
1. **Index update** ([index update resource](./resources/index-update-resource.yml)): this workflow will update the vector search index and embedding endpoints of the  chatbot_model to include new, temporary, files into the knowledge base and respective tables, triggered by the file arrival at a given directory (/Volumes/admin_govern_sta_des/brz_chatbot/ctti_rag_data/user_files/temporary_files/).
1. **Remove files** ([remove files resource](./resources/remove-files-resource.yml)): this workflow will remove the temporary files from the chatbot_model knowledge base, as well as the respective files that were added, triggered by a schedule.
1. **Evaluate model** ([evaluate model resource](./resources/evaluate-model-resource.yml)): this workflow will test the responses of the deployed chatbot_model model with a pre-defined set of user question-answer pairs.

***
## DAB Configuration

The [bundle resource file](databricks.yml), located in sta-chatbot, builds the Databricks Asset Bundle (DAB), defining the bundle name, experiment name, model name, the path to the .yml files associated to each workflow, and the workspaces.

### Chatbot RAG workflow
The model workflow is located in the [model workflow resource file](./resources/chatbot-rag-resource.yml). In this workflow, there are three different sequential tasks defined, each associated with a notebook:

* **[Build Document Index](./build_document_index/notebooks/Build_Document_Index.py)**: this task instantiates the required schemas and tables to deploy and maintain the chatbot_model application. It reads the core, raw, files from the respective volume and processes the text into chuncks and subsequently into embeddings, finally deploying the required embedding, vector search and llm endpoints, as well as the vector search index. This task takes the following parameters: environment, catalog, use case, model name, hierarchy volume, hierarchy volume core folder, hierarchy volume temporary folder, table functional name, bundle root path and vector search, embedding and llm endpoint names.
* **[Assemble Application](./assemble_application/notebooks/Assemble_Application.py)**: this task registers the model in Unity Catalog and logs it into Mlflow. This tasks takes the model name as a parameter and inherits the remaining parameters from the previous task (**[Build Document Index](./build_document_index/notebooks/Build_Document_Index.py)**), namely: catalog, use case, gold schema, table index and vector search, embedding and llm endpoint names.
* **[Deploy Application](./deploy_application/notebooks/Deploy_Application.py)**: this task deploys the custom model registered with MLflow in the prior notebook (**[Assemble Application](./assemble_application/notebooks/Assemble_Application.py)**) to Databricks model serving. The parameters for this task are mostly inherited from the previous task, namely:  catalog, gold schema, model name and use case, only taking the bundle root path as a parameter

### Index Update workflow
The index update workflow is located in the [index update resource file](./resources/index-update-resource.yml). This workflow comprises one single task, with the same name, which calls the document index instantiated in the previous workflow and updates it, and the respective tables, with temporary file content and information. This task takes the following parameters: environment, catalog, use case, hierarchy volume, table functional name, bundle root path, document type and vector seach, embedding and llm endpoint names.

### Remove Files workflow
The remove files workflow is located in the [remove files resource file](./resources/index-update-resource.yml). This workflow comprises one single task, with the same name, which accesses and deletes the temporary file information from the vector index, volumes and tables. This task takes the following parameters: catalog, use case, hierarchy volume and hierarchy volume user temporary folder, table functional name, flat volume, bundle root path and the vector search and embedding endpoint names.

### Evaluate model workflow
The evaluate model workflow is located in the [evaluate model resource file](./resources/evaluate-model-resource.yml). This workflow comprises one single task, with the same name, which evaluates the currently deployed model by asking a pre-defined set of user questions and saving the model output for later human evaluation. The parameters for this notebook are the following: catalog, use case, model name and bundle root path.

***
## Testing

A set of tests have been designed for unit and integration testing of the different components. These are essential to ensure that the software developed runs as expected, and remains functional even after code changes. These tests are made to be run when the chatbot use case is promoted from the development to the preproduction environment.

All tests are properly logged, so that any issues can be diagnosed by consulting the logs that are outputted to the console and/or to the log file located in the [test_run.log](./tests/logs/tests_run.log) file. This and other options can be further customised in the test configuration file [pytest.ini](./pytest.ini), where the path to the test files, and their prefix for test retrieval are defined, as well as markers, and log configuration.

### Unit Tests

The unit tests are located in [./tests/unit_tests](./tests/unit_tests), and cover function testing and file format testing. The unit tests include:
* [test_build_document_index](./tests/unit_tests/test_build_document_index.py): ensures that the helper methods for the Build Document Index task (**[build document index utils](./build_document_index/utils.py)**) function as expected.
* [test_deply_application](./tests/unit_tests/test_deploy_application.py): ensures that the helper methods for the Deploy Application task (**[deploy application utils](./deploy_application/utils.py)**) function as expected.
* [test_evaluate_model](./tests/unit_tests/test_evaluate_model.py): ensures that the helper methods for the Evaluate Model task (**[evaluate model utils](./evaluate_model/utils.py)**) function as expected. 
* [test_remove_deleted_file_info](./tests/unit_tests/test_remove_deleted_file_info.py): ensures that the helper methods for the Remove Deleted Files task (**[remove files utils](./remove_files/utils.py)**) function as expected.
* [test_update_index](./tests/unit_tests/test_update_index.py): ensures that the helper methods for the Update Index task (**[update index utils](./index_update/utils.py)**) function as expected.

More unit tests can be implemented to verify any new functionality specific to each use case.

### Integration Tests

The integration tests, usually designed to ensure the correct integration of different modules, are located in [./tests/integration_tests](./tests/integration_tests/). These are ran in the pre workspace, and start by the execution of the Chatbot RAG Workflow. For the chatbot use case, we test the following integrations:
* **[test_chatbot_app](./tests/integration_tests/test_chatbot_app.py)**: ensures connection to the chatbot app endpoint.
* **[test_chatbot_llm](./tests/integration_tests/test_chatbot_llm.py)**: ensures connection to the llm endpoint.
* **[test_embeddings](./tests/integration_tests/test_embeddings.py)**: ensures connection to the embedding endpoint.
* **[test_vector_search_index](./tests/integration_tests/test_vector_search_index.py)**: ensures connection to the vector search index.

Just as with unit tests, other integration tests can be devised to confirm integration with new features and modules.
