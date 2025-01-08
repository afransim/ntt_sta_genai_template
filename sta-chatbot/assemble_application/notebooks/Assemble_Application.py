# Databricks notebook source
##################################################################################
# This notebook instantiates the model in Unity Catalog and logs it into Mlflow.
#
# This notebook has the following parameter:
#
# * model_name (required) - Should describe the GenAI task
##################################################################################
# COMMAND ----------
# DBTITLE 1,Configuration of model name, unity catalog locations, endpoint names etc.
dbutils.widgets.text("model_name", "model_name", label="Model Name")
model_name = dbutils.widgets.get("model_name")
assert model_name.split("_")[-1]=="model", f"Model name validation failed: {model_name.split('_')[-1]}. Should end with _model."

catalog=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "catalog")
use_case=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "use_case")
gld_schema=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "gld_schema")
table_idx=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "table_idx")
vector_search_endpoint_name=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "vector_search_endpoint_name")
llm_endpoint_name=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "llm_endpoint_name")
embedding_endpoint_name=dbutils.jobs.taskValues.get(taskKey = "BuildDocumentIndex", key = "embedding_endpoint_name")

chain_config_file = 'rag_chain_config.yaml'

# COMMAND ----------
# DBTITLE 1,Configuration of rag chatmodel instructions
import mlflow
import yaml

rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": llm_endpoint_name,
        "vector_search_endpoint_name": vector_search_endpoint_name,
        "embedding_endpoint_name": embedding_endpoint_name,
    },
    "input_example": {
        "messages": [
            {'content': 'En quin format s’ha de facilitar l’accés a la informació pública?','role': 'user'},
            {"role": "assistant", "content": "En format reutilitzable."},
            {"role": "user", "content": "Si us plau, expliqueu la resposta amb més detalls."}
        ],
        "selected_documents": [
            "1920373 Llei 19-2013, de 9 de desembre, de transparència, accés a la informació pública i bon govern.pdf",
            "2017238 DECRET 8-2021, de 9 de febrer, sobre la transparència i el dret d'accés a la informació pública.pdf"
        ]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 4096, "temperature": 0.15},
     
        "contextualize_q_system_prompt" : "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",

        "qa_system_prompt"  : "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n\n CONTEXT: \n {context} \n\n If the context is [no context found.] respond with [No s'ha trobat informació rellevant a la base de dades vectorial. Revisa si el document seleccionat existeix.] \n Ensure the response is in the same language as the conversation below:"
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "parameters": {"k": 5, "query_type": "ann"},
        "data_pipeline_tag": "poc",
        "schema": {"text_chunk":"text_chunk", "file_name":"file_name", "folder_name":"folder_name", "chunk_location_in_doc":"chunk_location_in_doc",
                   "total_chunks_in_doc":"total_chunks_in_doc","document_type":"document_type"},
        "vector_search_index": f"{catalog}.{gld_schema}.{table_idx}",
    },
}

try:
    with open(chain_config_file, 'w') as f:
        yaml.dump(rag_chain_config, f)
except Exception as e:
    print('pass to work on build job: {e}')
    
model_config = mlflow.models.ModelConfig(development_config=chain_config_file)

# COMMAND ----------
# DBTITLE 1,Logging of model using MLflow
import os
from mlflow.models.signature import infer_signature, ModelSignature

mlflow.set_registry_uri("databricks-uc")

# Create a manual signature based on the input_example
input_signature = infer_signature(model_config.get("input_example"))

# You might need to create a mock output example to infer the output signature
mock_output_example = {
    "response": "Gencat is a public organization..."
}
output_signature = infer_signature(model_config.get("input_example"), mock_output_example)

signature = ModelSignature(inputs=input_signature.inputs, outputs=output_signature.outputs)

# Log the model to MLflow
with mlflow.start_run(run_name="chatbot"):
    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(os.getcwd(), 'chain_history.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config=chain_config_file,  # Chain configuration 
        artifact_path="chain_history",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        signature=signature, # Use the manually defined signature
        registered_model_name=f"{catalog}.{gld_schema}.{model_name}",
    )

print(logged_chain_info.model_uri)
print(type(logged_chain_info.model_uri))

# Test the chain locally
chain_loaded = mlflow.pyfunc.load_model(logged_chain_info.model_uri)

# COMMAND ----------
# DBTITLE 1,Testing of model with test query
model_input  = {'messages': [{'role': 'user','content': 'En quin format s’ha de facilitar l’accés a la informació pública?'},
                             {'role': 'assistant','content': "<p>L'accés a la informació pública ha de ser facilitat en un format reutilitzable. Això significa que la informació ha de ser presentada de tal manera que pugui ser explotada mitjançant la reproducció i divulgació per qualsevol mitjà. Això permet la creació de productes o serveis d'informació amb valor afegit basats en les dades públiques. A més, la informació reutilitzable ha de ser posada a disposició de manera que sigui processable de forma automatitzada, conjuntament amb les metadades i en formats oberts.</p><p>Documents utilitzats:</p><ul><li>2017238 DECRET 8-2021, de 9 de febrer, sobre la transparència i el dret d'accés a la informació pública.pdf</li><li>1920373 Llei 19-2013, de 9 de desembre, de transparència, accés a la informació pública i bon govern.pdf</li></ul>"},
                             {'role': 'user','content': "Com assegura el govern la transparència i l'actualització de les dades en formats reutilitzables?"}],
                "selected_documents": [
                    "1920373 Llei 19-2013, de 9 de desembre, de transparència, accés a la informació pública i bon govern.pdf",
                    "2017238 DECRET 8-2021, de 9 de febrer, sobre la transparència i el dret d'accés a la informació pública.pdf"
                    ]
                }

chain_loaded.predict(model_input)

# COMMAND ----------
# DBTITLE 1, Persisting variables needed for subsequent tasks

dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("use_case", use_case)
dbutils.jobs.taskValues.set("catalog", catalog)
dbutils.jobs.taskValues.set("gld_schema", gld_schema)