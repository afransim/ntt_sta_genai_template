import mlflow
import re
import pandas as pd
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, ConfigurableField
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import DatabricksEmbeddings
import time

# Enable MLflow Tracing
mlflow.langchain.autolog()

# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    """
    Extract the latest user query to the LLM in string format

    Args:
        chat_messages_array (List[str]): contains user prompts to model
    
    Returns:
        str: Latest user message to the LLM
    """
    return chat_messages_array[-1]["content"]


# Get the config from the local config file
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# Connect to the Vector Search Index
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
    index_name=retriever_config.get("vector_search_index"),
)
vector_search_schema = retriever_config.get("schema")

# Initialize the Databricks vector store with configurable field (to use only chunks from selected document)
semantic_search_vectorstore = DatabricksVectorSearch(
    vs_index,
    embedding= DatabricksEmbeddings(endpoint=databricks_resources.get("embedding_endpoint_name")),
    text_column=vector_search_schema.get("text_chunk"),
    columns=[
        vector_search_schema.get("text_chunk"),
        vector_search_schema.get("file_name"),
        vector_search_schema.get("folder_name"),
        vector_search_schema.get("chunk_location_in_doc"),
        vector_search_schema.get("total_chunks_in_doc"),
        vector_search_schema.get("document_type")
    ]
    ).as_retriever(
        search_kwargs=retriever_config.get("parameters")
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs_dynamic",
            name="Search Kwargs",
            description="Dynamic search parameters to filter by document name.",
        )
    )

# Intialize the LLM used for reasoning and question answering
model = ChatDatabricks(
    endpoint=databricks_resources.get("llm_endpoint_name"),
    extra_params=llm_config.get("llm_parameters"),
)

# Extracts just the answer from historical answers that are html formatted
def extract_only_answer_from_html(html_string):

    """
    Extract LLM answer from HTML source

    Args:
        html_string (str): contains formatted HTML content for frontend with LLM response
    
    Returns:
        str: Section of HTML string with the LLM answer (between <p>...</p> tags)
    """

    # Define the regex pattern to find text between the first <p> and </p> tags
    pattern = re.compile(r'<p>(.*?)</p>', re.DOTALL)    
    # Search for the pattern in the HTML string
    match = pattern.search(html_string)    
    if match:
        # Extract the content of the first <p>...</p> pair
        answer_part = match.group(1)
    else:
        # If no <p>...</p> pair is found, return the full text
        answer_part = html_string
    return answer_part

def format_chat_history(chat_messages_array):

    """
    Format chat history in langchain message style

    Args:
        chat_messages_array (List[str]): Chat history array
    
    Returns:
        List[str]: Langchain-formatted chat history array
    """

    formatted_chat_history = []
    if len(chat_messages_array) > 0:
        for chat_message in chat_messages_array:
            if chat_message["role"] == "user":
                formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
            elif chat_message["role"] == "assistant":
                # Get the answer part of the HTML string
                only_answer = extract_only_answer_from_html(chat_message["content"])              
                formatted_chat_history.append(AIMessage(content=only_answer))
    return formatted_chat_history

# Prompt to create suitable question for text retrieval (for follow-up questions like "Explain it with more words")
prompt_to_rewrite_question_using_chat_history = ChatPromptTemplate.from_messages(
    [
        ("system", llm_config["contextualize_q_system_prompt"]),
        MessagesPlaceholder("formatted_chat_history")
    ]
)

def format_context(docs):

    """
    Method to format the docs returned by the retriever into the prompt

    Args:
        docs: Embedded documents for formatting according to chunking template
    
    Returns:
        dict: Dict with formatted text chunkings and respective file names
    """

    chunk_template = retriever_config.get("chunk_template")
    if docs:
        chunk_contents = [
            chunk_template.format(
                chunk_text=d.page_content
            )
            for d in docs
        ]
        chunk_sources = [
            d.metadata[
                vector_search_schema.get("file_name")]
            for d in docs
        ]

    else:
        chunk_contents = "no context found."
        chunk_sources = []

    def remove_duplicates(input_list):
        """
        Remove duplicates from input list

        Args:
            input_list (List[_]): List of items to remove duplicates from

        Returns:
            List[]: Returns list of items without duplicates
        """

        seen = set()
        result = []
        for item in input_list:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result

    chunk_information = {"text": "".join(chunk_contents), "file_names": remove_duplicates(chunk_sources)}
    return chunk_information
    
def get_sources(input_dictionary):

    """
    Function to retrieve names of documents used in retrieval step

    Args:
        input_dictionary (dict): Dictionary with context and respective file names used for that context
    
    Returns:
        List[str]: List of file names used in retrieval
    """

    return input_dictionary["context"]["file_names"]

# Prompt Template for generation of final answer
prompt_to_answer_final_question = ChatPromptTemplate.from_messages(
    [
        ("system", llm_config["qa_system_prompt"]),
        # Note: This chain does not compress the history, but only the last 10 message are send by front-end.
        MessagesPlaceholder(variable_name="formatted_chat_history")
    ]
)

def format_final_answer(data):

    """
    Combine final answer and documents used and format as HTLM string

    Args:
        data (Any): Object that contains model answers for formatting into HTML
    
    Returns:
        str: HTML-formatted string with model response
    """

    # Replace newline characters with <br> tags in the answer
    answer_text=data['answer'].replace('\n', '<br>')
    answer_html = f"<p>{answer_text}</p>"
    sources_html = "<ul>"
    for source in data['sources']:
        sources_html += f"<li>{source}</li>"
    sources_html += "</ul>"
    result_html = f"{answer_html}<p>Documents utilitzats:</p>{sources_html}"
    return result_html

# Composition of final chain of actions
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "formatted_chat_history": itemgetter("messages") | RunnableLambda(format_chat_history),
    }
    | RunnablePassthrough()
    | 
    {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["formatted_chat_history"]) > 1,
                prompt_to_rewrite_question_using_chat_history | model | StrOutputParser(),
            ),
            itemgetter("question"), # so if is statement returns false, the original question is used
        )
        | semantic_search_vectorstore
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    |
    { 
        "answer": prompt_to_answer_final_question
        | model
        | StrOutputParser(),
        "sources": RunnableLambda(get_sources)
    }
    |
    RunnableLambda(format_final_answer)
)

# Wrapper to pass the selected documents into the chain as config
class LangChainModelWrapper(mlflow.pyfunc.PythonModel):

    mlflow.trace(name="langchain-predict")
    def predict(self, context, model_input):

        """
        Get the response of the model for a prompt.

        Args:
            self: Current LangChainModelWrapper
            context (Any): Contains context for model
            model_input (dict): Dictionary containing the messages and selected documents for the prediction
        
        Returns:
            str: HTML-formatted string with model response
        """

        messages = model_input["messages"]
        selected_documents = model_input.get("selected_documents",None)

        if isinstance(selected_documents, pd.Series):
            selected_documents = selected_documents[0]
        if isinstance(messages, pd.Series):
            messages = messages[0]
        
        if selected_documents:
            config = {"configurable": {"search_kwargs_dynamic" : {"k": 5, "query_type": "ann", "filters": {"file_name":selected_documents}}}}
        else:
            config = {"configurable": {"search_kwargs_dynamic" : {"k": 5, "query_type": "ann"}}}
        
        with mlflow.start_span(name="document_query") as query_span:
            query_start_time = time.time()
            query_span.set_inputs({"messages": messages, "config": config})

            result = chain.invoke(
                    {"messages": messages},
                    config=config
                )
        
            query_execution_time = time.time() - query_start_time

            query_span.set_outputs({"result": result})

            query_span.set_attributes({
                "num_selected_documents": len(selected_documents) if selected_documents else 0,
                "query_execution_time": query_execution_time,
                "num_retrieved_documents": len(selected_documents) if selected_documents else 0,
                "retrieval_filters": config["configurable"]["search_kwargs_dynamic"].get("filters", {}),
                "retrieval_method": config["configurable"]["search_kwargs_dynamic"].get("query_type", "unknown"),
                "batch_size": len(messages)
            })

        return result

# Define the custom PythonModel instance that will be used for inference
mlflow.models.set_model(model=LangChainModelWrapper())