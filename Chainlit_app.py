import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import chromadb
from chromadb.utils import embedding_functions

from chainlit.types import ThreadDict
import chainlit as cl

## Keys ##
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
DEPLOYMENT_URL =  os.getenv("DEPLOYMENT_URL")

def ConnectToAzure():

    model = AzureChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type=OPENAI_API_TYPE,
    )
    return model

def TextToList(df):
    text_list = df['Text'].tolist()
    return text_list

def CreateChromaVectorDatabase(text_list, embedding):
    chroma_client = chromadb.PersistentClient(path="Dostor_embedding_VDB")
    db = chroma_client.create_collection(name='Dostor_embedding_VDB', embedding_function=embedding)

    for i, d in enumerate(text_list):
        db.add(
        documents=d,
        ids=str(i)
        )

def GetChromaVectorDatabase(embedding):
    chroma_client = chromadb.PersistentClient(path="Dostor_embedding_VDB")
    db = chroma_client.get_collection(name="Dostor_embedding_VDB", embedding_function=embedding)
    return db

def Retriever(question, k):
    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")
    db = GetChromaVectorDatabase(embedding)
    results = db.query(query_texts=question, n_results=k, include=['distances', 'documents']) 
    context = results['documents'][0]
    return context

def Conversation():

    _DEFAULT_TEMPLATE = """
    you are an Arabic Lawyer who have knowledge in the constitution of Egypt, 
    Given the following context and question in Arabic find a precise Arabic answer
    . Your answer should be ONLY in Arabic and always mention the section that you get your answer from .\
    
    context: '''{context}''' \
    
    Conversation History: ''{history}'' \
    
    Current conversation:
    New human question: {input}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["input","context","history"], template=_DEFAULT_TEMPLATE
    )

    conversation = LLMChain(
        llm=ConnectToAzure(),
        prompt=prompt,
        verbose=True,
    )
    return conversation



@cl.on_chat_start
async def on_chat_start():
    model = ConnectToAzure()
    # set memory
    cl.user_session.set("memory", ConversationSummaryBufferMemory(return_messages=True, max_token_limit=500, llm = model, memory_key="history",))

    Conversation()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    model = ConnectToAzure()
    # set memory
    memory = ConversationSummaryBufferMemory(return_messages=True, max_token_limit=300, llm = model, memory_key="history", output_key='output')

    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    Conversation()

    cl.user_session.set("Conversation", Conversation())
    

@cl.on_message
async def main(message: str):

    memory = cl.user_session.get("memory") 
    
    Conversation = cl.user_session.get("Conversation")
    data = Retriever(message.content, 15)
    # Run model
    response = Conversation.invoke({"context": data,
                            "input": message.content,
                            "history": memory
                            })
    print("response ", response['text'])
    
    # Send a response back to the user
    await cl.Message(
        content = response['text'],

    ).send()

    memory.chat_memory.add_user_message(message.content)