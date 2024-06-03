import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
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

def MergeColumns(df):
    # filling NaN values in fasl_name with empty strings to avoid concatenation issues
    df['fasl_name'] = df['fasl_name'].fillna('')
    df['Bab_name'] = df['Bab_name'].fillna('')

    df['Text'] = df['Bab_name'] + '\n' + df['fasl_name'] + '\n' + df['Text']
    return df

def CreateChromaVectorDatabase(text_list, embedding):
    chroma_client = chromadb.PersistentClient(path="../Dostor_embedding_VDB")
    db = chroma_client.create_collection(name='Dostor_embedding_VDB', embedding_function=embedding)

    for i, d in enumerate(text_list):
        db.add(
        documents=d,
        ids=str(i)
        )

def GetChromaVectorDatabase(embedding):
    chroma_client = chromadb.PersistentClient(path="../Dostor_embedding_VDB")
    db = chroma_client.get_collection(name="Dostor_embedding_VDB", embedding_function=embedding)
    return db

def Retriever(question, k):
    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")
    db = GetChromaVectorDatabase(embedding)
    results = db.query(query_texts=question, n_results=k, include=['distances', 'documents']) 
    context = results['documents'][0]
    return context

def MawadNumbersChat(question):

    _DEFAULT_TEMPLATE = """
    you are an Arabic expert that know how to extract meanings well, 
    Given the following question I need you to extract only the المادة و الرقم and return the number
    only in english like an example [9] and if they are multiple add to the list and
    and if الديباجة is mentioned return 0, and if the question is none related respond 991 only.
    I have only three duplicated مادة (244,241,150) so when retrieveing their numbers retrieve the number and number(1)
    as an example [244,'244(1)'] so when retrieving the duplicate return it as a string
    
    question: {input}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["input"], template=_DEFAULT_TEMPLATE
    )

    conversation = LLMChain(
            llm=ConnectToAzure(),
            prompt=prompt,
            verbose=False,
    )

    response = conversation.invoke({"input": question})
    return response

def RetrieveMawadNumbers(question):
    file_path = '../Dostoor_Egy_Structured.xlsx'
    df = pd.read_excel(file_path)
    df = MergeColumns(df)
    response = MawadNumbersChat(question)
    if(response['text'] == '991'):
        print('Mfish haga kaml 3ady')
    else:
        unadjusted_resp = response['text']
        list_part = re.search(r"\[(.*?)\]", unadjusted_resp).group(0)
        mawad_list = eval(list_part)
        filtered_df = df[df['Madda'].isin(mawad_list)]
        madda_text_list = filtered_df['Text'].tolist()
        return madda_text_list


def Conversation():

    _DEFAULT_TEMPLATE = """
    you are an Arabic Lawyer who only have knowledge in the constitution of Egypt, if you are asked outside of your profession
    ,Respond that you don't know and tell the human to ask only questions about الدستور المصري
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
    cl.user_session.set("Conversation", conversation)



@cl.on_chat_start
async def on_chat_start():
    model = ConnectToAzure()
    # set memory
    cl.user_session.set("memory", ConversationSummaryBufferMemory(return_messages=True, max_token_limit=500, llm = model))

    Conversation()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    model = ConnectToAzure()
    # set memory
    memory = ConversationSummaryBufferMemory(return_messages=True, max_token_limit=500, llm = model)

    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    Conversation()


@cl.on_message
async def main(message: str):

    memory = cl.user_session.get("memory") 
    
    Conversation = cl.user_session.get("Conversation")
    madda_text_list = RetrieveMawadNumbers(message.content)
    context = Retriever(message.content, 15)
    if madda_text_list:
        data = madda_text_list + context
    else:
        data = context
    
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