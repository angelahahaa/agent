import dotenv

dotenv.load_dotenv('.env')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
DATABASE_URL='databases/vector_db'
embd = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)

def vector_store(collection_name):
    return Chroma(
        persist_directory=DATABASE_URL,
        collection_name=collection_name,
        embedding_function=embd,
    )

def retriever(collection_name):
    return vector_store(collection_name).as_retriever()

def add(collection_name:str, path:str):
    if path.startswith('http'):
        docs = WebBaseLoader(path).load()
        docs = text_splitter.split_documents(docs)
    else:
        docs = TextLoader(path).load()
        docs = text_splitter.split_documents(docs)
        for d in docs:
            d.metadata['source'] = os.path.basename(path)
    vector_store(collection_name=collection_name).add_documents(docs)

def get_sources(collection_name):
    metadatas = vector_store(collection_name).get(include=['metadatas'])['metadatas']
    return set([m['source'] for m in metadatas])

def get_database_info(collection_name):
    return {
        "data_source":list(get_sources(collection_name))
    }

if __name__ == '__main__':
    collection_name = 'test'
    add(collection_name, r'D:\dev\ai-suite\virtuosgpt-orch\tests\resources\limei.py')
    add(collection_name, 'https://langchain-ai.github.io/langgraph/how-tos/configuration/#base')
    print(get_database_info(collection_name))