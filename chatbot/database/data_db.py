import asyncio
import os
from typing import Dict, Literal

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from chatbot import config

# Simulate a database or shared storage for task statuses
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
text_splitter = TokenTextSplitter(chunk_size=1024, encoding_name="cl100k_base")

vector_store = PGVector(
    embeddings=embeddings,
    connection=create_async_engine(config.CONN_STRING_PSYCOPG3),
)

class PutDataRequest(BaseModel):
    data_id: str
    path: str
    filename: str | None

class DataStatus(BaseModel):
    status: Literal['started','completed','error']
    detail: str|None = None

data_statuses: Dict[str, DataStatus] = {}
# TODO: think about what happens if this grows very big

async def batch_aadd_documents(data_id:str, loader:BaseLoader, batch_size:int, metadata:Dict[str,str]={}):
    global vector_store, text_splitter
    documents = []
    async for document in loader.alazy_load():
        document.metadata.update(metadata)
        document.metadata['data_id'] = data_id
        documents.append(document)
        if len(documents) == batch_size:
            documents = list(await text_splitter.atransform_documents(documents))
            await vector_store.aadd_documents(documents)
            documents = []
    if documents:
        documents = list(await text_splitter.atransform_documents(documents))
        await vector_store.aadd_documents(documents)

async def put_file(data_id:str, path:str, metadata:Dict[str,str]={}):
    data_statuses[data_id] = DataStatus(status='started')
    try:
        ext = os.path.splitext(path)[1]
        if ext == '.txt':
            loader = TextLoader(path, autodetect_encoding=True)
            batch_size = 1
        else:
            raise NotImplementedError()
        await batch_aadd_documents(data_id, loader, batch_size, metadata)
        data_statuses[data_id] = DataStatus(status='completed')
    except Exception as e:
        data_statuses[data_id] = DataStatus(status='error',detail=str(e))
        raise

async def put_web(data_id:str, path:str, metadata:Dict[str,str]={}):
    data_statuses[data_id] = DataStatus(status='started')
    try:
        loader = WebBaseLoader(path)
        batch_size = 1
        await batch_aadd_documents(data_id, loader, batch_size, metadata)
        data_statuses[data_id] = DataStatus(status='completed')
    except Exception as e:
        data_statuses[data_id] = DataStatus(status='error',detail=str(e))
        raise

def get_data_status(data_id) -> DataStatus | None:
    """
    Retrieve the current status of a task.
    """
    if data_id in data_statuses:
        return data_statuses[data_id]
    else:
        # TODO: also check in database incase instance restarted, the query should be this
        """
        SELECT COUNT(*)
        FROM langchain_pg_embedding
        WHERE cmetadata->>'data_id' = %s; 
        """
        pass
    return None

if __name__ == '__main__':
    import sys
    from uuid import uuid4
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    async def main():
        data_id_1 = f"data_{uuid4()}"
        data_id_2 = f"data_{uuid4()}"   
        await put_file(data_id_1, r'D:\dev\ai-suite\virtuosgpt-orch\tests\resources\limei.txt', {'source':'limei.txt'})
        await put_web(data_id_2, "https://python.langchain.com/docs/integrations/document_loaders/web_base/")
        docs = await vector_store.asimilarity_search("limei", filter={"data_id":{"$in":[data_id_1, data_id_2]}})
        print([d.metadata for d in docs])
        print(len(docs))
    asyncio.run(main())