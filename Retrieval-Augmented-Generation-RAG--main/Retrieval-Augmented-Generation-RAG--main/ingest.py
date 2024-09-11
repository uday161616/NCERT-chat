#importing dependencies
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
import time

#loading data

loader = PyPDFDirectoryLoader('data/')
documents = loader.load()
print(len(documents))

#splitting

splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 500)
text_chunks = splitter.split_documents(documents)
print(len(text_chunks))

#loading HuggingFaceBGE embeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

print('Embeddings loaded!')

# creating NCERT Textbooks vector database.

t1 = time.time()
persist_directory = 'dbname'
vectordb = Chroma.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    collection_metadata = {"hnsw:space": "cosine"},
    persist_directory = persist_directory
)
t2 = time.time()
print('Time taken for building db : ', (t2 - t1))

