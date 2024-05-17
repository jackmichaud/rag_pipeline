# Useful Sources: https://www.youtube.com/watch?v=2TJxpyO3ei4


## LOADING THE DOCUMENTS FROM A DIRECTORY ##

from langchain.document_loaders import PyPDFDirectoryLoader

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

## SPLIT THE DOCUMENTS INTO CHUNKS ##

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(documents)

## RETURN THE EMBEDDING MODEL ##

from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name = "default",
        region_name = "us-east-1"
    )
    return embeddings



documents = load_documents()
chunks = split_documents(documents)

