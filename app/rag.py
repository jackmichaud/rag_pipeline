from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from get_embedding_function import get_embedding_function
from langchain.load import dumps
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

def update_vectorstore_collection(collection_name: str):
    # Load documents in a given colleciton
    document_loader = PyPDFDirectoryLoader(os.path.join("app", "data", collection_name))
    docs = document_loader.load()
    print("Loaded", len(docs), "documents")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        is_separator_regex = False
    )
    chunks = text_splitter.split_documents(docs)
    print("Documents split into " + str(len(chunks)) + " chunks")

    # Calculate chunk ids
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Find the index of the last slash
        last_slash_index = source.rfind('/')
        # Slice the string up to the last slash
        filter = source[:last_slash_index]

        # Create metadata that will be used for filtering during retrieval
        chunk.metadata["filter"] = filter

    vectorstore = Chroma(
        persist_directory="./app/chroma", 
        embedding_function=get_embedding_function()
    )

    # Add or Update the documents.
    existing_items = vectorstore.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
        print(new_chunks)
    else:
        print("✅ No new documents to add")

    

def stream_rag_pipeline(question: str, collection_name: str):
    prompt = ChatPromptTemplate.from_template("""You are a chatbot that is an expert on 
{expertise}. Here is the question you need to answer: {question}. 
\n\nUse the context below to develop your answer. If the context does not answer this question, say so. 
Do not overexplain. If you quote something from this context, copy it exactly without changing the 
words, and cite where you got the information from. The context chunks are ranked from 
most relevant (top) to the least relevant (bottom):
\n\n{context}
\n\nAccording to the context, the answer to {question} is:""")

    # Retrieve documents with similar embedding
    retriever = Chroma(
        persist_directory="./app/chroma", 
        embedding_function=get_embedding_function()
    )
    if(collection_name == "All Indexes"):
        filter = None
    else:
        filter = dict(filter = "app/data/" + collection_name)
    similar = retriever.similarity_search(question, k=6 , filter=filter)

    # Format chunks
    delimiter = "\n\n---\n\n"
    context = delimiter.join([dumps(doc.page_content) + "\nSource: " + 
                            doc.metadata["source"] + ", Page Number: " + 
                            str(doc.metadata["page"]) + ", Chunk ID: " + 
                            doc.metadata["id"] for doc in similar])

    print(context)

    parser = StrOutputParser()
    model = Ollama(model="llama3", temperature="0")

    chain = prompt | model | parser

    sources = [os.path.basename(doc.metadata.get("id", None)) for doc in similar]

    return {"response": chain.stream({"question": question, "context": context, "expertise": collection_name}), "sources": sources}

        