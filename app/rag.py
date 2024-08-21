import environment_variables
import os

from langchain_core.pydantic_v1 import BaseModel, Field, conlist, ConstrainedList
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from file_management import get_embedding_function, list_uploaded_files
import json
from langchain.load import dumps
from langchain_groq import ChatGroq

def stream_rag_pipeline(question: str, collection_name: str):
    prompt = ChatPromptTemplate.from_template("""You are a chatbot teaching assistant for the class  
{expertise}. Here is the question you need to answer: {question}. 
\n\nUse the context below to develop your answer. If the context does not answer this question, say so. 
Do not overexplain. If you quote something from this context, copy it exactly without changing the 
words, and cite where you got the information from. The context chunks are ranked from most relevant 
(top) to the least relevant (bottom):
\n\n{context}
\n\nIf the context does not answer the question, please respond with "I don't know." According to the 
context, the answer to {question} is:""")

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

    parser = StrOutputParser()
    model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    chain = prompt | model | parser

    sources = [os.path.basename(doc.metadata.get("id", None)) for doc in similar]

    return {"response": chain.stream({"question": question, "context": context, "expertise": collection_name}), "sources": sources}

def stream_rag_with_routing(question: str, collection_name: str):
    # TODO Add routing
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant whose goal is to address a prompt by listing documents/resources that will help answer their question."),
        ("ai", "Which documents can I choose from?"),
        ("system", "Here are a list of all the documents you can choose from. If no documents are relevant to the question, just say \"I don't know\". Document list: \n{documents_list}"),
        ("ai", "I'm ready to help. What is the user's question?"),
        ("human", "Hello! This is my question: {question}"),
        ("ai", "I will tell you the indexes of the relevant documents. They are: "),
    ])

    print("Your question is: ", question)
    print("Your collection is: ", collection_name)

    if(collection_name == "All Indexes"):
        documents_list = json.dumps(list_uploaded_files(), indent = 4) # needs work to enumerate correctly
    else:
        documents_list = list_uploaded_files(collection_name)
        stringified_documents_list = "\n".join([f"{i}. {doc}" for i, doc in enumerate(documents_list)]) + "\n"

    print("Possible documents are: ", documents_list)

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    structured_llm = llm.with_structured_output(IndexesOfDocuments)
    parser = StrOutputParser()

    chain = prompt | structured_llm 

    indexes_of_relevant_documents = chain.invoke({"question": question, "documents_list": stringified_documents_list})

    names_of_relevant_documents = [documents_list[index] for index in indexes_of_relevant_documents.indexes]

    print("Documents that may be relevant: " + str(names_of_relevant_documents))

    # Retrieve documents with similar embedding
    retriever = Chroma(
        persist_directory="./app/chroma", 
        embedding_function=get_embedding_function()
    )

    similar = []
    for name in names_of_relevant_documents:
        similar.extend(
            retriever.similarity_search(question, k=3, filter={"source": "app/data/" + collection_name + "/" + name}) 
        )
    
    # Format chunks
    delimiter = "\n\n---\n\n"
    context = delimiter.join([dumps(doc.page_content) + "\nSource: " + 
                            doc.metadata["source"] + ", Page Number: " + 
                            str(doc.metadata["page"]) + ", Chunk ID: " + 
                            doc.metadata["id"] for doc in similar])
    
    parser = StrOutputParser()
    model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    prompt = ChatPromptTemplate.from_template("""Answer the following question using only the context below: {question}\n\nContext: {context}\n\nAnswer:""")

    chain = prompt | model | parser

    # TODO: Add ability to query more documents

    return {"response": chain.stream({"question": question, "context": context}), "sources": names_of_relevant_documents}

class IndexesOfDocuments(BaseModel):
    indexes: List[int]

class Response(BaseModel):
    response: str
    solved_problem: bool