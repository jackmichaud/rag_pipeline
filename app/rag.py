
import environment_variables
import os

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from file_management import get_embedding_function
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
    pass