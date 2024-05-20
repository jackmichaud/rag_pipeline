import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the following question based on only this context. If there is not enough context to answer the question or is irrelevant,
then say that you do not know the answer:

{context}

---

Answer this question based on the above context: {question}
"""

MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

# Prepare the DB.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Generate documents using multi-query
    context_text = generate_multi_query(query_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the final llm call
    model = Ollama(model="llama2")
    response_text = model.invoke(prompt)

    # Format the repsonse
    sources = [doc.metadata.get("id", None) for doc in context_text]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    #print(formatted_response)
    return formatted_response

# Get unique union of documents
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    # Get unique documents
    unique_docs = list(set(flattened_docs))

    # Remove float elements that end up in list. TODO: Find out why there are floats and prevent them
    raw_union = [loads(doc) for doc in unique_docs]
    no_floats = [x for x in raw_union if not isinstance(x, float)]

    # Return
    return no_floats

def generate_multi_query(query_text: str):
    # Generate a list of rephrased questions
    generate_queries = (
        ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE) 
        | Ollama(model="llama2")
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    # Retrieve and return documents from multi-query
    retrieval_chain = generate_queries | retrieve_documents | get_unique_union
    docs = retrieval_chain.invoke({"question":query_text})
    len(docs)
    return docs

def retrieve_documents(query_text: str):
    # Search the DB for similar text.
    results = db.similarity_search_with_score(query_text, k=5)
    return results


if __name__ == "__main__":
    main()