import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the following question based on this context. If the context does not answer the question, say so. Do not overexplain. 
If you quote something from this context, copy it exactly without changing the words, and cite where you got the information
from. The context chunks are ranked from most relevant (top) to the least relevant (bottom):

{context}

---

Answer this question based on the above context: {question}
"""

MULTI_QUERY_TEMPLATE = """Genereate a list of a few different ways this question can be rephrased. If there are few/no ways
to rephrase the question without changing its meaning, that is ok. Do not deviate far from the original question. Separate 
each rephrased question by newlines. Do not respond with anything else except for the listof rephrased questions. 
Include the original question at the top of the list. Original question: {question}"""

# Prepare the DB.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

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

    # Format chunks and prompt
    delimiter = "\n\n---\n\n"
    context_stringified = delimiter.join([dumps(doc) for doc in context_text])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_stringified, question=query_text)

    # Invoke the final llm call
    model = Ollama(model="llama2", top_p="0.6")
    response_text = model.invoke(prompt)

    # Format the repsonse
    sources = [doc[0].metadata.get("id", None)[5:] for doc in context_text]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return formatted_response

# Get unique union of documents
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Take a list of lists and rank the documents using the RRF formula
def reciprocal_rank_fusion(documents: list[list]):
    # Optional value (I'm not exactly sure what this does :))
    k = 60

    # Initialize dictionary to hold scores for each document
    fused_scores = {}

    # Iterate through each list of documents
    for docs in documents:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)

            # Add new entry to fused_scores if document is not in it already
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            # Retrieve previous score and update fused scores
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    # Rerank the document chunks based on similarity
    reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse = True)]

    # Return the top 5 ranked documents in the list
    return reranked_results[:8]


def generate_multi_query(query_text: str):
    # Generate a list of rephrased questions
    generate_queries = (
        ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE) 
        | Ollama(model="llama2")
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    # Retrieve and return documents from multi-query
    retrieval_chain = generate_queries | retrieve_documents | reciprocal_rank_fusion

    docs = retrieval_chain.invoke({"question":query_text})
    return docs

def retrieve_documents(query_list: list[str]):
    # Search the DB for similar text.
    results = []
    for query in query_list:
        results.append(db.similarity_search(query, k=6))
    return results

if __name__ == "__main__":
    main()