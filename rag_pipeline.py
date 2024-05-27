import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter

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

QUERY_DECOMPOSITION_TEMPLATE = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 or fewer queries):"""

PROMPT_WITH_QA_TEMPLATE = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

# Prepare the DB.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

def query_rag(query_text: str):
    # Generate documents using multi-query
    context_text = generate_multi_query(query_text)

    # Generate sub-questions and generate qa pairs
    sub_queries = decompose_query(query_text)
    print(sub_queries)
    qa_pairs = generate_qa_pairs(sub_queries)

    # Format chunks
    delimiter = "\n\n---\n\n"
    context_stringified = delimiter.join([dumps(doc) for doc in context_text])
    
    # Format prompt
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)   # Comment out for query decomp
    # prompt = prompt_template.format(context=context_stringified, question=query_text) # Comment out for query decomp
    prompt_template = ChatPromptTemplate.from_template(PROMPT_WITH_QA_TEMPLATE)
    prompt = prompt_template.format(context=context_stringified, question=query_text, q_a_pairs=qa_pairs)

    # Invoke the final llm call
    model = Ollama(model="llama2", temperature="0")
    response_text = model.invoke(prompt)

    # Format the repsonse
    sources = [doc[0].metadata.get("id", None)[5:] for doc in context_text]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return formatted_response

# Get unique union of documents
def get_unique_union(documents: list[list]):
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
        results.append(db.similarity_search(query, k=4))
    return results

# Generate a list of sub-queries that break the problem into smaller problems
def decompose_query(query_text):
    prompt_decomposition = ChatPromptTemplate.from_template(QUERY_DECOMPOSITION_TEMPLATE)
    llm = Ollama(model="llama2", temperature="0")

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition.invoke({"question":query_text})

    return questions

# Helper function used for formatting
def format_qa_pair(question, answer):
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def generate_qa_pairs(questions: list):
    decomposition_prompt = ChatPromptTemplate.from_template(PROMPT_WITH_QA_TEMPLATE)
    llm = Ollama(model="llama2", temperature="0")

    # Generate qa pairs
    q_a_pairs = ""
    for q in questions:
        # Chain
        rag_chain = (
        {"context": itemgetter("question") | db.as_retriever(), 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        # Run
        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})

        # Format response
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    return q_a_pairs

