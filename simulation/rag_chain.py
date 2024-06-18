# RAG chain
from rag_class import RagBot

from indexing import vectorstores

rag_bot = RagBot(
    retriever=vectorstores[2],
    temperature=0.1,
    top_k=2,
    top_p=0.1,
)

def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"], "contexts": response["contexts"]}