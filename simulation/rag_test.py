# RAG chain
from rag_class import RagBot

from indexing import vectorstores

rag_bot = RagBot(
    retriever=vectorstores[2],
    temperature=0.1,
    top_k=2,
    top_p=0.1,
)

print(rag_bot.get_answer("How many victory points is a city worth?")["answer"])