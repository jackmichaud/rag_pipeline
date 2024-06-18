### RAG

from langsmith import traceable
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.load import dumps, loads

class RagBot:
    
    def __init__(self, retriever, temperature: float, top_k: int, top_p: float, model: str = "llama3"):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        self._client = Ollama(model=model, temperature=temperature, top_k=top_k, top_p=top_p)
        self._model = model
        self._expertise = "the board game Settlers of Catan"

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.similarity_search(question, k=6)

    @traceable()
    def get_answer(self, question: str):
        prompt = ChatPromptTemplate.from_template("""You are a chatbot that is an expert on 
{expertise}. Here is the question you need to answer: {question}. 
\n\nUse the context below to develop your answer. If the context does not answer this question, say so. 
Do not overexplain. If you quote something from this context, copy it exactly without changing the 
words, and cite where you got the information from. The context chunks are ranked from 
most relevant (top) to the least relevant (bottom):
\n\n{context}
\n\nAccording to the context, the answer to {question} is:""")
        
        # Retrieve documents with hyde
        hypothetical_doc = self.get_hypothetical_doc(question)
        similar = self.retrieve_docs(hypothetical_doc)

        # Format chunks
        delimiter = "\n\n---\n\n"
        context = delimiter.join([dumps(doc.page_content) + "\nSource: " + 
                                doc.metadata["source"] + ", Page Number: " + 
                                str(doc.metadata["page"]) + ", Chunk ID: " + 
                                doc.metadata["id"] for doc in similar])

        # Format prompt
        prompt = prompt.format(question=question, context=context, expertise=self._expertise)

        response_text = StrOutputParser().parse(self._client.invoke(prompt))

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": response_text,
            "contexts": context,
        }

    @traceable()
    def get_hypothetical_doc(self, question: str):
        hyde_template = """You are a chatbot that is an expert on {expertise}. Write one short 
hypothetical answer to the following question about: {question}. \nAnswer:"""
        hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
        prompt_string = hyde_prompt.format(question=question, expertise=self._expertise)

        hyde_chain = ( self._client | StrOutputParser() )
        hypothetical_doc = hyde_chain.invoke(prompt_string)
        return hypothetical_doc
    

