from langchain.evaluation import load_evaluator
from langchain.evaluation.loading import load_dataset
import asyncio
from tqdm.notebook import tqdm
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.load import dumps, loads

class RagPipeline:
    def __init__(self, temperature: float, top_k: int, top_p: float, slice_method: str):
        """
        Initialize the RagPipeline class.

        Args:
            temperature (float): The temperature parameter for the LLM.
            top_k (int): The top_k parameter for the LLM.
            top_p (float): The top_p parameter for the LLM.
            slice_method (str): The slice method for the documents. Valid options are "char", "recursive", "doc", "semantic", "agentic", "routed".

        Returns:
            None
        """
        self.llm = Ollama(model="llama3", temperature=temperature, top_k=top_k, top_p=top_p)
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(embedding_function=self.embedding_model)
        
        if(slice_method == "char"):
            self.text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
        elif(slice_method == "recursive"):
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
        elif(slice_method == "doc"):
            self.text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
        elif(slice_method == "semantic"):
            # TODO: implemented semantic, agentic, routed chunker class
            self.text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
        elif(slice_method == "agentic"):
            self.text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
        elif(slice_method == "routed"):
            self.text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
 
    def query(self, query_text: str, **kwargs):
        context_type = kwargs.get('context_type', 'query')
        recursive_decomp = kwargs.get('recursive_decomp', False)
        print_debug_info = kwargs.get('print_debug_info', False)
        
        prompt = PromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:""")
        
        # Search the DB for similar text.
        results = self.vectorstore.similarity_search(query_text, k=6)

        # Format chunks
        delimiter = "\n\n---\n\n"
        context = delimiter.join([dumps(doc) for doc in results])

        # Format prompt
        prompt = prompt.format(question=query_text, context=context)

        response_text = self.llm.invoke(query_text)
        #formatted_sources = [doc.metadata.get("id", None)[5:] for doc in context]
        #formatted_response = f"Response: {response_text}\n\nSources: {formatted_sources}"

        return StrOutputParser().parse(response_text)

    
    def load_documents(self, documents: list[Document]):
        chunks = self.text_splitter.split_documents(documents)
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = self.vectorstore.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")
    

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

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
    return chunks