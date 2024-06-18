import environment_variables
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema.document import Document


# Load documents
print("Loading documents...")
document_loader = PyPDFDirectoryLoader("./context")
docs = document_loader.load()
print("Loaded", len(docs), "documents")

# Set up embeddings
print("Setting up embeddings...")
embeddings = [OllamaEmbeddings(model="nomic-embed-text")]

vectorstores = []
print("Setting up Chroma with various embedding models and chunking techniques...")
for split in ["char", "recusive", "markdown"]:

    print(f"Splitting documents into {split} chunks...")
    if split == "char":
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    elif split == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    elif split == "markdown":
        text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)

    print("Calculating chunk ids...")
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

    for embedding in embeddings:
        print(f"Embedding with {embedding.model}...")

        vectorstore = Chroma(
            collection_name=f"rag-{split}-{embedding.model}-base",
            embedding_function=embedding,
            persist_directory="./chroma",
        )
        if vectorstore not in vectorstores:
            vectorstores.append(vectorstore)

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
            print(f"Adding {len(new_chunks)} new documents to the vectorstore...")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

print("Done!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

vectorstores