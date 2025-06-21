import os
import chromadb

from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.PersistentClient(path="./db/chroma_persistant_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

def load_documents_from_directory(directory_name:str):
    print("===Loading documents from directory===")
    documents = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_name, filename)) as file:
                documents.append({
                    "id": filename,
                    "text": file.read()
                })
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./data/new_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

# split docs into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("===splitting docs into chunks===")
    for idx, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_chunk{idx+1}",
            "text": chunk
        })

def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("===Generating embedding===")
    return embedding

for doc in chunked_documents:
    print("=== Generating embeddings.. ===")
    doc["embedding"] = get_openai_embedding(doc["text"])

for doc in chunked_documents:
    print("=== Inserting chunks into db ===")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

