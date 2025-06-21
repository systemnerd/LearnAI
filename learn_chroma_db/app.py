import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

# define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=doc["text"])

# define a query text
query = "random query"

results = collection.query(
    query_texts=[query],
    n_results=3,
)

for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"For the query: {query}, \n Found similar document: {document} (ID : {doc_id}, Distance: {distance})")
