import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path="./db/chroma_persist")
collection = chroma_client.get_or_create_collection("my_own_collection", embedding_function=openai_ef)

documents = [
    {"id": "doc1", "text": "Meanwhile, the printer at work kept screaming 'paper jam' even though there was none"},
    {"id": "doc2", "text": "She accidentally signed up for a cheese-making class in Vermont"},
    {"id": "doc3", "text": "A hummingbird paused mid-flight beside a blooming cactus"},
    {"id": "doc4", "text": "The moonlit alley shimmered with puddles of yesterday’s rain"},
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

query_text = "find document related to hummingbird"
results = collection.query(query_texts=[query_text], n_results=3)

for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"For the query: {query_text}, \n Found similar document: {document} (ID : {doc_id}, Distance: {distance})")
