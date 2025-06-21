from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()

name = "my name is sample"
collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

emb = default_ef(name)
print(emb)
