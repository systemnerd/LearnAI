import os
from llama_index.core import (
                                VectorStoreIndex, 
                                SimpleDirectoryReader, 
                                StorageContext,
                                load_index_from_storage
                             )
from llama_parse import LlamaParse

# check if storage already exists
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./data/constitution").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

response = query_engine.query("explain the third ammendment")
print(response)

# same thing with llamaparse
documents = LlamaParse(result_type="text").load_data("./data/appointment/appointment.pdf")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("whats the photo service fee?")
response = query_engine.query("whats appointment time?")
print(response)
