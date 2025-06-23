from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

# directory loader
dir_loader = DirectoryLoader("./data/", glob="**/*.txt")
dir_documents = dir_loader.load()

print("Directory documents", dir_documents)
