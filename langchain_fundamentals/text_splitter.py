from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your document
text_loader = TextLoader("./doc/dream.txt")
documents = text_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

# split the docs into chunks
splits = text_splitter.split_documents(documents)

for idx, split in enumerate(splits):
    print(f"Split {idx + 1} {split.page_content}\n")
