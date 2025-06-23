from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pprint
import re

load_dotenv()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

# Loading
documents = TextLoader("./doc/dream.txt").load()

# chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# clean the docs
texts = [clean_text(text.page_content) for text in texts]
print(texts)
