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

documents = TextLoader("./doc/dream.txt").load()

#clean the docs
cleaned_documents = [clean_text(document.page_content) for document in documents]

print(cleaned_documents)
