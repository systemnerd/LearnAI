from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
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
# print(texts)

# Load the OpenAI embeddings to vectorize the text
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# create the retriever from the loaded embeddings and documents
retriever = FAISS.from_texts(texts, embeddings).as_retriever(
    search_kwargs={"k": 2}
)

# query the retriever
query="what was the dream about?"
docs = retriever.invoke(query)

pprint.pprint(f" => DOCS : {docs}:")

# Chat with the model and our docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# create a chat prompt
prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs}, and answer the following question {query}"
)

# create a model
model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | StrOutputParser()

response = chain.invoke({
    "docs":docs,
    "query": query
})

print(f"Model response :: {response}")
