from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("./doc/linux-manual.pdf")

docs = pdf_loader.load()
print("PDF Documents:", docs)
