import os
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import uuid

load_dotenv()

#constants
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

class SimplePDFProcessor:
    """
        handles PDF processing and chunking
    """
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """
            Read PDF and extract text
        """
        reader = PyPDF2.PdfReader(pdf_file)
        text=""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        """
            Split text into chunks
        """
        chunks=[]
        start = 0

        while start < len(text):
            # find end of chunk
            end = start + self.chunk_size

            if start > 0:
                start = start - self.chunk_overlap
            
            # Get chunk
            chunk = text[start:end]

            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(
                {
                    "id":str(uuid.uuid4()),
                    "text":chunk,
                    "metadata": {"source": pdf_file.name},
                }
            )

            start = end
        
        return chunks

class SimpleRAGSystem:
    """
        simple RAG implementation
    """
    def __init__(self, embedding_model="openai", llm_model="openai") -> None:
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.db = chromadb.PersistentClient("./chroma_db")
        self.setup_embedding_function()
        self.setup_llm()
        self.setup_collection()

    def setup_embedding_function(self):
        """
            setup the appropriate embedding function
        """
        try:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small",
            )
        except Exception as e:
            print(f"Error setting up embedding function {str(e)}")

    def setup_llm(self):
        """
            Setup the LLM
        """
        try:
            self.llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        except Exception as e:
            print(f"error setting up LLM, {str(e)}")

    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        try:
            self.collection = self.db.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"model": self.embedding_model},
            )
        except Exception as e:
            print(f"error in setting up collection")
        
    def add_documents(self, chunks):
        if not self.collection:
            self.collection = self.setup_collection()

        # Add documents
        self.collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        return True
    
    def query_documents(self, query, n_results=3):
        """
        Query documents and return relevant chunks
        """
        if not self.collection:
            raise ValueError("No collection available")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def generate_response(self, query, context)->str|None:
        """
        Generate response using LLM
        """
        try:
            prompt=f"""
            Based on the following context, please answer the question.
            If you cant find the answer in the context, say so.

            Context: {context}

            Question: {query}

            Answer:
            """

            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system", "content":"You are a helpful assistant."},
                    {"role":"user", "content":prompt},
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
        
def main():
    st.title("Simple RAG system")

    if "rag_system" not in st.session_state:
        st.session_state["rag_system"] = None
    
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    

    try:
        if st.session_state["rag_system"] is None:
            st.session_state["rag_system"] = SimpleRAGSystem()
    except Exception as e:
        st.error(f"error initializing RAG system: {str(e)}")
        return

    pdf_file = st.file_uploader("Upload pdf file", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                # extract text
                text = processor.read_pdf(pdf_file)

                # create chunks
                chunks = processor.create_chunks(text, pdf_file)

                # Add to database
                if st.session_state.rag_system.add_documents(
                    chunks=chunks
                ):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("Query your documents")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response .... "):
                # Get relevant chunks
                results = st.session_state.rag_system.query_documents(query)
                if results and results["documents"]:
                    response = st.session_state.rag_system.generate_response(
                        query,
                        results["documents"][0],
                    )

                    if response:
                        st.markdown("###Answer: ")
                        st.write(response)

                        with st.expander("View sources"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"***Passage {idx}***")
                                st.info(doc)
    else:
        st.info("Please upload a PDF document to get started!")


if __name__=="__main__":
    main()
    