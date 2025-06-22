import csv
import chromadb
import os
from openai import OpenAI
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingModel:
    def __init__(self, model_type="openai") -> None:
        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text"
            )

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0, # very deterministic
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response : {str(e)}"

def select_models():
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4 model")
    print("2. Ollama 3.2")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    print("\nSelect Embedding Model:")
    print("1. Open AI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed text (ollama)")

    while True:
        choice = input("Enter choice 1, 2 or 3: ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1" : "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter 1, 2 or 3")
    
    return (llm_type, embedding_type)

def generate_csv():
    facts = [
        {"id": "1", "fact": "Space is completely silent because there is no atmosphere to carry sound."},
        {"id": "2", "fact": "The Sun accounts for 99.86% of the mass in our solar system."},
        {"id": "3", "fact": "One day on Venus is longer than its year."},
        {"id": "4", "fact": "Neutron stars can spin up to 700 times per second."},
        {"id": "5", "fact": "A full NASA space suit costs about $12 million."},
        {"id": "6", "fact": "There are more stars in the universe than grains of sand on Earth."},
        {"id": "7", "fact": "Jupiter has the shortest day of all planets—just under 10 hours."},
        {"id": "8", "fact": "The Moon is slowly drifting away from Earth, about 3.8 cm per year."},
        {"id": "9", "fact": "Saturn's rings are mostly made of ice particles."},
        {"id": "10", "fact": "In space, astronauts can grow up to 2 inches taller temporarily due to spinal decompression."}
    ]

    with open("space_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)
    
    print("CSV file 'space_facts.csv' created successfully")

def load_csv():
    df = pd.read_csv("space_facts.csv")
    documents = df["fact"].tolist()
    print("\nDocuments loaded:")
    for doc in documents:
        print(f"- {doc}")
    return documents

def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("space_facts")
    except:
        pass

    collection = client.create_collection(
        name="space_facts",
        embedding_function=embedding_model.embedding_fn
    )

    collection.add(
        documents=documents,
        ids = [str(i) for i in range(len(documents))]
    )
    
    print("\nDocuments added to chromaDB collection successfully")
    return collection

def find_related_chunks(query, collection, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    print("\nRelated chunks found")
    for doc in results["documents"][0]:
        print(f"- {doc}")

    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            )
        )
    )

def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augment_prompt = f"Context: \n{context}\n\nQuestion: {query}\nAnswer:"

    print("\nAugmented prompt:")
    print(augment_prompt)

    return augment_prompt

def rag_pipeline(query, collection, llm_model:LLMModel, top_k=2):
    print(f"\nProcessing query: {query}")

    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role":"system",
                "content": "You are a helpful assistant who can answer questions about space but only answer questions that are source documents given."
            },
            {
                "role":"user",
                "content":augmented_prompt
            }
        ]
    )

    print("\nGenerated response:")
    print(response)

    references = [chunk[0] for chunk in related_chunks]
    return response, references

def main():
    print("\nStarting the RAG pipeline demo...")

    # select models
    llm_type, embedding_type = select_models()

    # intialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"\nUsing Embeddings: {embedding_type.upper()}")

    generate_csv()
    documents = load_csv()

    collection = setup_chromadb(documents, embedding_model)

    queries = [
        "what causes astronauts to grow? Just answer in one or two words.",
    ]

    for query in queries:
        print("\n" + "=" * 15)
        print(f"Processing query : {query}")
        response, references = rag_pipeline(query,collection,llm_model)

        print("\nFinal Results:")
        print("-"*10)
        print("Response:", response)
        print("\nReferences used:")

        for reference in references:
            print(f"- {reference}")
        print("="*15)

if __name__=="__main__":
    main()
