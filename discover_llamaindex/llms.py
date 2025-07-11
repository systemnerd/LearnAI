from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import Prompt

with open("./docs/starter_example.md", "r") as f:
    text = f.read()

llm = OpenAI(model="gpt-4o-mini", temperature=0)

# defining prompts
text_qa_template = Prompt(
    "Context information is below. \n"
    "------------------------------\n"
    "{context_str}\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_template = Prompt(
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below. \n"
    "----------------------------------------------\n"
    "{context_msg}\n"
    "----------------------------------------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question : {query_str}."
    "If the context is not useful, output the original answer again. \n"
    "Original Answer: {existing_answer}"
)


question = "How to load an index from disk to memory?"
prompt = text_qa_template.format(context_str=text, query_str=question)
response = llm.complete(prompt)
print(response.text)

question = "How do I create an index? Write your answert using only code."
existing_answer = """
To create an index using LlamaIndex, you need to follow these steps:

1. Download the LlamaIndex repository by cloning it from GitHub.
2. Navigate to the `examples/paul_graham_essay` folder in the cloned repository.
3. Create a new Python file and import the necessary modules: `VectorStoreIndex` and `SimpleDirectoryReader`.
4. Load the documents from the `data` folder using `SimpleDirectoryReader('data').load_data()`.
5. Build the index using `VectorStoreIndex.from_documents(documents)`.
6. To persist the index to disk, use `index.storage_context.persist()`.
7. To reload the index from disk, use the `StorageContext` and `load_index_from_storage` functions.

Note: This answer assumes that you have already installed LlamaIndex and have the necessary dependencies.
"""
prompt = refine_template.format(context_msg=text, query_str=question, existing_answer=existing_answer)
response_gen = llm.stream_complete(prompt)

for response in response_gen:
    print(response.delta, end="")
