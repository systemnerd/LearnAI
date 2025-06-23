from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# define a prompt template
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# create a chat model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({
    "topic": "cocktaeil bird",
})

print(response)
