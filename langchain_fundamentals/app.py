from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")

response = model.invoke("Hello, how are you?")
print(response)

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
print("========")
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
print("=====")
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)
