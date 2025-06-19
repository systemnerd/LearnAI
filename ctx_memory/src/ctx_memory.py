from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import json
import os

load_dotenv()

def initialize_client(use_ollama=False):
    if use_ollama:
        pass
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_initialize_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant"},
    ]

def chat(
        user_input: str, messages: List[Dict[str, str]], client: OpenAI,
        model_name: str
) -> str:
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        return assistant_response
    except Exception as e:
        return f"Error with API: {str(e)}"

def summarize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    summary = "Previous conversation summarized: " + " ".join(
        [m["content"][:50] + "..." for m in messages[-5:]]
    )

    return [{"role":"system", "content":summary}] + messages[-5:]

def save_conversation(messages: List[Dict[str, str]], filename: str="conversation.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

def load_conversation(filename:str="conversation.json") -> List[Dict[str, str]]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No conversation found at {filename}")
        return create_initialize_messages()

def main():
    client = initialize_client()
    model_name = "gpt-4o-mini"

    # load conversation
    messages = create_initialize_messages()

    print("\nAvailable commands:")
    print("- 'save' : Save conversation")
    print(" - 'load' : Load conversation")
    print(" - 'summary' : Summarize conversation")

    while True:
        user_input = input("\n input your choice: ")

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'save':
            save_conversation(messages)
            print("Conversation saveed")
            continue
        elif user_input.lower() == 'load':
            load_conversation()
            print("conversation loaded")
            continue
        elif user_input.lower() == 'summary':
            summarize_messages(messages)    
            print("conversation summarized!")
            continue

        response = chat(user_input, messages, client, model_name)
        print(f"\n Assistant: {response}")

        if len(messages) > 10:
            messages = summarize_messages(messages)
            print("\n conversation automatically summarized")


if __name__=="__main__":
    main()
