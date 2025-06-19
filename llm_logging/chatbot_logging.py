from openai import OpenAI
import logging
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv
import os

load_dotenv()
def setup_logging():
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("chatbot_logs.json")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    return logger

def initialize_client(use_ollama: bool = False):
    if use_ollama:
        pass
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatBot:
    def __init__(self, use_ollama: bool = False):
        self.logger = setup_logging()
        self.session_id = str(uuid.uuid4())
        self.client = initialize_client()
        self.use_ollama = use_ollama
        self.model_name = "gpt-4o-mini"

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful customer support assistant."
            }
        ]

    def chat(self, user_input: str) -> str:
        log_entry = {
            "timestamp":datetime.now().isoformat(),
            "level":"INFO",
            "type": "user_input",
            "user_input": user_input,
            "metadata": {"session_id": self.session_id, "model": self.model_name}
        }
        self.logger.info(json.dumps(log_entry))

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        start_time = datetime.now()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()

        assistant_response = response.choices[0].message.content

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level":"INFO",
            "type":"model_response",
            "model_response":assistant_response,
            "metadata": {
                "session_id": self.session_id,
                "model": self.model_name,
                "response_time_seconds": response_time,
                "tokens_used": (
                    response.usage.total_tokens
                    if hasattr(response, "usage")
                    else None
                ),
            },
        }

        self.logger.info(json.dumps(log_entry))
        self.messages.append({
            "role": "assistant",
            "content": assistant_response
        })



