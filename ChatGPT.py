

import os


import openai
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# list models
models = openai.Model.list()

# print the first model's id
print(models.data[0].id)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "你知道毛泽东吗"}])

# print the chat completion
print(chat_completion.choices[0].message.content)