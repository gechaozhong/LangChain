import argparse
import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DocugamiLoader
from langchain.llms import OpenAI
from dotenv import load_dotenv

from constants import CHROMA_SETTINGS

load_dotenv()
if __name__ == '__main__':
    llm = OpenAI(model_name="text-ada-001", n=2, best_of=2, openai_api_key=os.getenv("OPENAI_API_KEY"))
    print(llm("Tell me a joke about data scientist"))
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
    已知内容:
    {context}
    问题:
    {question}"""

