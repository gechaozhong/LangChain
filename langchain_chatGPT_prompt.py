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

    llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2, openai_api_key=os.getenv("OPENAI_API_KEY"))
    print(llm("Tell me a joke about data scientist"))
    template = """ 
        I am travelling to {location}. What are the top 3 things I can do while I am there. Be very specific and respond as three bullet points 
        """
    prompt = PromptTemplate(
        input_variables=["location"],
        template=template,
    )

    final_prompt = prompt.format(location="test")

    print(f"LLM Output: {llm(final_prompt)}")
