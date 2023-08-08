import os

from langchain import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS


class LangChainLLM(object):
    def __init__(self, my_template):
        load_dotenv()
        self.prompt = PromptTemplate(
            input_variables=["content1", "content2"],
            template=my_template
        )
        try:
            self.llm = OpenAI(model_name=os.getenv("MODEL_NAME"), n=2, best_of=2,
                              openai_api_key=os.getenv("OPENAI_API_KEY"))
        except ConnectionError:
            return "The connection error occur"

    def request(self, message1, message2):
        print(f"---------------------------------------------")
        final_prompt = self.prompt.format(content1=message1, content2=message2, )
        print(f"LLM final_prompt: {final_prompt}")
        try:
            return self.llm(final_prompt)
        except ConnectionError:
            return "The connection error occur"


if __name__ == '__main__':
    template = """ 
        I am travelling to {content1}. What are the top {content2}. things I can do while I am there. Be very specific and respond as three bullet points 
        """
    LangChainLLM = LangChainLLM(template)
    print(LangChainLLM.request("beijing", "4"))
