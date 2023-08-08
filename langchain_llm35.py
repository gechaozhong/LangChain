import os

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv

from constants import CHROMA_SETTINGS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class LangChainLLM35(object):
    def __init__(self, my_template):
        load_dotenv()
        #system_message_prompt = SystemMessagePromptTemplate.from_template(my_template)
        #human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(my_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        try:
            self.llm = ChatOpenAI(temperature=0, model_name=os.getenv("MODEL_NAME_3.5"), openai_api_key=os.getenv("OPENAI_API_KEY"))
            print(f"LLM final_prompt: {chat_prompt}")
            self.chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        except ConnectionError:
            return "The connection error occur"

    def request(self, message):
        print(f"---------------------------------------------")
        try:
            return self.chain.run(input_language="Chinese", output_language="Chinese", text=message)
        except ConnectionError:
            return "The chain.run function execute failed."


if __name__ == '__main__':
    template = "{text}何许人也."
    LangChainLLM = LangChainLLM35(template)
    print(LangChainLLM.request("周瑜"))
