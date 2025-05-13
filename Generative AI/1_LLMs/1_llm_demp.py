from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.9)

result = llm("What is the capital of Pakistan?")

print(result)