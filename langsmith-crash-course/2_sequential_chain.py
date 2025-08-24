from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environment['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
model2 = ChatOpenAI(model='gpt-4o', temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags': ['llm app', 'generation', 'summarization'],
    'metadata': {
        'author': 'Uzair Afridi',
        'model1': 'gpt-4o-mini',
        'model2': 'gpt-4o',
        'parser': 'StrOutputParser'
    }
}

result = chain.invoke({'topic': 'Unemployment in Pakistan'}, config=config)

print(result)
