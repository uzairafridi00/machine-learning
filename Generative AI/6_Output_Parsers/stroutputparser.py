from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

# 1st Prompt -> Detailed Report
prompt1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

# 2nd Prompt -> Summary
prompt2 = PromptTemplate(
    template="Summarize the following report: /n {report}",
    input_variables=["report"],
)

# String Output Parser
parser = StrOutputParser()

# creating chains
chain = prompt1 | model | parser | model | prompt2 | parser

# running the chain
result = chain.invoke({"topic": "Artificial Intelligence"})

# printing the result
print(result)