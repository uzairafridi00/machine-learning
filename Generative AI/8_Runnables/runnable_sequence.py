from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

# define prompt 1
prompt1 = PromptTemplate(
    template = 'Write a Jokee about {topic}',
    input_variables = ['topic']
)

# call the model
model = ChatOpenAI()

# define output parser
parser = StrOutputParser()

# define prompt 2
prompt2 = PromptTemplate(
    template = 'Explain the following joke: {text}',
    input_variables = ['text']
)

# define the chain using runnable sequence
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# run the chain
# and print the result
print(chain.invoke({'topic': 'Python'}))