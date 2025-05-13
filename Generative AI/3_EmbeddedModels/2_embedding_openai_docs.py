from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Islamabad is the capital of India",
    "Tokyo is the capital of Japan",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))