from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Afridi is a Pakistani cricketer known for his aggressive batting and leadership.",
    "Babar Azam is a former Pakistani captain famous for his calm demeanor and finishing skills.",
    "Waseem Akram, also known as the 'God of Swing', holds many wicket records.",
    "Inzamam is known for his elegant batting and record-breaking double centuries.",
    "Waqar Younus is a Pakistani fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Waseem Akram'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)