from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Islamabad is the capital of India",
    "Tokyo is the capital of Japan",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))