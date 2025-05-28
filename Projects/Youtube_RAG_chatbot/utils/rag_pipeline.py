from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek

from .config import OPENAI_API_KEY, CHROMA_PATH
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_llm(model_name: str, api_key: str):
    if model_name == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(temperature=0)
    elif model_name == "DeepSeek":
        os.environ["DEEPSEEK_API_KEY"] = api_key
        return ChatDeepSeek(api_key=api_key)
    elif model_name == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(api_key=api_key)
    else:
        raise ValueError("Unsupported model")

def process_and_store_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([transcript])

    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=OpenAIEmbeddings(),
                                     persist_directory=CHROMA_PATH)
    vectordb.persist()
    return True


def query_transcript(question: str, model_name: str, api_key: str) -> str:
    vectordb = Chroma(persist_directory=CHROMA_PATH,
                      embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    llm = get_llm(model_name, api_key)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa.run(question)