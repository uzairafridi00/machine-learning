import streamlit as st
from utils.youtube_utils import fetch_transcript
from utils.rag_pipeline import process_and_store_transcript, query_transcript

st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎥 YouTube RAG Chatbot")

# Add model selector
model = st.selectbox("Select LLM Provider", ["OpenAI", "DeepSeek", "Groq"])

# Dynamically show relevant API key input
api_key = st.text_input(f"{model} API Key", type="password")

youtube_url = st.text_input("📺 YouTube Video URL")

if st.button("Load Transcript"):
    if not api_key or not youtube_url:
        st.warning("Please provide both API key and video URL")
    else:
        with st.spinner("Fetching transcript and processing..."):
            transcript = fetch_transcript(youtube_url)
            process_and_store_transcript(transcript)
        st.success("Transcript processed and stored in vector DB!")

st.markdown("---")

question = st.text_area("❓ Ask a question about the video")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Getting answer..."):
            answer = query_transcript(question, model, api_key)
        st.success("Answer:")
        st.write(answer)
