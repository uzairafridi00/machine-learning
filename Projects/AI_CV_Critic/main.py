import streamlit as st
import PyPDF2
import io
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Summarizer", page_icon="📚", layout="centered")

st.title("AI Resume Critiquer")
st.markdown(
    "Upload your resume in PDF format, and the AI will provide feedback on how to improve it."
)

# setting up API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# upload file
upload_file = st.file_uploader("Upload your resume (PDF or TXT format)", type=["pdf", "txt"])

# define the job roles
job_role = st.text_input(
    "Enter the job role you are applying for (e.g., Software Engineer, Data Scientist):",
    placeholder="Software Engineer",
)

# check if file is uploaded

analyze = st.button("Analyze Resume")

# utilities function to read files

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):

    # check if the file is a PDF
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    
    # if its not pdf then assume it's a text file
    return uploaded_file.read().decode("utf-8")

if analyze and upload_file:
    try:
        file_content = extract_text_from_file(upload_file)

        if not file_content.strip():
            st.error("The uploaded file is empty or could not be read.")
            st.stop()
        
        prompt = f"""" Please analyze this resume and provide constructive feedback.
        Focus on the following aspects:
        1. Content Clarity and Impact.
        2. Skills Presentation.
        3. Experience description.
        4. Specific improvements for the job role: {job_role if job_role else "General Job application"}

        Resume Content:
        {file_content}

        Please provider your analysis in a clear, structured format with specific recommendations.
        """

        # client = OpenAI(api_key=OPENAI_API_KEY)
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"roles": "system", "content": "You are an expert resume reviewser with years of experience in HR and recruitment."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.7,
        #     max_tokens=1000
        # )

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewser with years of experience in HR and recruitment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=1024
        )

        st.markdown("### AI Feedback:")
        st.write(response.choices[0].message.content.strip())
    
    except Exception as e:
        st.error(f"An Error occured: {e}")
        st.stop()
