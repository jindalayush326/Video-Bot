import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
import numpy as np
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Video Interview Question Generator")

# Initialize session state for questions and upload count
if 'questions' not in st.session_state:
    st.session_state.questions = []

# Function to process video and extract questions
def process_video(uploaded_file):
    # Save  file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(uploaded_file.read()) 
        temp_video_path = temp.name 
    
    # Extract audio from video
    video = mp.VideoFileClip(temp_video_path)
    audio_file = "audio.wav"
    video.audio.write_audiofile(audio_file)

    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            transcribed_text = recognizer.recognize_google(audio_data)
        except (sr.UnknownValueError, sr.RequestError):
            st.error("Could not process audio.")
            return

    # Create embeddings from the transcribed text using the Gemini model
    if transcribed_text:
        text_documents = [{"page_content": transcribed_text}]
        
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts([transcribed_text], embedding=embeddings_model)
        vector_store.save_local("faiss_index")

        prompt_template = """
        You are an interviewer. Based on the context below, ask a relevant question from the text.
        
        Context:
        {context}
        
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        llm_model = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
        chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=prompt)

        # Ask a question based on the context
        question = "Can you summarize the main points discussed in the text?"
        context = vector_store.similarity_search(question, k=1)

        # Generate response and store the question
        answer = chain.run(input_documents=context, question=question)
        st.session_state.questions.append(answer)
        if context:
            answer = chain.run(input_documents=context, question=question)
            st.session_state.questions.append(answer)

uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    process_video(uploaded_file)

    if st.session_state.questions:
        st.write(f"Generated Question: ", st.session_state.questions[-1])
    else:
        st.write("No question was generated.")

