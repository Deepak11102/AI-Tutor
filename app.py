# To run this code, save it as app.py and execute the following commands in order:
# 1. pip cache purge
# 2. pip install streamlit fastapi uvicorn aiohttp pydantic langchain-google-genai gtts --no-cache-dir
# 3. streamlit run app.py

import streamlit as st
import os
import base64
import random
import asyncio
from io import BytesIO
from gtts import gTTS
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
# IMPORTANT: This now securely retrieves the Gemini API key from your environment.
# You MUST set the GEMINI_API_KEY environment variable on your system for this to work.
# Alternatively, you can use st.secrets for Streamlit Cloud.
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_API_KEY = "AIzaSyCZsQMNUsIPP0YrtAeYVnjB7hsrFvobL9k"

# The model for text generation and embeddings.
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# A simple knowledge base file for the RAG pipeline.
KNOWLEDGE_FILE = "knowledge.txt"
if not os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "w") as f:
        f.write("The capital of France is Paris. Paris is also known for the Eiffel Tower. The largest planet in our solar system is Jupiter. Photosynthesis is the process used by plants to convert light energy into chemical energy. The primary colors are red, yellow, and blue. The human heart has four chambers.Physics is the branch of science concerned with the nature and properties of matter and energy. It is often called the “fundamental science” because it provides the foundation upon which chemistry, biology, and other sciences rest. Physics attempts to explain how the universe works using mathematical laws and experimental observations. It is traditionally divided into mechanics, thermodynamics, electromagnetism, optics, waves, and modern physics, each exploring different phenomena." \
                "Mechanics is the study of motion and the forces that cause it. Kinematics describes motion without considering its causes, focusing on displacement, velocity, and acceleration. For uniformly accelerated motion, the kinematic equations—such as v= u + at, s= ut+ 1/2(at^2), v^2 - u^2 = 2as. It provide relationships between the physical quantities. Dynamics introduces Newton’s Laws of Motion, which explain how forces govern the behavior of objects. The first law, the law of inertia, states that an object will remain at rest or continue in uniform motion unless acted upon by an external force. The second law quantifies force as the product of mass and acceleration (F = ma), while the third law asserts that for every action there is an equal and opposite reaction." \
                "Gravitation further extends mechanics by explaining the universal attraction between objects. Newton’s law of gravitation states that every two masses attract one another with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. This law explains planetary motion, tides, and falling objects on Earth.Work, energy, and power are essential concepts in mechanics." \
                "Work is defined as the product of force and displacement in the direction of the force. " \
                "Energy exists in many forms, including kinetic energy (1/2*mv^2) and potential energy (mgh). The work-energy theorem states that the net work done on an object is equal to the change in its kinetic energy. Power, on the other hand, measures how quickly work is done and is expressed as work done per unit time. The principle of conservation of energy emphasizes that energy cannot be created or destroyed; it can only change forms, ensuring the total energy of an isolated system remains constant.Thermodynamics is the study of heat, temperature, and the flow of energy. " \
                "The zeroth law defines temperature as the property that determines thermal equilibrium between objects. The first law, essentially a restatement of energy conservation, is given by Δ𝑈 = 𝑄−𝑊, where internal energy change equals the heat added minus work done by the system. The second law introduces entropy, a measure of disorder, stating that entropy in a closed system tends to increase." \
                "This law explains why heat flows spontaneously from hot to cold bodies and why perpetual motion machines are impossible. The third law asserts that as temperature approaches absolute zero, entropy approaches a minimum. Electromagnetism studies electric and magnetic phenomena. Electrostatics introduces Coulomb’s law, which describes the force between two charges as directly proportional to their magnitudes and inversely proportional to the square of their separation. Electric fields and potentials explain how charges interact in space. " \
                "Current electricity describes the flow of charges through conductors, with Ohm’s law relating current, voltage, and resistance (V = IR). Magnetism arises from moving charges, and magnetic fields can be described by lines of force. Faraday’s law of electromagnetic induction states that a changing magnetic field induces an electromotive force, forming the basis for electric generators and transformers." \
                "Maxwell’s equations later unified electricity and magnetism into a single theoretical framework.Data structures and algorithms are central to computer science. A data structure is a method of organizing and storing data to allow efficient access and modification." \
                "Arrays and linked lists represent linear data, with arrays offering fast indexing and linked lists enabling dynamic resizing. Stacks and queues enforce specific access orders—last-in-first-out and first-in-first-out, respectively. Trees model hierarchical data, with binary trees, binary search trees, AVL trees, and heaps being important variations. Graphs generalize data into networks of nodes and edges, with algorithms like breadth-first search and depth-first search enabling traversal. Sorting algorithms, such as bubble sort, merge sort, quicksort, and heap sort, arrange data in order, while searching algorithms like linear search and binary search locate elements efficiently." \
                "Algorithm analysis uses Big-O notation to classify time and space complexity, ensuring scalable solutions to computational problems.")

# --- RAG PIPELINE SETUP (Now a function to be cached) ---
# Use Streamlit's caching mechanism to avoid re-running this expensive function.
@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up and caches the RAG pipeline. This will run only once.
    """
    print("Setting up RAG pipeline...")
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            st.error("Please set the GEMINI_API_KEY environment variable or replace the placeholder in the code.")
            return None

        loader = TextLoader(KNOWLEDGE_FILE)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )

        vectorstore = Chroma.from_documents(docs, embedding_function)

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RAG pipeline setup complete.")
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None

# --- TTS (TEXT-TO-SPEECH) with gTTS ---
def tts_api(text: str) -> str:
    """
    Calls the gTTS library to generate audio from text and returns a base64 encoded string.
    """
    try:
        audio_stream = BytesIO()
        tts = gTTS(text=text, lang='en', tld='co.in')
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        audio_data = audio_stream.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        st.error(f"An error occurred in gTTS API call: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="AI Tutor Mascot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("AI Tutor Mascot")
st.markdown("Press the microphone to ask a question.")

# Initialize the RAG chain and store it in the session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_rag_pipeline()
if st.session_state.rag_chain is None:
    st.stop()

# This is a placeholder for the mascot animation. Streamlit does not have native
# animated components, so we'll just show a static image or a GIF.
# The user's original HTML/JS for a talking mascot is not directly portable.
st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGhscHhkaDBxYThqNWR5bHpwZ2x4bjl2eWtvMWVsd3dwcW54M3F0ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0O9z9c0qJ673d3aA/giphy.gif", width=200)

# We'll use Streamlit's native input/output widgets instead of a custom HTML/JS interface.
# The `st.chat_message` widget is perfect for this.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format='audio/ogg')

# Accept user input
if prompt := st.chat_input("Ask me a question about physics or computer science..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the RAG chain
    with st.spinner("Thinking..."):
        try:
            # Pass the query to the RAG chain
            query_with_fallback = prompt + ". if there is no context for query then just answer without any knowledge base. just answer with yourself. never say I don't know. you always have to answer the question of the user"
            result = st.session_state.rag_chain.invoke({"query": query_with_fallback})
            answer_text = result["result"]
            
            # Generate audio using the TTS function
            audio_base64 = tts_api(answer_text)
            
            # Display AI message in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    st.audio(audio_bytes, format='audio/ogg')
                    st.session_state.messages.append({"role": "assistant", "content": answer_text, "audio": audio_bytes})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "An error occurred while generating the response."})