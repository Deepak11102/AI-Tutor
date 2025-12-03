# Importing Dependencies

import uvicorn
import os
import base64
import random
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from gtts import gTTS  # Import the gTTS library
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from contextlib import asynccontextmanager

# --- FastAPi App Initialization ---

# Global variable for the RAG chain
rag_chain = None

# Lifespan event handler to set up the RAG pipeline on application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    global rag_chain
    rag_chain = await setup_rag_pipeline()
    yield
    print("Application shutdown...")


app = FastAPI(
    title="Conversational AI Tutor",
    description="An end-to-end, RAG-powered AI tutor with gTTS and Gemini LLM.",
    lifespan=lifespan
)

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

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

# --- RAG PIPELINE SETUP ---
# This function loads the knowledge file, chunks it, and sets up the RAG pipeline.
async def setup_rag_pipeline():
    """
    Sets up the RAG pipeline using a text file, a text splitter, an embedding model,
    and a vector store (ChromaDB). This version uses Gemini's models.
    """
    print("Setting up RAG pipeline...")
    try:
        if not GEMINI_API_KEY:
            print("WARNING: Gemini API key is not set as an environment variable. RAG pipeline cannot be initialized.")
            return None

        # Load the document
        loader = TextLoader(KNOWLEDGE_FILE)
        documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Using Gemini's embedding model
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )
        
        # Creating an in-memory vector store from the documents
        vectorstore = Chroma.from_documents(docs, embedding_function)

        # Initialize the LLM with the Gemini model
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY
        )

        # Creating the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        print("RAG pipeline setup complete.")
        return qa_chain
    except Exception as e:
        print(f"Error setting up RAG pipeline: {e}")
        return None

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    """Model for a single query request."""
    text: str

class APIResponse(BaseModel):
    """Model for the API response."""
    text: str
    emotion: str
    audio_base64: Optional[str] = None

# --- TTS (TEXT-TO-SPEECH) with gTTS ---
async def tts_api(text: str) -> str:
    """
    Calls the gTTS library to generate audio from text and returns a base64 encoded string.
    """
    try:
        # Create a BytesIO object to hold the audio data in memory
        audio_stream = BytesIO()
        tts = gTTS(text=text, lang='en', tld='co.in') # 'tld' can be changed for different accents
        tts.write_to_fp(audio_stream)
        
        # Move the cursor to the beginning of the stream before reading
        audio_stream.seek(0)
        audio_data = audio_stream.read()
        
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"An error occurred in gTTS API call: {e}")
        raise HTTPException(status_code=500, detail="Failed to get a response from the gTTS API.")


# --- API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the simple HTML/JS frontend for the mascot UI.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Tutor Mascot</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                color: #334155;
            }}
            .container {{
                background-color: white;
                border-radius: 2rem;
                padding: 2rem;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 90%;
                width: 400px;
            }}
            .mascot-container {{
                position: relative;
                width: 150px;
                height: 150px;
                background-color: #e2e8f0;
                border-radius: 50%;
                margin: 0 auto 2rem;
                transition: background-color 0.5s ease;
                overflow: hidden;
            }}
            .mascot-head {{
                position: absolute;
                bottom: 10px;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 100px;
                background-color: #fcd34d;
                border-radius: 50%;
                transition: transform 0.3s ease;
            }}
            .mascot-eyes {{
                position: absolute;
                top: 40px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 15px;
            }}
            .mascot-eye {{
                width: 10px;
                height: 10px;
                background-color: #334155;
                border-radius: 50%;
            }}
            .mascot-mouth {{
                position: absolute;
                bottom: 25px;
                left: 50%;
                transform: translateX(-50%);
                width: 40px;
                height: 10px;
                background-color: #ef4444;
                border-radius: 5px;
                transition: transform 0.3s ease;
            }}
            .talking .mascot-mouth {{
                animation: talk 0.3s infinite alternate;
            }}
            .happy .mascot-head {{
                transform: translateX(-50%) translateY(-5px);
            }}
            .happy .mascot-mouth {{
                border-radius: 5px 5px 20px 20px;
                height: 20px;
            }}
            .thinking .mascot-head {{
                animation: think 1s infinite;
            }}
            .thinking .mascot-eyes {{
                background-color: #334155;
                border-radius: 50%;
                width: 5px;
                height: 5px;
            }}
            @keyframes talk {{
                from {{ transform: scaleY(1); }}
                to {{ transform: scaleY(0.5); }}
            }}
            @keyframes think {{
                0% {{ transform: translateX(-50%) rotate(0deg); }}
                50% {{ transform: translateX(-50%) rotate(-5deg); }}
                100% {{ transform: translateX(-50%) rotate(0deg); }}
            }}
            .title {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}
            .subtitle {{
                font-size: 1rem;
                color: #64748b;
                margin-bottom: 1.5rem;
            }}
            .mic-button {{
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 1rem 2rem;
                font-size: 1rem;
                border-radius: 1.5rem;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.1s;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin: 0 auto;
            }}
            .mic-button:hover {{
                background-color: #2563eb;
            }}
            .mic-button:active {{
                transform: scale(0.98);
            }}
            .mic-button:disabled {{
                background-color: #94a3b8;
                cursor: not-allowed;
            }}
            .spinner {{
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-top: 4px solid white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                display: none;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .message-box {{
                margin-top: 1.5rem;
                background-color: #f1f5f9;
                border-radius: 1rem;
                padding: 1rem;
                min-height: 50px;
                text-align: left;
                word-wrap: break-word;
            }}
            .text-display {{
                font-size: 1rem;
                color: #1e293b;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">AI Tutor Mascot</h1>
            <p class="subtitle">Press the microphone to ask a question.</p>
            
            <div id="mascot" class="mascot-container">
                <div class="mascot-head">
                    <div class="mascot-eyes">
                        <div class="mascot-eye"></div>
                        <div class="mascot-eye"></div>
                    </div>
                    <div class="mascot-mouth"></div>
                </div>
            </div>

            <button id="mic-button" class="mic-button">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-mic">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" x2="12" y1="19" y2="22" />
                </svg>
                <span id="button-text">Speak</span>
                <div id="spinner" class="spinner"></div>
            </button>
            
            <div id="message-box" class="message-box">
                <p id="text-display" class="text-display">Waiting for you...</p>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const micButton = document.getElementById('mic-button');
                const buttonText = document.getElementById('button-text');
                const textDisplay = document.getElementById('text-display');
                const spinner = document.getElementById('spinner');
                const mascotContainer = document.getElementById('mascot');

                let isRecognizing = false;

                const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                
                // Function to set the mascot's emotional state
                function setMascotEmotion(emotion) {{
                    mascotContainer.className = 'mascot-container ' + emotion;
                }}

                micButton.addEventListener('click', () => {{
                    if (isRecognizing) {{
                        recognition.stop();
                        return;
                    }}
                    
                    isRecognizing = true;
                    micButton.disabled = true;
                    spinner.style.display = 'block';
                    buttonText.textContent = 'Listening...';
                    textDisplay.textContent = "Listening...";
                    setMascotEmotion('thinking');
                    
                    recognition.start();
                }});

                recognition.onresult = (event) => {{
                    const speechResult = event.results[0][0].transcript;
                    textDisplay.textContent = "You: " + speechResult;
                    
                    // Call the backend API
                    fetch('/query', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ text: speechResult }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.text) {{
                            textDisplay.textContent = "AI: " + data.text;
                        }} else {{
                             textDisplay.textContent = "Error: Could not get response from AI. Please check the API key or model availability.";
                        }}
                        
                        setMascotEmotion(data.emotion);
                        
                        // Play the audio
                        if (data.audio_base64) {{
                            const audioData = "data:audio/mpeg;base64," + data.audio_base64;
                            const audio = new Audio(audioData);
                            
                            audio.onplaying = () => {{
                                mascotContainer.classList.add('talking');
                            }};
                            
                            audio.onended = () => {{
                                mascotContainer.classList.remove('talking');
                                setMascotEmotion('calm');
                            }};
                            
                            audio.play().catch(error => {{
                                console.error('Error playing audio:', error);
                                textDisplay.textContent = "Error playing audio.";
                            }});
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        textDisplay.textContent = "Error: Could not get response.";
                        setMascotEmotion('calm');
                    }})
                    .finally(() => {{
                        isRecognizing = false;
                        micButton.disabled = false;
                        spinner.style.display = 'none';
                        buttonText.textContent = 'Speak';
                    }});
                }};

                recognition.onend = () => {{
                    if (isRecognizing) {{
                        isRecognizing = false;
                        micButton.disabled = false;
                        spinner.style.display = 'none';
                        buttonText.textContent = 'Speak';
                        setMascotEmotion('calm');
                    }}
                }};

                recognition.onerror = (event) => {{
                    console.error('Speech recognition error:', event.error);
                    textDisplay.textContent = 'Error: ' + event.error;
                    isRecognizing = false;
                    micButton.disabled = false;
                    spinner.style.display = 'none';
                    buttonText.textContent = 'Speak';
                    setMascotEmotion('calm');
                }};
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/query", response_model=APIResponse)
async def handle_query(request: QueryRequest):
    """
    Handles a single-turn query using the RAG pipeline and returns a response
    with text, emotion, and audio data.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline is not initialized. Please check the server logs."
        )

    print(f"Received query: {request.text}")
    request.text+= ". if there is no context for query then just answer without any knowledge base. just answer with yourself. never say I don't know. you always have to answer the question of the user"

    # Pass the query to the RAG chain
    result = rag_chain.invoke({"query": request.text})
    answer_text = result["result"]

    # Determine a random emotion for demonstration
    EMOTIONS = ["happy", "explaining", "calm", "curious"]
    emotion_state = random.choice(EMOTIONS)
    
    # Generating audio using the TTS function
    try:
        audio_base64 = await tts_api(answer_text)
    except Exception as e:
        print(f"Failed to generate TTS audio: {e}")
        audio_base64 = None

    response_data = {
        "text": answer_text,
        "emotion": emotion_state,
        "audio_base64": audio_base64
    }

    print(f"Responding with text: '{answer_text}' and emotion: '{emotion_state}'")
    return response_data

# --- START THE SERVER ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



