# Conversational AI Tutor 🤖

An end-to-end, RAG-powered AI tutor with gTTS and Gemini LLM.

This project is a simple yet powerful conversational AI tutor built with FastAPI, LangChain, and the Google Gemini models. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on a custom knowledge base and features a Text-to-Speech (TTS) integration using the Google Text-to-Speech (gTTS) library to provide spoken responses.

## ✨ Features

  * **FastAPI Backend**: A lightweight and fast web server for handling API requests.
  * **Gemini Integration**: Utilizes the Google Gemini LLM for generating intelligent responses.
  * **Retrieval-Augmented Generation (RAG)**: Augments the LLM's knowledge with a custom text file (`knowledge.txt`), ensuring answers are relevant to the provided context.
  * **Text-to-Speech (TTS)**: Converts the AI's textual responses into high-quality audio using `gTTS`.
  * **Web UI**: A simple HTML/JS frontend with a "mascot" that animates based on the AI's state (listening, thinking, talking).
  * **Speech Recognition**: Uses the browser's native Web Speech API to convert your spoken queries into text.

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

  * Python 3.8+
  * A Google Cloud account with access to the Gemini API.
  * The **`GEMINI_API_KEY`** environment variable set with your API key.

## 🚀 Setup & Installation

Follow these steps to get the application up and running.

### 1\. **Clone the Repository**

First, clone the repository or save the provided code into a file named `app.py`.

### 2\. **Set Up the Environment**

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. **Install Dependencies**

The `pip install` command is provided in the code comments. Run it to install all necessary libraries. The `--no-cache-dir` flag ensures a clean installation.

```bash
pip uninstall uvicorn -y
pip cache purge
pip install fastapi uvicorn aiohttp pydantic langchain-google-genai gtts --no-cache-dir
```

### 4\. **Configure Your API Key**

The application retrieves your Gemini API key from an environment variable. **This is the most secure way to manage your key.**

  * **On Linux/macOS:**
    ```bash
    export GEMINI_API_KEY="AIzaSyCZsQMNUsIPP0YrtAeYVnjB7hsrFvobL9k"
    ```
  * **On Windows (Command Prompt):**
    ```bash
    set GEMINI_API_KEY="AIzaSyCZsQMNUsIPP0YrtAeYVnjB7hsrFvobL9k"
    ```

**Note:** The key shown above is an example. Replace it with your actual key.

### 5\. **Run the Application**

The code is configured to start the server automatically when executed.

```bash
python app.py
```

The server will start at `http://0.0.0.0:8000`. You can access the application by navigating to this address in your web browser.

## 📖 How it Works

### Backend (`app.py`)

1.  **`lifespan`**: The `lifespan` function is a modern FastAPI feature that runs code on application startup. It calls `setup_rag_pipeline()` to prepare the AI.
2.  **`setup_rag_pipeline()`**: This function is the core of the AI logic:
      * It reads the `knowledge.txt` file.
      * The `CharacterTextSplitter` breaks the text into manageable chunks.
      * `GoogleGenerativeAIEmbeddings` converts these text chunks into numerical vectors (embeddings).
      * `Chroma.from_documents` creates an in-memory vector store, making the knowledge searchable.
      * Finally, a `RetrievalQA` chain is created. When a user asks a question, this chain finds the most relevant chunks from the vector store and provides them as context to the Gemini LLM to formulate a precise answer.
3.  **`/query` Endpoint**:
      * This API endpoint receives the user's text query.
      * It invokes the `rag_chain` to get a response based on the knowledge base.
      * It then calls the `tts_api` function to generate a base64-encoded audio string from the response text.
      * The response, including the text, a random emotion, and the audio data, is sent back to the frontend.
4.  **`tts_api()`**: A utility function that takes a string and uses `gTTS` to generate and encode the audio data.

### Frontend (HTML/JS)

1.  **UI**: A simple UI with a microphone button and an animated mascot.
2.  **Web Speech API**: The JavaScript code uses `window.SpeechRecognition` to capture audio from the user's microphone and convert it to text.
3.  **API Call**: The recognized text is sent to the `/query` endpoint via a `fetch` request.
4.  **Audio Playback**: The base64-encoded audio data received from the backend is converted into an audio object and played in the browser, providing a seamless conversational experience. The mascot's animation (`talking` class) is synchronized with the audio playback.

## 🤝 Contributing

Feel free to fork the repository, modify the `knowledge.txt` file, or enhance the UI and backend logic. Contributions are welcome\!