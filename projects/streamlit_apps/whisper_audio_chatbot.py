import os 
import tempfile
import hashlib
import streamlit as st
import whisper
import time
from whisper.tokenizer import LANGUAGES
from dotenv import load_dotenv
from gtts import gTTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from st_audiorec import st_audiorec

# --- Setup ---

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
load_dotenv()

st.set_page_config(page_title="Chat with Audio", page_icon="🎙️", layout="wide")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

try:
    from google.generativeai import configure
    configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google AI: {e}")
    st.stop()

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    whisper_model = whisper.load_model("base")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
    return whisper_model, llm, embeddings

whisper_model, llm, embeddings = load_models()

# --- Session State Init ---
def init_session_state():
    defaults = {
        "messages": [], "vectorstore": None, "transcript": "",
        "current_file_name": None, "last_audio_hash": None,
        "selected_language": "auto", "detected_language": "en",
        "target_language": "none"  
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
init_session_state()

# --- Language Consistency Helper ---
def get_session_language():
    """Determines the active language for the session (translation or original)."""
    target_lang = st.session_state.get("target_language", "none")
    if target_lang and target_lang != "none":
        return target_lang
    # Fallback to the language detected in the original audio
    return st.session_state.get("detected_language", "en")

# --- Translate Utility ---
def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using the Gemini LLM."""
    if target_lang == "none" or not text.strip():
        return text
    try:
        # Get the full language name for a better prompt
        target_language_name = LANGUAGES.get(target_lang, target_lang).title()

        prompt = f"Translate the following text to {target_language_name}. Only return the translated text, without any introductory phrases:\n\n---\n{text}\n---"
        time.sleep(1.5)
        # Invoke the LLM. The response from ChatGoogleGenerativeAI is an AIMessage.
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# --- Process Audio ---
def process_audio_file(uploaded_file, language_code: str):
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.transcript = ""
    st.session_state.current_file_name = uploaded_file.name

    with st.status("Processing audio file...", expanded=True) as status:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name

        try:
            lang_display = "Auto-Detect" if language_code == "auto" else LANGUAGES.get(language_code, language_code).title()
            status.update(label=f"🔍 Transcribing audio (Language: {lang_display})...")

            transcribe_options = {"fp16": False}
            if language_code != "auto":
                transcribe_options["language"] = language_code

            result = whisper_model.transcribe(audio_path, **transcribe_options)
            original_text = result.get("text", "Transcription failed.")
            detected_lang_code = result.get("language", "en")

            # Store actual detected language
            st.session_state.detected_language = detected_lang_code

            # Translate if needed
            target_lang = st.session_state.target_language
            translated_text = translate_text(original_text, target_lang)

            st.session_state.transcript = translated_text

            if not translated_text.strip():
                status.update(label="⚠️ Transcription resulted in empty text.", state="warning")
                return

            status.update(label="📚 Splitting transcript and creating embeddings...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = [Document(page_content=translated_text)]
            chunks = splitter.split_documents(docs)

            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            status.update(label="✅ Ready to chat!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"An error occurred: {e}", state="error")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

# --- Chat Response ---
def get_rag_response(user_query: str) -> str:
    vectorstore = st.session_state.vectorstore
    if not vectorstore:
        return "Vector store not initialized."

    relevant_chunks = vectorstore.similarity_search(user_query, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_chunks)

    # Determine the language for the response using the session language.
    response_lang_code = get_session_language()
    response_language_name = LANGUAGES.get(response_lang_code, response_lang_code).title()

    prompt_template = f"""
You are a helpful assistant. Your goal is to be accurate and concise.
Answer the user's question based *only* on the provided transcript context.
You MUST respond in the following language: **{response_language_name}**.

Context from the audio transcript:
---
{{context}}
---

Question: {{question}}

Answer (in {response_language_name}):
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    time.sleep(1.5)
    result = chain.invoke({"context": context, "question": user_query})
    return result.content

# --- Text-to-Speech ---
def generate_audio_response(text: str, lang: str) -> bytes | None:
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            tmp_file.seek(0)
            audio_data = tmp_file.read()
        os.remove(tmp_file.name)
        return audio_data
    except Exception as e:
        st.error(f"TTS failed: {e}")
        return None

# --- Handle Queries ---
def handle_query_submission(query: str):
    if not query.strip():
        return
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_rag_response(query)
            # Determine language for Text-to-Speech using the session language
            audio_lang = get_session_language()
            audio_bytes = generate_audio_response(answer, lang=audio_lang)
        st.markdown(answer)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "audio": audio_bytes
        })

# --- UI Layout ---
st.title("🎙️ Chat with Your Audio")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload & Process Audio")
    uploaded_file = st.file_uploader("Supported: .mp3, .wav, .m4a, .mp4", type=["mp3", "wav", "m4a", "mp4"])

    language_options = {"auto": "Auto-Detect"}
    language_options.update({code: name.title() for code, name in LANGUAGES.items()})
    selected_language = st.selectbox("Select Audio Language", options=list(language_options.keys()),
                                     format_func=lambda key: language_options[key], key="selected_language")

    # 🌍 Target language for translation
    st.selectbox(
        "Translate Transcript To", 
        options=["none", "en", "hi", "te", "ta", "fr"], 
        format_func=lambda x: "No Translation" if x == "none" else x.upper(),
        key="target_language"
    )

    if st.button("Process Audio", disabled=not uploaded_file, use_container_width=True):
        process_audio_file(uploaded_file, st.session_state.selected_language)

    if st.session_state.transcript:
        st.header("2. Full Transcript")
        st.text_area("Transcribed text", st.session_state.transcript, height=300, key="transcript_display")
        st.download_button(
            label="📥 Download Transcript",
            data=st.session_state.transcript.encode('utf-8'),
            file_name=f"{os.path.splitext(st.session_state.current_file_name)[0]}_transcript.txt",
            mime="text/plain"
        )

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Interface ---
if not st.session_state.vectorstore:
    st.info("Please upload an audio file to start the conversation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg["role"] == "assistant" and msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

# Voice Query
query_to_process = None

# Always show mic input when vectorstore is ready
if st.session_state.vectorstore:
    st.write("🎙️ Speak your next question below:")
    wav_audio_data = st_audiorec()

    if wav_audio_data:
        audio_hash = hashlib.sha256(wav_audio_data).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing your voice query..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_q:
                    tmp_audio_q.write(wav_audio_data)

                    query_lang_code = get_session_language()
                    transcribe_options = {"fp16": False, "language": query_lang_code}
                    result = whisper_model.transcribe(tmp_audio_q.name, **transcribe_options)
                    q_text = result.get("text", "").strip()

                    if q_text:
                        st.write("🗣️ Transcribed query:", q_text)
                        query_to_process = q_text

                os.remove(tmp_audio_q.name)


# Text Input
if text_prompt := st.chat_input("Ask your question..."):
    if st.session_state.vectorstore:
        query_to_process = text_prompt
    else:
        st.toast("Please upload an audio file first.", icon="⚠️")

# Submit query
if query_to_process:
    handle_query_submission(query_to_process)
    st.rerun()
