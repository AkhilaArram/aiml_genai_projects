# Gemini PDF Chatbot with RouterChain, History, and Summarization
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob
import re # For robust JSON extraction
import json
import time
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfMerger
import streamlit as st
import whisper
import tempfile
import hashlib
from st_audiorec import st_audiorec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.router.llm_router import RouterOutputParser # Keep for inheritance
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableBranch
from typing import Dict, Any

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# --- Constants ---
PDF_DIR = "pdfs"
CHAT_HISTORY_FILE = "chat_history.json"
SUMMARY_FILE = "summary.txt"

# --- Model and Tool Initialization ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Main LLM for generation
    llm_instance = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/gemini-1.5-flash", temperature=0.7)
    # LLM for routing - low temperature for deterministic JSON output
    router_llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/gemini-1.5-flash", temperature=0.0)
except Exception as e:
    st.error(f"Failed to initialize Google Generative AI: {e}")
    st.stop()

@st.cache_resource(show_spinner="Loading speech-to-text model...")
def load_whisper_model():
    model = whisper.load_model("base")
    return model

whisper_model = load_whisper_model()

# Create PDFs if not exists
topics = ["forest", "beach", "sea", "trees", "flowers", "mountains", "desert", "river", "lake", "rainforest",
          "savanna", "tundra", "volcano", "island", "canyon", "waterfall", "meadow", "valley", "swamp", "reef",
          "prairie", "glacier", "bay", "lagoon", "delta", "grove", "orchard", "jungle", "cliff", "plateau",
          "hill", "plain", "dune", "marsh", "steppe", "woodland", "mangrove", "oasis", "peninsula", "cape",
          "gulf", "fjord", "atoll", "archipelago", "shoal", "moor", "badlands", "rainbow", "aurora", "geyser"]

os.makedirs(PDF_DIR, exist_ok=True)
if not all(os.path.exists(os.path.join(PDF_DIR, f"{topic}.pdf")) for topic in topics):
    with st.status("Generating missing PDFs...", expanded=True) as status:
        pdfs_generated_this_session = 0
        for i, topic in enumerate(topics):
            pdf_path = os.path.join(PDF_DIR, f"{topic}.pdf")
            if os.path.exists(pdf_path):
                continue
            
            try:
                model_gen = genai.GenerativeModel('gemini-1.5-flash')
                response = model_gen.generate_content(f"Write an article about {topic}.")
                content = response.text
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                safe_content = content.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, safe_content)
                pdf.output(pdf_path)
                pdfs_generated_this_session += 1
                status.update(label=f"Generated PDF for {topic} ({i+1}/{len(topics)})")
                time.sleep(3)
                if pdfs_generated_this_session > 0 and pdfs_generated_this_session % 10 == 0:
                    status.update(label=f"Generated {pdfs_generated_this_session} PDFs, pausing for 10s...")
                    time.sleep(10)
            except Exception as e:
                st.warning(f"Error generating PDF for {topic}: {e}")
                time.sleep(5)
                continue
        if pdfs_generated_this_session > 0:
            status.update(label="PDF generation complete.", state="complete")
        else:
            status.update(label="All PDFs already exist.", state="complete")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model():
    """Initializes and caches the embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

@st.cache_resource(show_spinner="Embedding PDF documents...")
def get_pdf_vectorstore(_embeddings):
    """Loads all PDFs, splits them, and creates a FAISS vector store."""
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_files:
        st.error(f"No PDF files found in '{PDF_DIR}'. Cannot proceed.")
        return None
 
    all_docs = []
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load PDF {file_path}: {e}")
 
    if not all_docs:
        st.error("No documents could be loaded from the PDFs.")
        return None
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)
 
    try:
        vectordb = FAISS.from_documents(splits, _embeddings)
        return vectordb
    except Exception as e:
        st.error(f"Failed to create FAISS vector store for PDF content: {e}")
        return None

@st.cache_resource(show_spinner="Generating and caching document summary...")
def get_summary_vectorstore(_llm, _embeddings):
    """Generates a summary if it doesn't exist, then creates a vector store for it."""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_text = f.read()
    else:
        pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
        docs_for_summary = [] 
        pdf_files_for_summary = pdf_files[:1] # Summarize only the first PDF

        try:
            loader_summary = PyPDFLoader(pdf_files_for_summary[0])
            docs_for_summary.extend(loader_summary.load())
            summarizer = load_summarize_chain(_llm, chain_type="map_reduce")
            summary_result = summarizer.invoke({"input_documents": docs_for_summary})
            summary_text = summary_result.get("output_text", "Summary could not be generated.")
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                f.write(summary_text)
        except Exception as e:
            st.warning(f"Could not generate summary: {e}")
            summary_text = "Summary is unavailable due to an error."

    summary_doc = Document(page_content=summary_text)
    return FAISS.from_documents([summary_doc], _embeddings)

@st.cache_resource(show_spinner="Loading chat history...")
def get_history_vectorstore(_embeddings):
    """Initializes a vector store for chat history, loading from file if it exists."""
    placeholder_doc = Document(page_content="Initial placeholder for chat history vector store.")
    history_vectordb = FAISS.from_documents([placeholder_doc], _embeddings)

    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                chat_history_from_file = json.load(f)
            
            history_docs = [
                Document(page_content=f"Previous Question: {item['question']}\nPrevious Answer: {item['text']}")
                for item in chat_history_from_file
                if item.get("sender") == "Gemini" and item.get("question") and item.get("text")
            ]
            if history_docs:
                history_vectordb.add_documents(history_docs)
        except Exception as e:
            st.warning(f"Error loading chat history from {CHAT_HISTORY_FILE}: {e}")
    return history_vectordb

# --- Initialize Vector Stores and Retrievers ---
embeddings_model = get_embeddings_model()
vectordb = get_pdf_vectorstore(embeddings_model)
summary_vectordb = get_summary_vectorstore(llm_instance, embeddings_model)

if "history_vectordb" not in st.session_state:
    st.session_state.history_vectordb = get_history_vectorstore(embeddings_model)

if not vectordb or not summary_vectordb or not st.session_state.history_vectordb:
    st.error("A required vector store could not be initialized. The application cannot continue.")
    st.stop()
HISTORY_QA_TEMPLATE = """
You are a helpful AI assistant. You have access to previous parts of the conversation history.
Your SOLE purpose when answering the user's current question is to use the "Retrieved Conversation History" provided below.
Do NOT use your general knowledge or any other information source to answer about the topic itself if it's not in the history.

Instructions for answering:
1.  **Analyze the User's Current Question:**
    *   Identify the main subject of the user's question about past discussions. Let's call this the Query_Subject.
    *   Determine if the question is simple (e.g., "Did we discuss Query_Subject?") or compound (e.g., "Did we discuss Query_Subject? If so, explain Query_Subject.").
    *   Also, handle general queries like "What did we talk about?".

2.  *   For Specific Topic Queries (e.g., "Did we discuss [Query_Subject]?"):**
    *   Examine the "Retrieved Conversation History" ({context}).
    *   Determine if any Q&A pairs in the {context} are directly about the Query_Subject or a very closely related aspect or sub-topic of the Query_Subject.
        (For example, if Query_Subject is "volcanoes", a "very closely related aspect" could be "volcano eruptions" or "types of volcanoes". If Query_Subject is "trees", a related aspect could be "deforestation" or "pine trees".)
    *   If such relevant Q&A pairs are found (let's call the topic found in history History_Topic):
        *   Acknowledge this. If History_Topic is the same as Query_Subject, you can say: "Yes, we previously discussed [Query_Subject]."
        *   If History_Topic is slightly different but very closely related, state what was discussed. For example: "Yes, we touched upon a related topic. Our conversation included discussion about [History_Topic]."
        *   If the user's question was simple (e.g., "Did we discuss volcanoes?"), and you found "volcano eruptions" as the History_Topic, you might add: "Were you referring to our discussion on volcano eruptions, or do you have a more general question about volcanoes?"
    *   If no Q&A pairs directly about Query_Subject or a very closely related aspect are found in {context} (or {context} is empty/placeholder):
        Respond: "No, it doesn't look like we've specifically discussed [Query_Subject] in our current conversation. If you'd like to know if I have information on [Query_Subject] from my documents, you can ask a direct question such as 'Tell me about [Query_Subject]' or 'What do the documents say about [Query_Subject]?'."

3.  **Address the "Conditional Explanation" Part of the Query (if present):**
    *   This applies if step 2 determined that Query_Subject (or a closely related History_Topic) *was* discussed.
    *   **Scenario A: "IF SO, explain [Query_Subject]"**:
        a.  Look for the specific explanation of Query_Subject (or History_Topic if that's what was found and is relevant to the explanation request) in the "Retrieved Conversation History".
        b.  If the explanation is found in the history: Provide it directly from the history. (e.g., "Yes, we previously discussed mountains. Regarding their explanation, the history shows: [explanation from history].")
        c.  If an explanation for the specific Query_Subject (or the relevant History_Topic) is NOT found in the history (even if the topic was mentioned):
            Acknowledge what was discussed (e.g., "Yes, we discussed [History_Topic]."). Then add: "...However, a specific explanation for [Query_Subject or History_Topic, as appropriate] is not in our retrieved conversation. If you'd like an explanation from the documents, please ask this as a new, direct question (e.g., 'Explain [Query_Subject]')."    
             Do NOT provide an explanation from general knowledge.

    *   **Scenario B: "IF NOT, explain [Query_Subject]"** (and Query_Subject was *not* discussed according to step 2):
        a.  Start with the response from step 2 for "[Query_Subject] was NOT discussed". Then append: "Since we haven't discussed it, I cannot provide an explanation based on our past conversation. As mentioned, if you would like an explanation of [Query_Subject] from the documents, please ask a new, direct question (e.g., 'Explain [Query_Subject]')."
          
        Do NOT provide an explanation from general knowledge or PDFs yourself. Your role is to report on history and guide the user.

    *   **Scenario C: Direct question routed to history (e.g., user asks "Explain [Query_Subject]" and router thinks it's a follow-up):**
        a.  Check if an explanation for Query_Subject is in the "Retrieved Conversation History".
        b.  If found: Provide it. (e.g., "Regarding X, our previous conversation includes: [content from history].")
        c.  If not found: State: "I don't have information about [Query_Subject] in our current conversation history. If you'd like this information from the documents, please ask this as a new, direct question."
4.  **For General Queries about Past Discussions (e.g., "What topics/questions did we discuss?"):**
    *   Examine the "Retrieved Conversation History" ({context}).
    *   If the {context} is empty, contains only a placeholder (e.g., "Initial placeholder for chat history vector store.", "Chat history is currently unavailable."), or does not contain any actual Q&A pairs from the current session:
        Respond: "Based on our current conversation, it seems we haven't discussed any specific topics yet, or the history is not detailed enough to list them."
    *   Otherwise (if {context} contains actual Q&A pairs from the current session):
        List the main questions or topics evident from the "Previous Question:" parts of the {context}. For example: "Based on our current conversation, we've touched upon questions like: '[Previous Question 1]', '[Previous Question 2]'." or "So far in our conversation, we've discussed topics related to: [topic from Q1], [topic from Q2]."
        Be literal to the content of the retrieved Q&A pairs. Do not invent topics.

5.  **Strict Adherence to Provided Context and Role:**
    *   If the user's question is only "Did we discuss X?", your answer should be the full response formulated in step 2.
    *   Do not add any extra information or explanations unless explicitly requested as part of a conditional clause AND that information is present in the retrieved history.
    *   Your response should clearly indicate what information comes from history and when information is lacking in history.

    *   Remember, your primary function here is to report on the contents (or absence of contents) of the "Retrieved Conversation History" and guide the user on how to get information from documents if it's not in the history. You do not access documents or general knowledge yourself in this role.
Retrieved Conversation History:
{context}

User's Current Question:
{question}

Answer:"""
history_qa_prompt = PromptTemplate(template=HISTORY_QA_TEMPLATE, input_variables=["context", "question"])

# --- Setup Chains (do this once and store in session state) ---
if "chains_initialized" not in st.session_state:
    # 1. PDF Content QA Chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm_instance, retriever=vectordb.as_retriever(), return_source_documents=True
    )

    # 2. Summary QA Chain
    st.session_state.summary_chain = RetrievalQA.from_chain_type(
        llm=llm_instance, retriever=summary_vectordb.as_retriever(), return_source_documents=True
    )

    # 3. History QA Chain
    history_retriever = st.session_state.history_vectordb.as_retriever(search_kwargs={"k": 2})
    st.session_state.history_chain = RetrievalQA.from_chain_type(
        llm=llm_instance,
        retriever=history_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": history_qa_prompt},
        return_source_documents=True,
    )

    # 4. Direct "Last Question" Chain
    def retrieve_previous_user_question_text(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        last_q_text = "You haven't asked any questions before this one in this session."
        # Access the global `chat_history` list
        if chat_history and isinstance(chat_history[-1], dict) and "question" in chat_history[-1]:
            last_q_text = chat_history[-1]["question"]
        return {"result": f"The last question you asked was: \"{last_q_text}\""}

    st.session_state.last_question_direct_chain = RunnableLambda(retrieve_previous_user_question_text)

    st.session_state.chains_initialized = True

# --- Chain to directly get the text of the actual previous user question ---
destination_info = [
    {
        "name": "pdf_content_qa",
        "description": "Good for answering questions about the specific content of the uploaded PDF documents. Use this for factual queries based on the provided texts.",
        "chain": st.session_state.qa_chain,
    },
    {
        "name": "chat_history_qa",
        "description": "Good for answering questions about what has been discussed previously in this conversation. Use this if the question refers to past interactions, is a follow-up, or is a repeat of a previous question.",
        "chain": st.session_state.history_chain,
    },
    {
        "name": "document_summary_qa",
        "description": "Good for answering questions about the overall summary of the PDF documents. Use this for high-level overview questions.",
        "chain": st.session_state.summary_chain,
    },
    {
        "name": "get_last_user_question",
        "description": "Use this if the user is asking specifically 'what was the last question I asked', 'what did I just ask', 'what was my previous query', or similar direct inquiries about their immediately preceding utterance. Do not use this for follow-up questions on a topic, only for recalling the literal previous question text.",
        "chain": st.session_state.last_question_direct_chain,
    }
]
# KNOWN_DESTINATION_NAMES and router setup will now use the correctly defined history_chain

KNOWN_DESTINATION_NAMES = [d["name"] for d in destination_info]

CUSTOM_ROUTER_TEMPLATE_STRING = """
Given the user's query and a list of available destinations, select the most appropriate destination.
The available destinations are:
{destinations}

User Query: {input}

<< FORMATTING >>
Respond with ONLY a JSON object. Do not include any other text, explanations, or markdown code block syntax.
The JSON object must conform to the following schema:
{{
    "destination": "string_value_representing_destination_name",
    "next_inputs": "string_value_of_original_or_rephrased_user_query"
}}

"""
# --- Custom Sanitizing Router Output Parser ---
class SanitizingRouterOutputParser(RouterOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        print(f"\n[ROUTER_DEBUG] Raw text from LLM for routing: >>>{text}<<<")
        json_text_for_error_reporting = text # Store original text for error reporting
        try:
            # Attempt to extract JSON block more robustly
            match_md = re.search(r"```json\s*([\s\S]+?)\s*```", text, re.DOTALL)
            if match_md:
                json_text = match_md.group(1)
            else:
                # If no markdown fence, try to find a JSON object starting with { and ending with }
                match_obj = re.search(r"^\s*({[\s\S]*})\s*$", text, re.DOTALL)
                if match_obj:
                    json_text = match_obj.group(1)
                else:
                    # Fallback: strip the whole text and hope it's JSON.
                    json_text = text.strip()

            json_text_for_error_reporting = json_text # Update for more specific error context
            json_text = json_text.strip() # Clean the extracted/stripped text
            print(f"[ROUTER_DEBUG] Attempting to parse JSON: >>>{json_text}<<<")

            if not (json_text.startswith("{") and json_text.endswith("}")):
                raise json.JSONDecodeError("Extracted text does not appear to be a JSON object.", json_text, 0)

            parsed_json = json.loads(json_text)

            sanitized_response = {}
            for key, value in parsed_json.items():
                sanitized_key = key.strip() # Strip leading/trailing whitespace from the key string
                # Additionally, remove potential surrounding quotes from the key itself
                if len(sanitized_key) > 1: # Ensure key is not empty or a single quote
                    if sanitized_key.startswith('"') and sanitized_key.endswith('"'):
                        sanitized_key = sanitized_key[1:-1]
                    elif sanitized_key.startswith("'") and sanitized_key.endswith("'"):
                        sanitized_key = sanitized_key[1:-1]
                sanitized_response[sanitized_key] = value

            # Now validate the SANITIZED response
            if "destination" not in sanitized_response:
                raise OutputParserException(
                    f"Sanitized output missing 'destination' key. Original text: '{text[:500]}...'. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Sanitized response keys: {list(sanitized_response.keys())}")
            if "next_inputs" not in sanitized_response:
                raise OutputParserException(
                    f"Sanitized output missing 'next_inputs' key. Original text: '{text[:500]}...'. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Sanitized response keys: {list(sanitized_response.keys())}")

            # --- Clean and validate the 'destination' VALUE ---
            raw_destination_name = sanitized_response["destination"]
            if not isinstance(raw_destination_name, str):
                raise OutputParserException(
                    f"Value for 'destination' key from LLM should be a string, but got {type(raw_destination_name)}. Value: {raw_destination_name}")

            cleaned_destination_value = raw_destination_name.strip()
            if len(cleaned_destination_value) > 1: # Remove potential surrounding quotes from the value itself
                if cleaned_destination_value.startswith('"') and cleaned_destination_value.endswith('"'):
                    cleaned_destination_value = cleaned_destination_value[1:-1]
                elif cleaned_destination_value.startswith("'") and cleaned_destination_value.endswith("'"):
                    cleaned_destination_value = cleaned_destination_value[1:-1]

            if cleaned_destination_value not in KNOWN_DESTINATION_NAMES:
                raise OutputParserException(
                    f"LLM returned an invalid or unknown destination name: '{cleaned_destination_value}'. "
                    f"Original value from LLM: '{raw_destination_name}'. Valid destinations are: {KNOWN_DESTINATION_NAMES}")
            # --- End cleaning and validation of 'destination' VALUE ---

            # Ensure next_inputs from LLM is a string as per the prompt's instruction
            raw_next_inputs = sanitized_response["next_inputs"]
            if not isinstance(raw_next_inputs, str):
                raise OutputParserException(
                    f"'next_inputs' from LLM should be a string but got {type(raw_next_inputs)}. Value: {raw_next_inputs}")

            # Format next_inputs for the destination chain (e.g., RetrievalQA expects "query")
            formatted_next_inputs = {"query": raw_next_inputs}

            # Prepare the result dictionary using the cleaned destination value
            result_dict = {
                "destination": cleaned_destination_value,
                "next_inputs": formatted_next_inputs
            }

            print(f"[ROUTER_DEBUG] Successfully parsed and sanitized router output: {result_dict}\n")
            return result_dict
        except json.JSONDecodeError as e:
            print(f"[ROUTER_ERROR] JSONDecodeError during routing: {e}. Attempted to parse: '{json_text_for_error_reporting[:500]}...'")
            raise OutputParserException(f"Failed to decode LLM output as JSON. Attempted to parse: '{json_text_for_error_reporting[:500]}...'. Original text: '{text[:500]}...'. Error: {e}") from e
        except Exception as e:
            print(f"[ROUTER_ERROR] Generic exception during routing: {e}. Original text: '{text[:500]}...'")
            raise OutputParserException(f"Failed to parse or sanitize LLM output. Original text: '{text[:500]}...'. Error: {e}") from e

router_parser = SanitizingRouterOutputParser()
destinations_str = "\n".join([f"{d['name']}: {d['description']}" for d in destination_info])

# Router prompt template
router_prompt_template_obj = PromptTemplate(
    template=CUSTOM_ROUTER_TEMPLATE_STRING,
    input_variables=["input", "destinations"],
    # output_parser is removed here, will be piped in LCEL
)

# --- LCEL Router Chain Construction ---
# This lambda function takes the input dictionary (which contains the user's "input")
# and adds the "destinations" string to it, so the prompt template can be formatted.
# destinations_str is captured from the outer scope.
def add_destinations_to_router_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {**input_dict, "destinations": destinations_str}

lcel_router_chain = (
    RunnableLambda(add_destinations_to_router_input)
    | router_prompt_template_obj
    | router_llm
    | router_parser
)

# Helper lambda to extract the 'next_inputs' (which is {"query": "..."})
# for the destination chains, as they expect this format.
extract_next_inputs = RunnableLambda(lambda x: x["next_inputs"])

# Define the chains for each branch by piping the extracted inputs to the respective QA chain from session_state
pdf_branch_chain = extract_next_inputs | st.session_state.qa_chain
history_branch_chain = extract_next_inputs | st.session_state.history_chain
summary_branch_chain = extract_next_inputs | st.session_state.summary_chain
direct_last_q_branch_chain = st.session_state.last_question_direct_chain # This chain doesn't need extract_next_inputs as it works differently
# Default chain if no other branch is matched (e.g., router outputs an unexpected destination)
# This also needs to process the 'next_inputs'
default_branch_chain_for_routing = extract_next_inputs | st.session_state.qa_chain

# Create the RunnableBranch
# Each branch is a tuple: (condition_lambda, chain_to_run_if_true)
# The condition_lambda operates on the output of lcel_router_chain
lcel_routing_branch = RunnableBranch(
    (lambda x: x["destination"] == "pdf_content_qa", pdf_branch_chain),
    (lambda x: x["destination"] == "chat_history_qa", history_branch_chain),
    (lambda x: x["destination"] == "document_summary_qa", summary_branch_chain),
    (lambda x: x["destination"] == "get_last_user_question", direct_last_q_branch_chain),
    default_branch_chain_for_routing,  # Default case
)

# Combine the router chain with the branch
# The output of lcel_router_chain is fed into lcel_routing_branch
final_chain = lcel_router_chain | lcel_routing_branch

# --- Chat history list (for saving to JSON and UI display) ---
chat_history = [] # This is the Python list for raw history
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except Exception as e:
        st.warning(f"Error loading chat_history list from {CHAT_HISTORY_FILE}: {e}")
        chat_history = []

# Streamlit UI
st.title("Gemini PDF Chatbot with RouterChain")

if "messages" not in st.session_state:
    st.session_state.messages = []
    if chat_history:
        for item in chat_history:
            sender = item.get("sender", "Unknown")
            message_content = item.get("text") if sender == "Gemini" else item.get("question", item.get("text", ""))
            if message_content:
                 st.session_state.messages.append((sender, message_content))

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# --- Combined Input Handling (Text and Voice) ---
query_to_process = None

# Voice Input
st.markdown("---")
st.subheader("Ask with your voice")
wav_audio_data = st_audiorec()

if wav_audio_data:
    audio_hash = hashlib.sha256(wav_audio_data).hexdigest()
    # Process only if it's a new recording
    if audio_hash != st.session_state.get("last_audio_hash"):
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing your voice query..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                tmp_audio_file.write(wav_audio_data)
                audio_path = tmp_audio_file.name
            
            try:
                # Transcribe using whisper
                transcription_result = whisper_model.transcribe(audio_path, fp16=False)
                transcribed_text = transcription_result.get("text", "").strip()
                
                if transcribed_text:
                    st.info(f"🗣️ Transcribed query: \"{transcribed_text}\"")
                    query_to_process = transcribed_text
                else:
                    st.warning("Could not transcribe audio, please try speaking again or type your question.")
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

# Text Input
if text_input := st.chat_input("Ask your question..."):
    query_to_process = text_input

# Process the query if one was received from either voice or text
if query_to_process:
    st.session_state.messages.append(("You", query_to_process))
    with st.spinner("Thinking and routing your question..."):
        try:
            chain_input_for_lcel = {"input": query_to_process}
            full_result = final_chain.invoke(chain_input_for_lcel)

            answer_text = full_result.get("result") or full_result.get("text") or str(full_result)
            st.session_state.messages.append(("Gemini", answer_text))

            chat_history.append({"sender": "You", "question": query_to_process})
            chat_history.append({"sender": "Gemini", "question": query_to_process, "text": answer_text})
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, indent=2)

            if st.session_state.get("history_vectordb"):
                history_doc_content = f"Previous Question: {query_to_process}\nPrevious Answer: {answer_text}"
                new_history_document_for_vdb = Document(page_content=history_doc_content)
                st.session_state.history_vectordb.add_documents([new_history_document_for_vdb])

        except Exception as e:
            st.error(f"Error processing your question: {e}")
            error_message = f"Sorry, I encountered an error trying to answer your question.\n\n**Details:** {str(e)}"
            st.session_state.messages.append(("Gemini", error_message))
            chat_history.append({"sender": "You", "question": query_to_process})
            chat_history.append({"sender": "Gemini", "question": query_to_process, "text": f"Error: {str(e)}"})
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, indent=2)
    st.rerun()

# Display history
st.markdown("---")
st.subheader("Chat History")
for sender, msg_text in st.session_state.messages:
    if sender == "You":
        st.markdown(f"<div style='text-align:right;background-color:#DCF8C6;padding:10px;border-radius:7px;margin-bottom:5px;'><b>You:</b> {msg_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;background-color:#F0F0F0;padding:10px;border-radius:7px;margin-bottom:5px;'><b>Gemini:</b> {msg_text}</div>", unsafe_allow_html=True)

if st.button("Clear Chat Session and History File"):
    st.session_state.messages = []
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            os.remove(CHAT_HISTORY_FILE)
            st.success(f"Cleared {CHAT_HISTORY_FILE}")
        except Exception as e:
            st.error(f"Could not clear {CHAT_HISTORY_FILE}: {e}")
    st.rerun()
