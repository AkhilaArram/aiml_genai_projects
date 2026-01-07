import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Load environment variables ---
load_dotenv(override=True)

# --- App Title ---
st.title("📄 Multi-User PDF Chatbot (Gemini + Neo4j)")


# --- Simulated login (can be replaced with real auth) ---
if "username" not in st.session_state:
    st.session_state.username = "default_user"

st.sidebar.info(f"Logged in as: **{st.session_state.username}**")

# --- Initialize Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Cached Neo4j connection ---
@st.cache_resource(show_spinner="🔗 Connecting to Neo4j...")
def get_neo4j_connection():
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
    NEO4J_USERNAME = st.secrets.get("NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

    if not all([GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        st.error("Missing Neo4j or Gemini credentials.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    db = Neo4jVector.from_documents(
        documents=[],  # Pass an empty list to initialize
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
       index_name=f"pdf_chunks_{st.session_state.username}", 
        node_label="PdfChunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    return db, embeddings

neo4j_db, embeddings = get_neo4j_connection()

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_name = uploaded_file.name
    user_id = st.session_state.username

    # Check if chunks already exist for this user and PDF
    st.spinner("Checking database for existing data...")
    result = neo4j_db.query(
        """
        MATCH (n:PdfChunk {pdf_name: $pdf_name, user_id: $user_id})
        RETURN count(n) > 0 AS exists
        """,
        params={"pdf_name": pdf_name, "user_id": user_id}
    )

    if not result[0]["exists"]:
        st.info(f"Indexing new PDF: {pdf_name} for user '{user_id}'...")

        with st.spinner("⏳ Processing PDF..."):
            temp_path = f"temp_{pdf_name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            for chunk in chunks:
                chunk.metadata["pdf_name"] = pdf_name
                chunk.metadata["user_id"] = user_id

            neo4j_db.add_documents(chunks)

            # Link chunks to a PDF node
            neo4j_db.query(
                """
                MERGE (pdf:PDF {name: $pdf_name, user_id: $user_id})
                WITH pdf
                MATCH (chunk:PdfChunk {pdf_name: $pdf_name, user_id: $user_id})
                MERGE (pdf)-[:HAS_CHUNK]->(chunk)
                """,
                params={"pdf_name": pdf_name, "user_id": user_id}
            )

            st.success(f"✅ Indexed and linked chunks for '{pdf_name}'")
            os.remove(temp_path)
    else:
        st.success(f"✅ Ready to chat with: {pdf_name}")

    # --- LLM Setup ---
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    # --- Filtered retriever ---
    retriever = neo4j_db.as_retriever(
        search_kwargs={"filter": {"pdf_name": pdf_name, "user_id": user_id}, "k": 5}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # --- Chat Form ---
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question...")
        send = st.form_submit_button("Ask")
        if send and user_input:
            with st.spinner("🤖 Thinking..."):
                # Use invoke for better compatibility with LCEL and future versions
                result = qa.invoke({"query": user_input})
                answer = result.get("result", "I could not find an answer in the document.")
            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Bot", answer))

    # --- Display Chat ---
    st.markdown("💬 Chat History")
    for i, (sender, msg) in enumerate(st.session_state.messages):
        if sender == "You":
            st.markdown(f"<div style='text-align:right; background:#DCF8C6; padding:8px; border-radius:8px; margin:5px'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background:#F1F0F0; padding:8px; border-radius:8px; margin:5px'><b>Bot:</b> {msg}</div>", unsafe_allow_html=True)

    # --- Clear Chat Button ---
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

else:
    st.info("📤 Please upload a PDF to begin.")