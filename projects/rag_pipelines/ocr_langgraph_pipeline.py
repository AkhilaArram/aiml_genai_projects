# Pdf with text and Images -OCR
from dotenv import load_dotenv
import os
import re
from typing import TypedDict, Annotated, Sequence, List, Optional
from operator import add as add_messages
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

# Load env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. Q&A functionality will require it.")
    
# ======= PDF PATH =========
pdf_path = r"C:\Users\AKHILA\OneDrive\Desktop\Hyderabad.pdf" 

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

# ======= State Definition =========
class PDFRAGState(TypedDict):
    pdf_path: str
    extracted_text_docs: List[Document]
    extracted_images_with_pages: List[dict]
    ocr_results: List[dict]
    docs_for_chunking: List[Document]
    chunked_docs: Optional[List[Document]] # Will hold all final chunks (text + OCR)
    retriever: Optional[BaseRetriever]
    user_query: Optional[str]
    answer: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ======= Node 1: Text Extraction =========
def extract_text_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- Text Extraction Node ---")
    try:
        loader = UnstructuredPDFLoader(
            state['pdf_path'], mode="elements", strategy="hi_res", pdf_image_converter="poppler"
        )
        docs = loader.load()
        print(f"Extracted {len(docs)} text elements from PDF.")
        return {"extracted_text_docs": docs}
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return {"extracted_text_docs": []}

# ======= Node 2: Image Extraction =========
def extract_images_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- Image Extraction Node ---")
    images_data = []
    overall_image_count = 0
    try:
        doc = fitz.open(state['pdf_path'])
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    images_data.append({
                        "image_obj": pil_image,
                        "page_number": page_num + 1,
                        "overall_image_index": overall_image_count + 1
                    })
                    overall_image_count += 1
                except Exception as e:
                    print(f"Warning: Could not extract image xref {xref} on page {page_num+1}: {e}")
        doc.close()
        print(f"Extracted {len(images_data)} images from PDF.")
    except Exception as e:
        print(f"Error during image extraction: {e}")
    return {"extracted_images_with_pages": images_data}

# ======= Node 3: OCR Node =========
def ocr_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- OCR Node ---")
    ocr_results_list = []
    images_to_process_details = state.get('extracted_images_with_pages', [])

    if not images_to_process_details:
        print("No images found to OCR.")
        return {"ocr_results": []}

    print(f"Processing {len(images_to_process_details)} extracted images with OCR.")
    for img_detail in images_to_process_details:
        try:
            # Convert to grayscale for better OCR performance
            pil_image_to_ocr = img_detail["image_obj"].convert("L")
    
            text = pytesseract.image_to_string(pil_image_to_ocr, lang='eng') 
            print(f"OCR attempt for Image {img_detail['overall_image_index']} (Page: {img_detail['page_number']}, Length: {len(text)}). Content snippet: '{text[:70].strip().replace('\n', ' ')}'")
            ocr_results_list.append({
                "text": text,
                "page_number": img_detail["page_number"],
                "original_image_index": img_detail["overall_image_index"]
            })
        except Exception as e:
            print(f"Warning: OCR failed for image {img_detail['overall_image_index']} (Page: {img_detail['page_number']}): {e}")
            ocr_results_list.append({"text": "", "page_number": img_detail["page_number"], "original_image_index": img_detail["overall_image_index"]})
    return {"ocr_results": ocr_results_list}

# ======= Node 4: Restructure/Merge Node =========
def restructure_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- Restructure Node ---")
    # Start with text documents. Make a copy to avoid modifying original state's list
    all_docs = list(state.get('extracted_text_docs', []))
    
    if state.get('ocr_results'):
        for ocr_item in state['ocr_results']:
            ocr_text_stripped = ocr_item["text"].strip()
            if ocr_text_stripped:
        
                ocr_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                ocr_parts = ocr_text_splitter.split_text(ocr_text_stripped)

                for part_index, part_text in enumerate(ocr_parts):
                    stripped_part_text = part_text.strip()
                    if stripped_part_text:
                        page_content_for_embedding = stripped_part_text
                        ocr_metadata_prefix = (
                            f"IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_{ocr_item['original_image_index']}"
                            f"_PAGE_{ocr_item['page_number']}: (Part {part_index + 1})"
                        )
                        all_docs.append(Document(
                            page_content=page_content_for_embedding,
                            metadata={
                                "source_type": "image_ocr",
                                "page_number": ocr_item["page_number"],
                                "image_index": ocr_item["original_image_index"],
                                "ocr_prefix": ocr_metadata_prefix
                            },
                        ))
    print(f"Prepared {len(all_docs)} documents (text + OCR) for chunking.")
    return {"docs_for_chunking": all_docs}

# ======= Node 5: Chunk Text Node =========
def chunk_text_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- Chunk Text Node ---")
    docs_to_chunk = state.get('docs_for_chunking', [])
    if not docs_to_chunk:
        print("No documents to chunk.")
        return {"chunked_docs": []}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(docs_to_chunk)
    print(f"Documents split into {len(chunked_docs)} chunks.")
    return {"chunked_docs": chunked_docs}

# ======= Node 6: Create Retriever Node =========
def create_retriever_node(state: PDFRAGState) -> PDFRAGState:
    print("\n--- Create Retriever Node (with FAISS Caching) ---")
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not set. Cannot create embeddings or retriever.")
        return {"retriever": None}

    # Use a unique index path based on PDF to prevent conflicts if you process multiple PDFs
    pdf_filename = os.path.basename(state['pdf_path']).replace('.', '_')
    faiss_index_path = f"faiss_index_cache_{pdf_filename}"

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        if os.path.exists(faiss_index_path) and os.listdir(faiss_index_path):
            print(f"Loading FAISS index from {faiss_index_path}...")
            # built with older versions or potentially from untrusted sources. Use with caution in production environments.
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        else:
            print(f"No existing FAISS index found at {faiss_index_path}. Building and saving new index...")
            if not state.get('chunked_docs'):
                print("Error: No chunked documents available to build FAISS index.")
                return {"retriever": None}
            vectorstore = FAISS.from_documents(documents=state['chunked_docs'], embedding=embeddings)
            os.makedirs(faiss_index_path, exist_ok=True)
            vectorstore.save_local(faiss_index_path)
            print(f"FAISS index built and saved to {faiss_index_path}.")

        # Retrieve top 15 relevant docs for general queries
        retriever_instance = vectorstore.as_retriever(search_kwargs={"k": 30})
        return {"retriever": retriever_instance}
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return {"retriever": None}

# ======= Graph Building =========
graph_builder = StateGraph(PDFRAGState)

# Add nodes
graph_builder.add_node("extract_text", extract_text_node)
graph_builder.add_node("extract_images", extract_images_node)
graph_builder.add_node("ocr", ocr_node)
graph_builder.add_node("restructure", restructure_node)
graph_builder.add_node("chunk_text", chunk_text_node)
graph_builder.add_node("create_retriever", create_retriever_node)
# adding edges
graph_builder.add_edge(START, "extract_text")
graph_builder.add_edge(START, "extract_images")
graph_builder.add_edge("extract_images", "ocr")
graph_builder.add_edge("extract_text", "restructure")
graph_builder.add_edge("ocr", "restructure")
graph_builder.add_edge("restructure", "chunk_text")

# Chunked documents are used to create/load the retriever
graph_builder.add_edge("chunk_text", "create_retriever")
graph_builder.add_edge("create_retriever", END)

# Compile the graph
pdf_processing_graph = graph_builder.compile()
# pdf_processing_graph.get_graph().draw_mermaid_png(output_file_path="chat_visualization.png")
# print("Graph visualization saved to chat_visualization.png")

# ======= Run PDF Processing Graph =========
initial_state: PDFRAGState = {
    "pdf_path": pdf_path,
    "extracted_text_docs": [],
    "extracted_images_with_pages": [],
    "ocr_results": [],
    "docs_for_chunking": [],
    "chunked_docs": None, 
    "retriever": None,   
    "user_query": None,
    "answer": None,
    "messages": []
}
print("\n===== Starting PDF Processing and Indexing Graph =====")
try:
    # Invoke the graph to process the PDF and build the index
    final_processed_state = pdf_processing_graph.invoke(initial_state)
    print("\n===== PDF Processing and Indexing Complete =====")
except Exception as e:
    print(f"\n===== PDF Processing and Indexing FAILED: {e} =====")
    final_processed_state = {} 

# ======= Q&A Loop using the created retriever =========
if final_processed_state and final_processed_state.get('retriever') and final_processed_state.get('chunked_docs'):
    retriever = final_processed_state['retriever']
    # Get direct access to the FAISS vectorstore from the retriever
    vectorstore = retriever.vectorstore
    all_processed_chunks = final_processed_state['chunked_docs'] # Access the list of all chunks for direct lookup

    if not GOOGLE_API_KEY:
        print("\nGOOGLE_API_KEY is required for the Q&A chatbot. Exiting Q&A.")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        print("\n--- Q&A Chatbot Ready (type 'exit' to quit) ---")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            if not user_query.strip():
                continue

            relevant_docs: List[Document] = []
            # Check for specific "text in image X" query
            image_text_query_match = re.search(r"(?:what text is in|text from)\s+image\s*(\d+)", user_query, re.IGNORECASE)
            
            is_specific_image_text_query = False
            target_image_index = None

            if image_text_query_match:
                target_image_index = int(image_text_query_match.group(1))
                query_lower = user_query.lower()
                text_keywords = ["text", "content", "ocr", "words", "caption", "label"]
                if any(txt_kw in query_lower for txt_kw in text_keywords):
                    is_specific_image_text_query = True

            if is_specific_image_text_query:
                print(f"DEBUG: Detected specific query for text in image {target_image_index}. Attempting direct lookup for OCR text.")
                ocr_documents_found_by_direct_lookup = []
                for doc in all_processed_chunks: # Iterate through all chunks
                    if doc.metadata.get('source_type') == 'image_ocr' and \
                       doc.metadata.get('image_index') == target_image_index:
                        ocr_documents_found_by_direct_lookup.append(doc)
                
                if ocr_documents_found_by_direct_lookup:
                    relevant_docs = ocr_documents_found_by_direct_lookup
                    print(f"DEBUG: Found {len(relevant_docs)} specific OCR documents for image {target_image_index} via direct lookup.")
                else:
                    print(f"DEBUG: No specific OCR documents found for image {target_image_index} via direct lookup.")
                    # In this case, relevant_docs remains empty, and the LLM prompt will handle the "not found" scenario
            else:
                # For all other queries (general or "what does image X show" without "text in")
                print("DEBUG: Performing general retrieval.")
                try:
                    # Use the general retriever for semantic search
                    relevant_docs = retriever.invoke(user_query)
                except Exception as e:
                    print(f"Error during general retrieval: {e}")
                    relevant_docs = []

            # --- Debugging: Print retrieved documents ---
            print(f"\nDEBUG: Top {len(relevant_docs)} retrieved documents for query '{user_query}':")
            if relevant_docs:
                for i_debug, doc_debug in enumerate(relevant_docs):
                    page_num_debug = str(doc_debug.metadata.get('page_number', 'N/A'))
                    source_type_debug = doc_debug.metadata.get('source_type', 'text')
                    image_idx_debug = doc_debug.metadata.get('image_index', '')
                    category_debug = doc_debug.metadata.get('category', '')

                    type_info_debug = f"Type: {source_type_debug}"
                    if source_type_debug == 'image_ocr':
                        type_info_debug += f" (Image Index: {image_idx_debug})"
                        ocr_prefix_debug = doc_debug.metadata.get('ocr_prefix', '')
                        content_snippet_debug = f"{ocr_prefix_debug} {doc_debug.page_content[:200].strip().replace('\n', ' ')}..." if ocr_prefix_debug else f"{doc_debug.page_content[:200].strip().replace('\n', ' ')}..."
                    elif category_debug:
                        type_info_debug += f" (Category: {category_debug})"
                        content_snippet_debug = f"{doc_debug.page_content[:200].strip().replace('\n', ' ')}..."
                    else:
                        content_snippet_debug = f"{doc_debug.page_content[:200].strip().replace('\n', ' ')}..."

                    print(f"  Doc {i_debug+1} (Page: {page_num_debug}, {type_info_debug}): {content_snippet_debug}")
            else:
                print("  No documents retrieved.")
            print("--- End of raw retrieved docs for debugging ---")

            # Prepare context for the LLM
            context_for_llm_parts = []
            if not relevant_docs and is_specific_image_text_query:
                # This condition covers the case where a specific OCR query was made,
                # but no relevant OCR documents were found by direct lookup.
                context_for_llm_parts.append(
                    f"No OCR text found for Image {target_image_index} in the provided documents."
                )
            else:
                for i, doc in enumerate(relevant_docs):
                    page_number_str = str(doc.metadata.get('page_number', 'N/A'))
                    doc_type_str = "Unknown Type"
                    doc_content_for_llm = doc.page_content

                    if doc.metadata.get('source_type') == 'image_ocr':
                        image_index = doc.metadata.get('image_index', 'N/A')
                        doc_type_str = f"OCR from Image {image_index}"
                        ocr_prefix_from_meta = doc.metadata.get('ocr_prefix', '')
                        if ocr_prefix_from_meta:
                            doc_content_for_llm = f"{ocr_prefix_from_meta} {doc.page_content}"
                        else:
                            doc_content_for_llm = f"IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_{image_index}_PAGE_{page_number_str}: {doc.page_content}"
                    elif doc.metadata.get('category'):
                        doc_type_str = doc.metadata.get('category')

                    context_for_llm_parts.append(
                        f"Document {i+1} (Page: {page_number_str}, Type: {doc_type_str}):\n{doc_content_for_llm}"
                    )

            # Check if any context was generated at all
            if not context_for_llm_parts:
                print(f"\n--------------------------------------------------")
                print(f"You: {user_query}")
                print(f"Gemini: Based on the provided document, I cannot answer that question as no relevant context was found or extracted for your query.")
                print(f"--------------------------------------------------")
                continue # Skip LLM call if no context

            context_str = "\n\n".join(context_for_llm_parts)

            # Refined System Prompt (focuses on the consistent prefix and handling image text)
            system_message_content = """You are an assistant that answers questions based EXCLUSIVELY on the provided context from a PDF document.
            - Your answer must be derived solely from the text in the 'Context from the PDF document' section.
            - The context may contain text directly from the PDF and text extracted from images. Text extracted from images will be clearly prefixed with "IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_X_PAGE_Y:". Treat this image-derived text as part of the overall document context for answering any relevant question.
            - Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

            - When answering a question about a specific image (e.g., "what does image 5 show?", "describe image 3", or "what text is in image 1?"):
                - Look for document parts in the context that begin with "IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_X_PAGE_Y:" where X is the image number from the user's question.
                - If the question specifically asks for "text" from an image (e.g., "what text is in image X?", "show me the text from image X"):
                    - If a corresponding "IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_X_PAGE_Y:" part IS found in the context for the requested image number:
                        - Your answer MUST be the exact text content following that prefix for the specified image. For example, if the context has "IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_1_PAGE_1: (Part 1) Hello world", and the question is "What text is in image 1?", your answer must be "(Part 1) Hello world".
                        - Do not summarize or rephrase this OCR text.
                    - If a corresponding "IMAGE_DESCRIPTION_OCR_IMAGE_INDEX_X_PAGE_Y:" part IS NOT found in the context for the requested image number:
                        - You MUST state: "The OCR text for Image X was not found in the retrieved context." (Replace X with the image number from the user's question).
                - For other questions about an image (e.g., "what does image 5 show?"), prioritize information from these "IMAGE_DESCRIPTION_OCR..." parts for your answer, but you can synthesize a descriptive answer.
            - If, after considering all provided context (including text from images), the information to answer the question is not present, you MUST state: 'Based on the provided document, I cannot answer that question.' or 'The document does not contain information on this topic.'
            - Do not attempt to answer if the information is not in the context.
            - Be concise and directly answer the question using only the provided information, following all the rules above."""

            human_message_content = f"""Context from the PDF document:
---
{context_str}
---
User Question: {user_query}

Answer:"""
            response = llm.invoke([
                    SystemMessage(content=system_message_content),
                    HumanMessage(content=human_message_content)
                ])
            # --- Display the response ---
            print(f"\n--------------------------------------------------")
            # Print the context snippets that were sent to the LLM
            for snippet_info_str in context_for_llm_parts:
                print(snippet_info_str)
            print(f"--------------------------------------------------")
            print(f"You: {user_query}")
            print(f"Gemini: {response.content.strip()}")
            print(f"--------------------------------------------------")
else:
    print("\nRetriever not created, or no chunked documents available. Q&A chatbot cannot start.")