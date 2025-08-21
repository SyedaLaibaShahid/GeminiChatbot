from dotenv import load_dotenv
import os
import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# ==============================
# Configure Gemini API
# ==============================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found! Please check your .env file.")

genai.api_key = api_key
client = genai.Client()

# ==============================
# Embedding Model for PDFs
# ==============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def create_faiss_index(text):
    sentences = [s for s in text.split("\n") if s.strip()]
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, sentences

def retrieve_context(query, index, sentences, k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return "\n".join([sentences[i] for i in I[0]])

# ==============================
# Gemini API Call
# ==============================
def gemini_answer(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.replace("*", "").strip()
        return text
    except Exception as e:
        return f"Gemini API error: {e}"

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Gemini AI Assistant", layout="wide")

if "messages_history" not in st.session_state:
    st.session_state.messages_history = []

if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_saved" not in st.session_state:
    st.session_state.chat_saved = False

# ------------------------------
# Sidebar - Chat history and New Chat
# ------------------------------
with st.sidebar:
    if st.button("➕ New Chat"):
        if st.session_state.messages and not st.session_state.chat_saved:
            if st.session_state.current_chat_index is not None:
                st.session_state.messages_history[st.session_state.current_chat_index] = st.session_state.messages.copy()
            else:
                st.session_state.messages_history.append(st.session_state.messages.copy())
            st.session_state.chat_saved = True  

        st.session_state.messages = []
        st.session_state.current_chat_index = None
        st.session_state.chat_saved = False
        if "pdf_index" in st.session_state:
            del st.session_state.pdf_index
        if "pdf_sentences" in st.session_state:
            del st.session_state.pdf_sentences
        st.rerun()

    st.header("Chats")
    for i, chat in enumerate(st.session_state.messages_history):
        if chat:
            title = chat[0]["content"][:30] + ("..." if len(chat[0]["content"]) > 30 else "")
            if st.button(title, key=f"chat_{i}"):
                # Save current chat once
                if st.session_state.messages and not st.session_state.chat_saved:
                    if st.session_state.current_chat_index is not None:
                        st.session_state.messages_history[st.session_state.current_chat_index] = st.session_state.messages.copy()
                    else:
                        st.session_state.messages_history.append(st.session_state.messages.copy())
                    st.session_state.chat_saved = True

                st.session_state.messages = chat.copy()
                st.session_state.current_chat_index = i
                st.session_state.chat_saved = True  
                st.rerun()

# ------------------------------
# Main Chat Interface
# ------------------------------
st.title("Gemini AI Assistant")
st.write("Ask anything via text or PDFs!")

# Initialize session state for current chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in current chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# Chat input
# ------------------------------
if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Optional PDF context
    if "pdf_index" in st.session_state:
        context = retrieve_context(prompt, st.session_state.pdf_index, st.session_state.pdf_sentences)
        prompt_with_context = f"Context:\n{context}\n\nQuestion: {prompt}"
    else:
        prompt_with_context = prompt

    answer = gemini_answer(prompt_with_context)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

# ------------------------------
# PDF Upload under prompt 
# ------------------------------
with st.expander("📎 Upload PDF (optional)", expanded=False):
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
    if pdf_file:
        pdf_text = load_pdf(pdf_file)
        st.success("PDF Loaded!")
        pdf_index, pdf_sentences = create_faiss_index(pdf_text)
        st.session_state.pdf_index = pdf_index
        st.session_state.pdf_sentences = pdf_sentences
