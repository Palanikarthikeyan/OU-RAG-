import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title=" Local RAG with Ollama", layout="wide")
st.title("RAG App using Ollama (Local LLM)")

# Load vectorstore from file
@st.cache_resource
def build_vectorstore(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Load vectorstore
with st.sidebar:
    st.header("Load Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        with open("uploaded.txt", "wb") as f:
            f.write(uploaded_file.read())
        filepath = "uploaded.txt"
    else:
        filepath = "docs/sample.txt"

vectorstore = build_vectorstore(filepath)

# Setup local LLM (Ollama)
llm = Ollama(model="gemma2:2b")  

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# User query
query = st.text_input(" Ask a question based on the document:")
if query:
    with st.spinner("Thinking..."):
        result = rag_chain({"query": query})

    st.subheader(" Answer")
    st.write(result["result"])

    st.subheader("Source Document(s)")
    for i, doc in enumerate(result["source_documents"]):
        st.markdown(f"**Chunk {i+1}:**")
        st.code(doc.page_content.strip())
