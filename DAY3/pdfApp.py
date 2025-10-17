import PyPDF2
import streamlit as st
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

llm = Ollama(model="gemma2:2b")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a PDF document and split it into pages
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        # Split text into chunks of roughly 1000 characters
        chunks.extend([text[i:i+1000] for i in range(0, len(text), 1000)])
    return chunks

# Import the SentenceTransformer model
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define sample statements
statements = ["Artificial intelligence is revolutionizing the world.",
              "Machine learning is a subset of AI."]

# Generate embeddings for the statements
embeddings = model.encode(statements)

# Print the embeddings
print(embeddings)



st.title('Chat with your PDF using LLaMA and RAG')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def create_embeddings(chunks):
    return embedding_model.encode(chunks)

def find_most_relevant_chunk(query, chunks, chunk_embeddings):
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    most_relevant_idx = np.argmax(similarities)
    return chunks[most_relevant_idx]


def get_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt)
    return response


if uploaded_file:
    # Load the PDF and create embeddings
    with st.spinner('Processing PDF...'):
        chunks = load_pdf(uploaded_file)
        chunk_embeddings = create_embeddings(chunks)
    st.success('PDF processed successfully!')
    
    # User input for query
    query = st.text_input("Ask your question about the PDF!")
    
    # Display the response
    if query:
        with st.spinner('Searching for an answer...'):
            relevant_chunk = find_most_relevant_chunk(query, chunks, chunk_embeddings)
            response = get_response(query, relevant_chunk)
        st.write(response)



