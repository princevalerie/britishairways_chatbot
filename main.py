import os
import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Sidebar: Input API key untuk Gemini
api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Masukkan API key Anda")
if not api_key:
    st.sidebar.error("API key tidak ditemukan. Silakan masukkan GEMINI_API_KEY.")
    st.stop()
else:
    genai.configure(api_key=api_key)

# Konfigurasi model Gemini
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

# Inisialisasi model embedding (SentenceTransformer)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fungsi untuk scraping teks dari halaman web dengan caching
@st.cache_data(show_spinner=False)
def scrape_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping {url}: {e}")
        return ""

# Fungsi untuk melakukan chunking teks
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Daftar URL yang akan di-scrape
urls = [
    "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
]

# Scraping data dan memproses chunk
all_chunks = []
st.info("Sedang melakukan scraping dan memproses data, mohon tunggu...")
for url in urls:
    full_text = scrape_text(url)
    if full_text:
        chunks = chunk_text(full_text, chunk_size=1000)
        all_chunks.extend(chunks)

# Precompute embedding untuk setiap chunk dan normalisasi (untuk cosine similarity)
@st.cache_data(show_spinner=False)
def compute_embeddings(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    return embeddings

chunk_embeddings = compute_embeddings(all_chunks)

# Membangun FAISS index dengan Inner Product (setelah normalisasi, inner product sama dengan cosine similarity)
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(chunk_embeddings)

# UI Chatbot
st.title("Customer Review Analysis Chatbot with RAG (Embedding + FAISS)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Halo, ada yang bisa saya bantu?"}]

# Tampilkan history chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input chat menggunakan st.chat_input
user_input = st.chat_input("Tulis pertanyaan Anda di sini...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi yang relevan..."):
            # Hitung embedding untuk query pengguna dan normalisasi
            query_embedding = embedding_model.encode([user_input], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Lakukan pencarian FAISS untuk menemukan top 3 chunk yang relevan
            k = 3
            distances, indices = index.search(query_embedding, k)
            relevant_chunks = [all_chunks[i] for i in indices[0]]
            context = "\n\n".join(relevant_chunks)
            
            # Buat pesan dengan instruksi dan konteks data, lalu gabungkan dengan pertanyaan
            instructions = "Gunakan data berikut untuk menganalisis dan menjawab pertanyaan mengenai review pelanggan:\n\n"
            message_text = f"{instructions}{context}\n\nPertanyaan: {user_input}"
            
            # Mulai sesi chat dengan model Gemini menggunakan role 'user'
            chat_session = model.start_chat(history=[{"role": "user", "parts": [{"text": message_text}]}])
            response = chat_session.send_message(user_input)
            
            if response and response.text:
                st.write("### Jawaban:")
                st.markdown(response.text)
                st.session_state["messages"].append({"role": "assistant", "content": response.text})
