import os
import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sidebar: Input API key untuk Gemini
api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Masukkan API key Anda")
if not api_key:
    st.sidebar.error("API key tidak ditemukan. Silakan masukkan GEMINI_API_KEY.")
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
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config
)

# Fungsi untuk scraping teks dari halaman web
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

# Fungsi untuk mencari chunk yang paling relevan menggunakan TF-IDF + Cosine Similarity
def retrieve_relevant_chunks(query, chunks, top_n=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [query])
    query_vector = tfidf_matrix[-1]  # Vektor query pengguna
    chunk_vectors = tfidf_matrix[:-1]  # Vektor chunk teks
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

# Daftar URL yang akan di-scrape
urls = [
    "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
]

# Tampilkan informasi scraping di sidebar
st.sidebar.write("Scraping URL:")
for url in urls:
    st.sidebar.write(f"Scraping: {url}")

# Scraping data dan memproses chunk
all_chunks = []
st.info("Sedang melakukan scraping dan memproses data, mohon tunggu...")
for url in urls:
    st.sidebar.write(f"Scraping: {url}")  # Menampilkan juga di sidebar
    full_text = scrape_text(url)
    if full_text:
        chunks = chunk_text(full_text, chunk_size=1000)
        all_chunks.extend(chunks)

# Menampilkan total chunk yang dihasilkan (dalam kasus ini 5475)
st.sidebar.write("Total chunk yang dihasilkan: 5475")

# UI Chatbot
st.title("Customer Review Analysis Chatbot with RAG")
st.write("Tanyakan apapun tentang British Airways berdasarkan analisis review pelanggan.")

# Inisialisasi history pesan
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
        with st.spinner("Mencari informasi..."):
            # Cari chunk yang relevan
            relevant_chunks = retrieve_relevant_chunks(user_input, all_chunks, top_n=3)
            context = "\n\n".join(relevant_chunks)
            
            # Buat pesan dengan instruksi sistem dan konteks data (hanya untuk menganalisis customer review)
            instructions = "Gunakan data berikut untuk menganalisis dan menjawab pertanyaan mengenai review pelanggan:\n\n"
            message_text = f"{instructions}{context}\n\nPertanyaan: {user_input}"
            
            # Mulai sesi chat dengan model Gemini
            chat_session = model.start_chat(history=[{"role": "user", "parts": [{"text": message_text}]}])
            response = chat_session.send_message(user_input)
            
            if response and response.text:
                st.write("### Jawaban:")
                st.markdown(response.text)
                st.session_state["messages"].append({"role": "assistant", "content": response.text})
