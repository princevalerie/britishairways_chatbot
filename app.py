import os
import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API key tidak ditemukan. Mohon set GEMINI_API_KEY di file .env")
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

# URLs yang akan di-scrape
urls = [
    "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
]

# Scraping data dan memproses chunk
all_chunks = []
for url in urls:
    st.write(f"Scraping: {url}")
    full_text = scrape_text(url)
    if full_text:
        chunks = chunk_text(full_text, chunk_size=1000)
        all_chunks.extend(chunks)
st.write(f"Total chunk yang dihasilkan: {len(all_chunks)}")

# Streamlit UI
st.title("Generative AI Chatbot dengan RAG")
st.write("Tanyakan apapun tentang British Airways berdasarkan data review.")

# Input dari pengguna
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Apa yang ingin Anda ketahui?", placeholder="Tulis pertanyaan di sini...")
    submit_button = st.form_submit_button("Submit")

if submit_button and user_input:
    with st.spinner("Mencari informasi..."):
        relevant_chunks = retrieve_relevant_chunks(user_input, all_chunks, top_n=3)
        context = "\n\n".join(relevant_chunks)
        
        messages = [
            {"role": "system", "parts": [{"text": "Gunakan data berikut untuk menjawab pertanyaan pengguna:"}]},
            {"role": "user", "parts": [{"text": f"{context}\n\nPertanyaan: {user_input}"}]}
        ]
        
        chat_session = model.start_chat(history=messages)
        response = chat_session.send_message(user_input)
        
        if response and response.text:
            st.write("### Jawaban:")
            st.markdown(response.text)
