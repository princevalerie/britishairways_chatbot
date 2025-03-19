# app.py
import os
import time
from typing import List, Dict

import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Konfigurasi awal
load_dotenv()
st.set_page_config(page_title="British Airways Reviews Analysis", page_icon="✈️", layout="wide")

class Config:
    """Kelas untuk menyimpan konfigurasi aplikasi"""
    GEMINI_MODEL = "gemini-2.0-flash-lite"
    EMBEDDING_MODEL = "models/embedding-001"
    SCRAPE_DELAY = 1
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SEARCH_KWARGS = {"k": 5}

class APIManager:
    """Mengelola API keys dan konfigurasi"""
    def __init__(self):
        self._init_session_state()
        
    def _init_session_state(self):
        session_defaults = {
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "firecrawl_api_key": os.getenv("FIRECRAWL_API_KEY", ""),
            "initialized": False,
            "scraped_content": [],
            "vector_store": None,
            "chat_history": []
        }
        
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @property
    def headers(self):
        return {
            'Authorization': f'Bearer {st.session_state.firecrawl_api_key}',
            'Content-Type': 'application/json'
        }

class DataScraper:
    """Handles data scraping from multiple sources"""
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        
    def _process_html(self, html: str, review_type: str) -> Dict:
        """Memproses HTML review menjadi dictionary terstruktur dengan deteksi field dinamis"""
        soup = BeautifulSoup(html, 'html.parser')
        review_data = {'type': review_type}
        
        # Ambil semua elemen dengan class yang mungkin berisi data
        all_elements = soup.select('[class]')
        
        # Iterasi semua elemen dan ekstrak data
        for element in all_elements:
            # Periksa apakah elemen memiliki teks
            if element.text and element.text.strip():
                # Gunakan class sebagai key, atau text pertama jika tidak ada class spesifik
                for class_name in element.get('class', []):
                    # Hindari class yang terlalu generik
                    if class_name not in ['item', 'review-list', 'text']:
                        # Bersihkan teks
                        text_content = element.text.strip()
                        
                        # Cek apakah ini adalah teks yang bermakna
                        if len(text_content) > 1:  # Minimal 2 karakter untuk menghindari simbol
                            field_name = class_name.replace('-', '_')
                            review_data[field_name] = text_content
        
        # Tambahkan elemen khusus yang mungkin penting
        # Rating
        rating_element = soup.select_one('.rating-10 .rating, .rating')
        if rating_element:
            review_data['rating'] = rating_element.text.strip()
        
        # Review text - ini biasanya konten utama
        text_content = soup.select_one('.text_content')
        if text_content:
            review_data['review_text'] = text_content.text.strip()
            
        # Rekomendasi
        recommend = soup.select_one('.recommend-yes, .recommend-no')
        if recommend:
            review_data['recommendation'] = recommend.text.strip()
        
        # Tambahkan full text untuk pencarian
        review_data['full_text'] = soup.get_text(separator=' ', strip=True)
        
        # Tambahkan deteksi field dengan parsing label
        for label in soup.select('strong, b, .label'):
            label_text = label.text.strip().lower()
            if label_text and ':' in label_text:
                # Ini mungkin label field
                field_name = label_text.split(':')[0].strip().replace(' ', '_')
                next_element = label.next_sibling
                if next_element and isinstance(next_element, str) and next_element.strip():
                    review_data[field_name] = next_element.strip()
        
        return review_data

    def scrape_with_firecrawl(self, url: str) -> List[Dict]:
        """Menggunakan FireCrawl API untuk scraping"""
        try:
            payload = {
                "url": url,
                "selectors": {
                    "reviews": {
                        "selector": ".review-list .item",
                        "type": "list",
                        "properties": {"full_html": {"selector": ".", "type": "html"}}
                    }
                }
            }
            
            response = requests.post(
                "https://api.firecrawl.dev/scrape",
                headers=self.api_manager.headers,
                json=payload
            )
            response.raise_for_status()
            
            return [
                self._process_html(review['full_html'], self._get_review_type(url))
                for review in response.json().get('reviews', [])
            ]
        except Exception as e:
            st.warning(f"FireCrawl error: {str(e)}")
            return []

    def scrape_with_bs(self, url: str) -> List[Dict]:
        """Fallback scraping dengan BeautifulSoup"""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return [
                self._process_html(str(item), self._get_review_type(url))
                for item in soup.select('.review-list .item')
            ]
        except Exception as e:
            st.error(f"Scraping error: {str(e)}")
            return []

    def _get_review_type(self, url: str) -> str:
        """Mendeteksi tipe review dari URL"""
        if 'seat-reviews' in url: return 'seat'
        if 'lounge-reviews' in url: return 'lounge'
        return 'airline'

class RAGSystem:
    """Mengelola sistem RAG"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
    def initialize(self, scraped_data: List[Dict]) -> FAISS:
        """Inisialisasi sistem vektor"""
        documents = [
            "\n".join(f"{k}: {v}" for k, v in review.items() if v)
            for review in scraped_data
        ]
        
        chunks = self.text_splitter.create_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=self.api_key
        )
        
        return FAISS.from_documents(chunks, embeddings)

class ChatInterface:
    """Mengelola antarmuka chat"""
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs=Config.SEARCH_KWARGS)
        
        self.prompt_template = PromptTemplate.from_template("""
            Anda adalah analis data profesional untuk ulasan British Airways. 
            Jawab pertanyaan berikut berdasarkan konteks:
            
            Konteks: {context}
            
            Pertanyaan: {question}
            
            Jawaban:
        """)
        
        self.model = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=st.session_state.gemini_api_key,
            temperature=0.2
        )
    
    def generate_response(self, question: str) -> str:
        """Generate jawaban RAG-based"""
        context = "\n\n".join([
            doc.page_content 
            for doc in self.retriever.get_relevant_documents(question)
        ])
        
        return self.model.invoke(
            self.prompt_template.format(context=context, question=question)
        ).content

# UI Components
def setup_sidebar():
    """Menyiapkan sidebar aplikasi"""
    with st.sidebar:
        st.title("Pengaturan API")
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password"
        )
        
        st.session_state.firecrawl_api_key = st.text_input(
            "FireCrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password"
        )
        
        if st.button("Simpan API Keys"):
            st.session_state.initialized = False
            st.rerun()
        
        st.markdown("---")
        st.title("Tentang Aplikasi")
        st.write("Aplikasi ini menganalisis ulasan pelanggan British Airways menggunakan teknologi AI Gemini.")

def display_stats():
    """Menampilkan statistik data"""
    col1, col2, col3 = st.columns(3)
    counts = {
        'airline': sum(1 for r in st.session_state.scraped_content if r['type'] == 'airline'),
        'seat': sum(1 for r in st.session_state.scraped_content if r['type'] == 'seat'),
        'lounge': sum(1 for r in st.session_state.scraped_content if r['type'] == 'lounge')
    }
    
    with col1: st.metric("Ulasan Maskapai", counts['airline'])
    with col2: st.metric("Ulasan Kursi", counts['seat'])
    with col3: st.metric("Ulasan Lounge", counts['lounge'])

# Main App Flow
def main():
    """Alur utama aplikasi"""
    api_manager = APIManager()
    setup_sidebar()
    
    if not st.session_state.gemini_api_key:
        st.error("Harap masukkan Gemini API Key di sidebar")
        return

    st.title("Analisis Ulasan British Airways")
    
    if not st.session_state.initialized:
        initialize_application(api_manager)
    else:
        run_application()

def initialize_application(api_manager: APIManager):
    """Inisialisasi awal aplikasi"""
    st.write("Aplikasi ini membutuhkan inisialisasi awal...")
    
    if st.button("Mulai Analisis"):
        with st.status("Memproses data..."):
            urls = [
                "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
                "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
                "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
            ]
            
            scraper = DataScraper(api_manager)
            scraped_data = []
            
            for url in urls:
                data = scraper.scrape_with_firecrawl(url) or scraper.scrape_with_bs(url)
                scraped_data.extend(data)
                time.sleep(Config.SCRAPE_DELAY)
            
            if scraped_data:
                st.session_state.scraped_content = scraped_data
                st.session_state.vector_store = RAGSystem(
                    st.session_state.gemini_api_key
                ).initialize(scraped_data)
                st.session_state.initialized = True
                st.rerun()
            else:
                st.error("Gagal mendapatkan data. Silakan coba lagi.")

def run_application():
    """Menjalankan aplikasi utama"""
    display_stats()
    
    if st.session_state.scraped_content:
        with st.expander("Kolom Data yang Terdeteksi"):
            fields = set().union(*st.session_state.scraped_content)
            st.write(", ".join(sorted(fields)))

    st.header("Tanya Tentang Ulasan")
    if prompt := st.chat_input("Apa yang ingin Anda ketahui?"):
        handle_chat_input(prompt)

def handle_chat_input(prompt: str):
    """Menangani input chat pengguna"""
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.spinner("Menganalisis..."):
        chat_interface = ChatInterface(st.session_state.vector_store)
        response = chat_interface.generate_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    display_chat_history()

def display_chat_history():
    """Menampilkan riwayat percakapan"""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if __name__ == "__main__":
    main()
