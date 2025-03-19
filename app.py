# app.py
import os
import time
from typing import List, Dict

import streamlit as st
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Konfigurasi
load_dotenv()
st.set_page_config(page_title="BA Reviews AI Analyst", page_icon="✈️", layout="wide")

class Config:
    GEMINI_MODEL = "models/gemini-1.5-pro-latest"
    EMBEDDING_MODEL = "models/embedding-001"
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 500
    SEARCH_KWARGS = {"k": 7}

class FireCrawlManager:
    def __init__(self):
        self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    def crawl_reviews(self, url: str) -> List[Dict]:
        try:
            crawl_result = self.app.crawl_url(
                url,
                params={
                    'limit': 5,
                    'scrapeOptions': {
                        'formats': ['markdown'],
                        'extractorOptions': {
                            'reviews': {
                                'selector': '.review-list .review-article',
                                'type': 'list',
                                'fields': {
                                    'rating': '.rating-10',
                                    'title': '.text_header',
                                    'content': '.text_content',
                                    'date': 'time[itemprop="datePublished"]',
                                    'author': '.userStatusWrapper',
                                    'metadata': {
                                        'selector': '.review-stats',
                                        'fields': {
                                            'seat_type': '.review-stats__seatType',
                                            'route': '.review-stats__route',
                                            'class': '.review-stats__cabin'
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            )
            
            processed_reviews = []
            for page in crawl_result:
                if page['data'] and 'reviews' in page['data']:
                    for review in page['data']['reviews']:
                        processed = self._process_review(review, page['url'])
                        processed_reviews.append(processed)
            
            return processed_reviews
        
        except Exception as e:
            st.error(f"FireCrawl error: {str(e)}")
            return []

    def _process_review(self, review: Dict, url: str) -> Dict:
        return {
            'type': self._get_review_type(url),
            'rating': review.get('rating', 'N/A').strip(),
            'title': review.get('title', '').strip(),
            'content': review.get('content', '').strip(),
            'date': review.get('date', '').strip(),
            'author': review.get('author', 'Anonymous').strip(),
            'metadata': review.get('metadata', {}),
            'url': url
        }

    def _get_review_type(self, url: str) -> str:
        if 'seat-reviews' in url: return 'seat'
        if 'lounge-reviews' in url: return 'lounge'
        return 'airline'

class RAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def init_vector_store(self, reviews: List[Dict]) -> Chroma:
        documents = [
            f"Review {idx}:\nRating: {r['rating']}\nType: {r['type']}\nContent: {r['content']}"
            for idx, r in enumerate(reviews)
        ]
        
        chunks = self.text_splitter.create_documents(documents)
        
        return Chroma.from_documents(
            documents=chunks,
            embedding=GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
        )

class AnalystAssistant:
    def __init__(self, vector_store: Chroma):
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=Config.SEARCH_KWARGS
        )
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.4
        )
    
    def analyze_reviews(self, question: str) -> str:
        context = "\n\n".join([
            doc.page_content 
            for doc in self.retriever.get_relevant_documents(question)
        ])
        
        prompt = f"""Anda adalah analis profesional British Airways dengan akses ke data review terkini.
        Pertanyaan: {question}
        
        Konteks Review:
        {context}
        
        Format Jawaban:
        1. Identifikasi pola utama
        2. Sertakan statistik kuantitatif
        3. Berikan contoh spesifik dari review
        4. Rekomendasi perbaikan
        """
        
        return self.llm.invoke(prompt).content

# UI Components
def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Konfigurasi")
        st.write("Pastikan API key sudah diisi di file .env")
        st.markdown("""
            **Key Requirements:**
            - [FireCrawl API Key](https://firecrawl.dev)
            - [Gemini API Key](https://ai.google.dev)
            """)
        
        if st.button("🔄 Muat Ulang Data"):
            st.session_state.clear()
            st.rerun()

def display_analytics(reviews: List[Dict]):
    st.subheader("📊 Analisis Data")
    
    col1, col2 = st.columns(2)
    with col1:
        ratings = [int(r['rating'].split('/')[0]) for r in reviews if r['rating'].isdigit()]
        if ratings:
            st.metric("Rata-rata Rating", f"{sum(ratings)/len(ratings):.1f}/10")
    
    with col2:
        review_types = [r['type'] for r in reviews]
        st.write("Distribusi Jenis Review:", dict(zip(*np.unique(review_types, return_counts=True))))

# Main App
def main():
    setup_sidebar()
    st.title("✈️ British Airways Review AI Analyst")
    
    if 'reviews' not in st.session_state:
        st.session_state.reviews = []
    
    if not st.session_state.reviews:
        with st.status("🕷️ Crawling data review...", expanded=True) as status:
            crawler = FireCrawlManager()
            urls = [
                "https://www.airlinequality.com/airline-reviews/british-airways",
                "https://www.airlinequality.com/seat-reviews/british-airways",
                "https://www.airlinequality.com/lounge-reviews/british-airways"
            ]
            
            for url in urls:
                st.write(f"Memproses: {url}")
                st.session_state.reviews.extend(crawler.crawl_reviews(url))
                time.sleep(2)
            
            status.update(label="✅ Crawling selesai", state="complete")
    
    if st.session_state.reviews:
        display_analytics(st.session_state.reviews)
        
        rag = RAGSystem()
        vector_store = rag.init_vector_store(st.session_state.reviews)
        assistant = AnalystAssistant(vector_store)
        
        st.subheader("💬 Tanya Analis AI")
        if prompt := st.chat_input("Apa yang ingin Anda analisis?"):
            with st.spinner("🔍 Menganalisis..."):
                response = assistant.analyze_reviews(prompt)
                st.write(response)
    else:
        st.error("Gagal memuat data review")

if __name__ == "__main__":
    main()
