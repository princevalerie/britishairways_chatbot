import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Set up the page
st.set_page_config(
    page_title="British Airways Reviews Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for API keys and data
def initialize_session_state():
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    if "firecrawl_api_key" not in st.session_state:
        st.session_state.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.scraped_content = []
        st.session_state.vector_store = None
        st.session_state.chat_history = []

initialize_session_state()

# Sidebar for API key input
with st.sidebar:
    st.title("Configuration")
    
    # API Key Inputs
    with st.expander("🔑 API Keys Setup", expanded=True):
        new_gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        new_firecrawl_key = st.text_input(
            "FireCrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password",
            help="Optional: Get from firecrawl.dev"
        )
        
        if st.button("💾 Save Keys"):
            st.session_state.gemini_api_key = new_gemini_key
            st.session_state.firecrawl_api_key = new_firecrawl_key
            st.session_state.initialized = False
            st.success("Keys saved! Please re-initialize the system.")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    st.title("About")
    st.markdown("""
    **British Airways Review Analyst** ✈️
    
    This AI-powered tool analyzes customer reviews from:
    - Airline Services
    - Seat Comfort
    - Lounge Experiences
    
    **How it works:**
    1. Scrapes review data from airlinequality.com
    2. Processes and indexes content using AI
    3. Answers questions using Gemini Pro
    4. Provides data-backed insights
    """)

# Main application
st.title("✈️ British Airways Review Analysis")
st.caption("Powered by Google Gemini AI • Real-time Web Scraping • RAG Architecture")

# Validate API keys
if not st.session_state.gemini_api_key:
    st.error("❌ Gemini API key is required. Please enter it in the sidebar.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=st.session_state.gemini_api_key)

# Scraping functions
def handle_scraping(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Try FireCrawl first if API key is available
        if st.session_state.firecrawl_api_key:
            firecrawl_response = scrape_with_firecrawl(url)
            if firecrawl_response:
                return firecrawl_response
        
        # Fallback to BeautifulSoup
        return scrape_with_beautifulsoup(url)
    
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return []

def scrape_with_firecrawl(url):
    try:
        headers = {'Authorization': f'Bearer {st.session_state.firecrawl_api_key}'}
        response = requests.post(
            "https://api.firecrawl.dev/v0/scrape",
            headers=headers,
            json={"url": url, "pageOptions": {"includes": ["mainContent"]}},
            timeout=15
        )
        response.raise_for_status()
        return [{"full_text": response.json()['data']['markdown']}]
    except Exception as e:
        st.warning(f"FireCrawl error: {str(e)}")
        return []

def scrape_with_beautifulsoup(url):
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = []
        
        for item in soup.select('.review-list .item'):
            review_text = ' '.join(item.stripped_strings)
            reviews.append({
                'full_text': review_text,
                'type': 'airline' if 'airline-reviews' in url else 
                        'seat' if 'seat-reviews' in url else 
                        'lounge'
            })
            time.sleep(0.1)  # Respectful scraping delay
        
        return reviews
    except Exception as e:
        st.error(f"BeautifulSoup error: {str(e)}")
        return []

# RAG System Initialization
def initialize_rag_system():
    urls = [
        "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
        "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
        "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
    ]
    
    with st.status("🚀 Initializing Analysis System...", expanded=True) as status:
        try:
            # Phase 1: Data Collection
            st.write("🔍 Collecting review data...")
            all_reviews = []
            for url in urls:
                reviews = handle_scraping(url)
                all_reviews.extend(reviews)
                time.sleep(1)  # Rate limiting
            
            if not all_reviews:
                st.error("No reviews collected. Check internet connection or scraping configuration.")
                return False
            
            # Phase 2: Data Processing
            st.write("⚙️ Processing content...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            documents = [review['full_text'] for review in all_reviews]
            chunks = text_splitter.create_documents(documents)
            
            # Phase 3: Vector Database
            st.write("🧠 Creating knowledge base...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=st.session_state.gemini_api_key
            )
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ System Ready!", state="complete", expanded=False)
            return True
        
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return False

# Chat Interface
def chat_interface():
    st.markdown("### 💬 Ask About British Airways Reviews")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("🔍 Analyzing reviews..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.get_relevant_documents(prompt)
                context = "\n\n".join([d.page_content for d in docs])
                
                # Generate response
                prompt_template = f"""
                As a British Airways customer experience analyst, answer this question:
                {prompt}
                
                Use this context from real customer reviews:
                {context[:30000]}  # Limit context size
                
                Provide a detailed, data-supported response. If unsure, say so.
                """
                
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",  # Model yang diminta
                    temperature=0.3,
                    google_api_key=st.session_state.gemini_api_key
                )
                
                response = model.invoke(prompt_template)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.content
                })
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main Application Flow
if not st.session_state.initialized:
    st.markdown("## 🛠 Initial Setup Required")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Step 1: API Configuration")
        st.markdown("""
        1. Get a [Gemini API Key](https://aistudio.google.com/app/apikey)
        2. (Optional) Get a [FireCrawl API Key](https://firecrawl.dev)
        3. Enter keys in the sidebar
        4. Click 'Initialize System' below
        """)
        
        if st.button("🚀 Initialize Analysis System"):
            if initialize_rag_system():
                st.session_state.initialized = True
                st.rerun()
    
    with col2:
        st.markdown("### Current Configuration")
        st.json({
            "Gemini Key Configured": bool(st.session_state.gemini_api_key),
            "FireCrawl Key Configured": bool(st.session_state.firecrawl_api_key),
            "System Status": "Initialized" if st.session_state.initialized else "Pending"
        })

else:
    # Show analysis interface
    st.success("✅ System Initialized - Start Asking Questions!")
    
    # Statistics Dashboard
    st.subheader("📊 Review Statistics")
    col1, col2, col3 = st.columns(3)
    airline_count = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'airline')
    seat_count = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'seat')
    lounge_count = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'lounge')
    
    with col1:
        st.metric("Airline Reviews", airline_count)
    with col2:
        st.metric("Seat Reviews", seat_count)
    with col3:
        st.metric("Lounge Reviews", lounge_count)
    
    # Chat interface
    chat_interface()
