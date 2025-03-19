import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Generative AI
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API key not found. Please set GEMINI_API_KEY in the .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Function to scrape data using BeautifulSoup
def scrape_website(url):
    try:
        # Direct web scraping
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all review items
        review_items = soup.select('.review-list .item')
        
        # Determine review type based on URL
        review_type = "unknown"
        if 'airline-reviews' in url:
            review_type = 'airline'
        elif 'seat-reviews' in url:
            review_type = 'seat'
        elif 'lounge-reviews' in url:
            review_type = 'lounge'
        
        # Extract all data from each review
        all_reviews = []
        for item in review_items:
            # Create a dictionary for all content
            review_data = {'type': review_type}
            
            # Extract all text and attributes from the review item
            # Convert the entire review item to text, preserving structure
            review_html = str(item)
            review_text = item.get_text(separator=' ', strip=True)
            
            # Add the extracted data to the review dictionary
            review_data['full_text'] = review_text
            
            # Try to extract specific sections for better organization
            # These are optional and won't limit what's captured
            try:
                if item.select_one('.rating-10 .rating'):
                    review_data['rating'] = item.select_one('.rating-10 .rating').text.strip()
                
                if item.select_one('.text_content'):
                    review_data['review_text'] = item.select_one('.text_content').text.strip()
                
                if item.select_one('.review-meta'):
                    review_data['metadata'] = item.select_one('.review-meta').text.strip()
                
                # Try to extract other potential fields
                if item.select_one('.cabin-flown'):
                    review_data['cabin'] = item.select_one('.cabin-flown').text.strip()
                
                if item.select_one('.aircraft'):
                    review_data['aircraft'] = item.select_one('.aircraft').text.strip()
                
                if item.select_one('.route'):
                    review_data['route'] = item.select_one('.route').text.strip()
                
                if item.select_one('.recommend-yes, .recommend-no'):
                    review_data['recommendation'] = item.select_one('.recommend-yes, .recommend-no').text.strip()
            except:
                # If any extraction fails, we still have the full text
                pass
            
            all_reviews.append(review_data)
        
        return all_reviews
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return []

# Initialize RAG components
def initialize_rag(scraped_content):
    # Convert scraped content to text documents
    documents = []
    
    for review in scraped_content:
        # Create a single document from all available fields
        doc_text = ""
        
        # Add all fields available in the review
        for key, value in review.items():
            if value and isinstance(value, str):
                doc_text += f"{key}: {value}\n"
        
        documents.append(doc_text)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.create_documents(documents)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# Set up the page
st.set_page_config(page_title="British Airways Reviews Analysis", page_icon="✈️")

# Main app
st.title("British Airways Reviews Analysis")
st.write("This app analyzes customer reviews from British Airways using RAG and Google Gemini AI.")

# URLs to scrape
urls = [
    "https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/seat-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000",
    "https://www.airlinequality.com/lounge-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=200000"
]

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.scraped_content = []
    st.session_state.vector_store = None
    st.session_state.chat_history = []

# Scrape data and initialize RAG on first run
if not st.session_state.initialized:
    with st.spinner("Scraping data and initializing RAG. This may take a minute..."):
        # Scrape content from all URLs
        all_reviews = []
        for url in urls:
            reviews = scrape_website(url)
            all_reviews.extend(reviews)
            # Add a small delay to avoid overwhelming the server
            time.sleep(1)
        
        st.session_state.scraped_content = all_reviews
        
        # Initialize RAG
        if all_reviews:
            st.session_state.vector_store = initialize_rag(all_reviews)
            st.session_state.initialized = True
            st.success(f"Successfully scraped {len(all_reviews)} reviews and initialized RAG.")
        else:
            st.error("Failed to scrape any reviews. Please check your internet connection and try again.")

# Display basic stats
if st.session_state.initialized:
    # Count reviews by type
    airline_reviews = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'airline')
    seat_reviews = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'seat')
    lounge_reviews = sum(1 for r in st.session_state.scraped_content if r.get('type') == 'lounge')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Airline Reviews", airline_reviews)
    with col2:
        st.metric("Seat Reviews", seat_reviews)
    with col3:
        st.metric("Lounge Reviews", lounge_reviews)

# Create the chat interface
st.header("Ask about British Airways Reviews")
st.write("Ask questions about British Airways reviews, seats, or lounges.")

# User input form
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("What would you like to know?", 
                               placeholder="Example: What are the most common complaints about British Airways?")
    submit_button = st.form_submit_button("Submit")

# Process the form submission
if submit_button and user_input and st.session_state.initialized:
    with st.spinner("Analyzing reviews..."):
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # RAG query
        if st.session_state.vector_store:
            # Get relevant documents
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents(user_input)
            
            # Extract context from relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt template
            prompt_template = """
            You are a detailed and professional data analyst specializing in British Airways reviews.
            
            Answer the following question based solely on the context provided.
            If the context doesn't contain enough information to answer the question, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Set up the model
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=api_key,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            # Create and execute the chain
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Generate response
            response = model.invoke(
                prompt.format(
                    context=context,
                    question=user_input
                )
            )
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})

# Display the conversation history
st.header("Conversation History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")

# Add sidebar with information
with st.sidebar:
    st.title("About")
    st.write("This app analyzes British Airways customer reviews using Retrieval-Augmented Generation (RAG) and Google's Gemini AI.")
    
    st.header("Data Sources")
    st.markdown("""
    - [Airline Reviews](https://www.airlinequality.com/airline-reviews/british-airways/)
    - [Seat Reviews](https://www.airlinequality.com/seat-reviews/british-airways/)
    - [Lounge Reviews](https://www.airlinequality.com/lounge-reviews/british-airways/)
    """)
    
    st.header("How it works")
    st.write("""
    1. The app scrapes reviews from the websites
    2. Detects and extracts all available data fields
    3. Chunks the text and creates embeddings
    4. Stores them in a vector database
    5. Retrieves relevant context for your questions
    6. Uses Gemini AI to generate insights
    """)
    
    # Add a section to show discovered fields
    if st.session_state.initialized and st.session_state.scraped_content:
        st.header("Discovered Fields")
        # Get all unique keys from the scraped content
        all_keys = set()
        for review in st.session_state.scraped_content:
            all_keys.update(review.keys())
        
        # Display them
        st.write(", ".join(sorted(all_keys)))
