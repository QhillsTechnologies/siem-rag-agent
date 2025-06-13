import streamlit as st
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
from agent import analyze_logs_with_agent
# Set page config
st.set_page_config(page_title="JSON Log Processor", page_icon="📊")

st.title("📊 JSON Log Processor")

# Create tabs
tab1, tab2 = st.tabs(["📁 Upload Interface", "💬 Chat Interface"])

with tab1:
    st.header("Upload JSON Logs")
    
    # API Key input
    api_key = os.getenv("OPENAI_API_KEY")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
    
    # Process button
    if st.button("Process JSON Logs"):
        if not uploaded_file:
            st.error("Please upload a JSON file")
        elif not api_key:
            st.error("Please enter your OpenAI API key")
        else:
            try:
                # Read JSON file
                json_data = json.load(uploaded_file)
                
                # Check if it's an array
                if not isinstance(json_data, list):
                    st.error("JSON file should contain an array of objects")
                else:
                    st.info(f"Found {len(json_data)} log entries")
                    
                    with st.spinner("Processing logs and storing in ChromaDB..."):
                        # Initialize OpenAI embeddings
                        embeddings = OpenAIEmbeddings(
                            model="text-embedding-ada-002",
                            openai_api_key=api_key
                        )
                        
                        # Convert each JSON object to Document
                        documents = []
                        for i, log_entry in enumerate(json_data):
                            # Convert JSON object to string
                            text_content = json.dumps(log_entry)
                            
                            # Create Document
                            doc = Document(
                                page_content=text_content,
                                metadata={"index": i}
                            )
                            documents.append(doc)
                        
                        # Store in ChromaDB
                        vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            persist_directory="./chroma_db"
                        )
                        
                        st.success(f"✅ Successfully processed {len(documents)} logs!")
                        st.info("Embeddings stored in ChromaDB")
            
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def get_relevant_chunks(question, api_key, k=5):
    """
    Fetch relevant chunks from ChromaDB based on the question
    
    Args:
        question (str): User's question
        api_key (str): OpenAI API key
        k (int): Number of relevant chunks to return
    
    Returns:
        list: List of relevant documents
    """
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )
        # Load existing ChromaDB
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        # Search for relevant chunks
        relevant_docs = vectorstore.similarity_search(question, k=k)
        return relevant_docs
    
    except Exception as e:
        st.error(f"Error fetching relevant chunks: {str(e)}")
        return []


with tab2:
    st.header("💬 Intelligent Log Analysis")
    
    # Check if ChromaDB exists
    try:
        # Get API key from environment
        search_api_key = os.getenv("OPENAI_API_KEY")
        
        if search_api_key:
            # Question input
            question = st.text_input(
                "Ask a question about your logs:", 
                placeholder="e.g., What are the most common errors? Show me authentication failures."
            )
            
            # Number of chunks to retrieve
            num_chunks = st.slider("Number of log entries to analyze", 1, 20, 5)
            
            if st.button("🔍 Analyze Logs", type="primary"):
                if question:
                    with st.spinner("🔍 Searching relevant logs..."):
                        # Get relevant chunks
                        relevant_docs = get_relevant_chunks(question, search_api_key, k=num_chunks)
                        
                        if relevant_docs:
                            with st.spinner("🧠 Analyzing logs with AI..."):
                                # Use PydanticAI agent to analyze
                                response = analyze_logs_with_agent(question, relevant_docs, search_api_key)
                                
                                # Display results
                                st.subheader("📊 Analysis Results")
                                
                                # Main answer
                                st.markdown("### 💡 Answer")
                                st.write(response.answer)

                                # Show relevant logs from analysis
                                if response.relevant_logs:
                                    with st.expander("📄 Relevant Log Entries (From Analysis)"):
                                        for i, log_entry in enumerate(response.relevant_logs):
                                            st.markdown(f"**Relevant Log {i+1}:**")
                                            st.json(log_entry)
                                            st.divider()
                                
                        else:
                            st.warning("No relevant logs found for your question.")
                else:
                    st.warning("Please enter a question")
        else:
            st.error("Please set OPENAI_API_KEY environment variable")
            st.info("You can set it by running: `export OPENAI_API_KEY=your_api_key_here`")
            
    except Exception as e:
        st.warning("No ChromaDB found. Please upload and process JSON logs first in the Upload Interface tab.")
        st.error(f"Error: {str(e)}")
