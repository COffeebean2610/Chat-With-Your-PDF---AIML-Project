import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Page config
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and Introduction
st.title("📄 Chat with your PDF")
st.markdown("---")

# What is this application
st.header("🤖 What is this Application?")
st.markdown("""
This is an AI-powered document analysis tool that allows you to have interactive conversations with your PDF documents. 
Upload any PDF and ask questions about its content - the AI will provide accurate answers based on the document.
""")

# How it's made
st.header("🔧 How it's Built")
st.markdown("""
• **Frontend**: Streamlit for user interface
            
• **PDF Processing**: PyMuPDF for text extraction
            
• **Text Processing**: LangChain for document chunking
            
• **Embeddings**: HuggingFace all-MiniLM-L6-v2 model
            
• **Vector Database**: Chroma for similarity search
            
• **AI Model**: Ollama with Phi3-mini for fast responses
            
• **Architecture**: RAG (Retrieval Augmented Generation)
""")

# How it works
st.header("⚙️ How it Works")
st.markdown("""
• **Step 1**: PDF text is extracted and split into chunks
            
• **Step 2**: Text chunks are converted to embeddings (numerical representations)
            
• **Step 3**: Embeddings are stored in a vector database
            
• **Step 4**: When you ask a question, relevant chunks are retrieved
            
• **Step 5**: AI generates answers using retrieved context
""")

# Applications
st.header("🎯 Applications & Use Cases")
st.markdown("""
• **Research**: Quickly find information in academic papers
            
• **Legal**: Analyze contracts and legal documents
            
• **Business**: Extract insights from reports and proposals
            
• **Education**: Study materials and textbooks
            
• **Technical**: Navigate manuals and documentation
            
• **Personal**: Organize and query personal documents
""")

# How to use
st.header("📋 How to Use")
st.markdown("""
**Step 1**: Upload your PDF file using the sidebar

**Step 2**: Wait for processing (text extraction and indexing)

**Step 3**: Start asking questions about your document

**Step 4**: Get instant AI-powered answers with context

**Step 5**: Continue the conversation or upload a new PDF
""")

st.markdown("---")
st.header("💬 Chat Interface")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and st.session_state.retriever is None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyMuPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                
                # Create vector store
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                
                # Store retriever in session state with fewer chunks
                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 2}  # Reduced from 3 to 2 for speed
                )
                
                st.success(f"PDF processed successfully! Extracted {len(chunks)} text chunks.")
                
                # Debug: Show first chunk
                if chunks:
                    with st.expander("Preview first chunk"):
                        st.text(chunks[0].page_content[:500] + "..." if len(chunks[0].page_content) > 500 else chunks[0].page_content)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

# Main chat interface
if st.session_state.retriever is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize Ollama LLM with faster settings
                llm = Ollama(
                    model="phi3:mini",
                    temperature=0.1,
                    num_predict=256,  # Limit response length
                    top_k=10,
                    top_p=0.3
                )
                
                # Create custom prompt template
                prompt_template = """You have been provided with content from a PDF document. Answer the question using this content. Do not mention that you don't have access to the PDF - you DO have access through the content below.

Document Content:
{context}

Question: {question}

Answer:"""
                
                rag_prompt = PromptTemplate.from_template(prompt_template)
                
                # Create RAG chain using LCEL
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Get response
                answer = rag_chain.invoke(prompt)
                
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF file to start chatting!")
    
    # Clear chat history when no PDF is loaded
    if st.session_state.messages:
        st.session_state.messages = []

# Reset button in sidebar
with st.sidebar:
    if st.button("Reset Chat"):
        st.session_state.retriever = None
        st.session_state.messages = []
        st.rerun()