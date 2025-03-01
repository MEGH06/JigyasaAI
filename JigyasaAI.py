import streamlit as st
import os
import io
from PIL import Image
import time
import ch2 as rag_system  # Import your existing module

# Configure page
st.set_page_config(
    page_title="JigyasaAI",
    page_icon="📖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# App title and description
st.title("JigyasaAI: Multimodal RAG Chatbot")
st.markdown("""
This chatbot can answer questions about PDF documents by using text and images from the document.
Upload a PDF, wait for processing to complete, then ask questions about the content.
""")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF... This may take a few minutes depending on the file size."):
            # Save the uploaded file temporarily
            temp_file_path = f"temp_upload_{int(time.time())}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                # Process the PDF using your existing function
                rag_system.process_pdf(temp_file_path)
                st.session_state.pdf_processed = True
                st.success(f"✅ '{uploaded_file.name}' successfully processed!")
                
                # Clean up the temp file
                os.remove(temp_file_path)
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    if st.session_state.pdf_processed:
        st.success("Document is ready for queries!")
        if st.button("Process Another Document"):
            st.session_state.pdf_processed = False
            st.rerun()

# Initialize system
if not st.session_state.initialized:
    # Load processed files
    rag_system.load_processed_files()
    st.session_state.initialized = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "images" in message:
            # Display text response
            st.write(message["content"])
            
            # Display images in columns
            if message["images"]:
                st.markdown("**Relevant images from the document:**")
                cols = st.columns(min(3, len(message["images"])))
                
                for i, (img_url, col) in enumerate(zip(message["images"], cols)):
                    with col:
                        st.image(img_url, caption=f"Image {i+1}", use_column_width=True)
                        st.markdown(f"[Open image]({img_url})")
        else:
            st.write(message["content"])

# Accept user input if a document has been processed
if st.session_state.pdf_processed:
    user_query = st.chat_input("Ask something about the document...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response
                response = rag_system.generate_response(user_query)
                
                # Extract image URLs if present
                images = []
                text_response = response
                
                # Parse the response to separate text and image URLs
                if "Relevant images from the document:" in response:
                    parts = response.split("Relevant images from the document:")
                    text_response = parts[0].strip()
                    
                    # Extract URLs
                    image_section = parts[1]
                    for line in image_section.strip().split("\n"):
                        if line.startswith("[Image") and "]" in line:
                            url = line.split("] ")[1].strip()
                            images.append(url)
                
                # Display text response
                st.write(text_response)
                
                # Display images
                if images:
                    st.markdown("**Relevant images from the document:**")
                    cols = st.columns(min(3, len(images)))
                    
                    for i, (img_url, col) in enumerate(zip(images, cols)):
                        with col:
                            st.image(img_url, caption=f"Image {i+1}", use_column_width=True)
                            st.markdown(f"[Open image]({img_url})")
                
                # Add response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": text_response,
                    "images": images
                })
else:
    st.info("👈 Please upload a PDF document to get started.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by [JigyasaAI](https://jigyasa-ai.com)")