import os
import re
import io
import fitz  # PyMuPDF
import faiss
import numpy as np
import cloudinary
import google.generativeai as genai
from PIL import Image
from imagehash import phash
from typing import List, Dict, Set, Optional
from sentence_transformers import SentenceTransformer
from cloudinary.utils import cloudinary_url
import cloudinary.uploader
import tiktoken
import json
import time
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

# Environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Gemini configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# FAISS vector database setup
FAISS_DB_PATH = "faiss_db"
FAISS_INDEX_FILE = os.path.join(FAISS_DB_PATH, "index.faiss")
FAISS_META_FILE = os.path.join(FAISS_DB_PATH, "metadata.json")
PROCESSED_FILES_PATH = os.path.join(FAISS_DB_PATH, "processed_files.json")

# Create directory for FAISS DB
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('clip-ViT-B-32')
EMBEDDING_DIM = 512  # Dimension of CLIP embeddings

print('All imports done')
PROCESSED_FILES = {}
IMAGE_HASH_CACHE = {}
IMAGE_METADATA_CACHE = {}
DOCUMENT_STORE = []  # Store for document content and metadata

# Initialize FAISS index
def init_faiss_index():
    try:
        if os.path.exists(FAISS_INDEX_FILE):
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(FAISS_META_FILE, 'r') as f:
                meta = json.load(f)
            print(f"Loaded existing FAISS index with {index.ntotal} vectors")
            DOCUMENT_STORE.extend(meta.get("documents", []))
            return index
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
            print("Created new FAISS index")
            return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        print("Created new FAISS index")
        return index

# Global FAISS index
faiss_index = init_faiss_index()

def load_processed_files():
    """Load the record of processed files and their image hashes"""
    global PROCESSED_FILES, IMAGE_HASH_CACHE, IMAGE_METADATA_CACHE
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                PROCESSED_FILES = json.load(f)
                print(f"Loaded {len(PROCESSED_FILES)} processed file records.")
                
                # Load image hash cache from processed files
                for file_key, file_data in PROCESSED_FILES.items():
                    if "image_hashes" in file_data:
                        IMAGE_HASH_CACHE.update(file_data["image_hashes"])
                    if "image_metadata" in file_data:
                        IMAGE_METADATA_CACHE.update(file_data["image_metadata"])
        except Exception as e:
            print(f"Error loading processed files: {e}")
            PROCESSED_FILES = {}
    else:
        PROCESSED_FILES = {}

def save_processed_files():
    """Save the record of processed files and their image hashes"""
    try:
        os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
        
        # Update image metadata in the processed files
        for file_key in PROCESSED_FILES:
            PROCESSED_FILES[file_key]["image_hashes"] = IMAGE_HASH_CACHE
            PROCESSED_FILES[file_key]["image_metadata"] = IMAGE_METADATA_CACHE
        
        with open(PROCESSED_FILES_PATH, 'w') as f:
            json.dump(PROCESSED_FILES, f)
        print(f"Saved {len(PROCESSED_FILES)} processed file records with {len(IMAGE_HASH_CACHE)} image hashes.")
    except Exception as e:
        print(f"Error saving processed files: {e}")

def save_faiss_index():
    """Save the FAISS index and metadata"""
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        with open(FAISS_META_FILE, 'w') as f:
            json.dump({"documents": DOCUMENT_STORE}, f)
        print(f"Saved FAISS index with {faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def get_file_hash(file_path):
    """Get a simple hash of the file to detect changes"""
    import hashlib
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def process_pdf(pdf_path):
    """Processes PDF page by page with incremental processing support"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
    
    # Load previously processed files
    load_processed_files()
    
    # Check if file has been processed before
    file_hash = get_file_hash(pdf_path)
    file_key = os.path.basename(pdf_path)
    
    if file_key in PROCESSED_FILES and PROCESSED_FILES[file_key]["hash"] == file_hash:
        print(f"File {file_key} has already been processed and hasn't changed.")
        return
    
    # Process the PDF
    all_text_chunks = []
    all_chunk_images = []
    all_images_with_metadata = []
    
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text from the page
            page_text = page.get_text("text").strip()
            if not page_text:
                continue  # Skip blank pages
                
            # Extract and process images from the page
            page_images, page_metadata = process_images_on_page(page, page_num)
            all_images_with_metadata.extend(page_metadata)
            
            # Chunk text
            text_chunks = semantic_chunk(page_text)
            
            # Link images to text chunks
            chunk_images = []
            for _ in text_chunks:
                chunk_images.append([img["url"] for img in page_images])
            
            all_text_chunks.extend(text_chunks)
            all_chunk_images.extend(chunk_images)
    
    # Store content in FAISS
    if all_text_chunks:
        store_content_in_faiss(all_text_chunks, all_chunk_images, all_images_with_metadata, file_hash)
    
    # Update processed files record
    PROCESSED_FILES[file_key] = {
        "hash": file_hash,
        "last_processed": datetime.now().isoformat(),
        "image_hashes": IMAGE_HASH_CACHE,
        "image_metadata": IMAGE_METADATA_CACHE
    }
    save_processed_files()
    save_faiss_index()

def process_images_on_page(page, page_num):
    """Extract and process images from a given page with duplicate detection across sessions"""
    global IMAGE_HASH_CACHE, IMAGE_METADATA_CACHE
    
    images = []
    image_metadata = []
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png").lower()
            img_stream = io.BytesIO(image_bytes)
            
            # Create a unique identifier for the image
            image_id = f"page_{page_num+1}_img_{img_index}_{xref}"
            
            # Process image with PIL for hash
            try:
                pil_img = Image.open(img_stream)
                width, height = pil_img.size
                img_hash = str(phash(pil_img))
                
                # Skip duplicates - check in current session and previous sessions
                if img_hash in IMAGE_HASH_CACHE:
                    image_url = IMAGE_HASH_CACHE[img_hash]
                    images.append({"url": image_url, "page": page_num + 1})
                    
                    # Add metadata if we have it
                    if img_hash in IMAGE_METADATA_CACHE:
                        image_metadata.append(IMAGE_METADATA_CACHE[img_hash])
                    continue
                    
            except Exception as e:
                print(f"Error processing image with PIL: {e}")
                continue
                
            # Upload to Cloudinary if not already cached
            img_stream.seek(0)  # Reset stream for upload
            try:
                # Add metadata to the upload
                upload_result = cloudinary.uploader.upload(
                    img_stream, 
                    public_id=f"pdf_images/{image_id.replace(' ', '_')}",
                    folder="rag_images",
                    resource_type="image",
                    format=image_ext,
                    tags=["rag_system", f"page_{page_num+1}"]
                )
                
                image_url = upload_result["secure_url"]
                
                # Ensure the URL is properly formed and accessible
                image_url = cloudinary_url(upload_result["public_id"], 
                                          format=image_ext, 
                                          secure=True)[0]
                
                # Add to cache
                IMAGE_HASH_CACHE[img_hash] = image_url
                
                # Create metadata
                image_meta = {
                    "url": image_url,
                    "page": page_num + 1,
                    "format": image_ext,
                    "width": width,
                    "height": height,
                    "public_id": upload_result["public_id"],
                    "resource_type": "image",
                    "hash": img_hash
                }
                
                # Cache the metadata
                IMAGE_METADATA_CACHE[img_hash] = image_meta
                
                images.append({
                    "url": image_url,
                    "page": page_num + 1,
                    "format": image_ext
                })
                
                image_metadata.append(image_meta)
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Cloudinary upload failed for image {image_id}: {e}")
                
        except Exception as e:
            print(f"Error extracting image {xref}: {e}")
            
    print(f"Processed {len(images)} images on page {page_num+1}")
    return images, image_metadata

def semantic_chunk(text: str, chunk_size: int = 512, overlap: int = 50, max_chunks: int = None) -> List[str]:
    """Context-aware text chunking by paragraphs while respecting token limits."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = enc.encode(paragraph)
        paragraph_length = len(paragraph_tokens)

        if paragraph_length > chunk_size:
            # If a paragraph is too large, split it into sentences
            sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', paragraph)
            for sentence in sentences:
                sentence_tokens = enc.encode(sentence)
                sentence_length = len(sentence_tokens)

                if current_length + sentence_length > chunk_size:
                    chunks.append(enc.decode(current_chunk))
                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks  # Stop if max_chunks is reached
                    current_chunk = current_chunk[-overlap:] if overlap else []
                    current_length = len(current_chunk)

                current_chunk.extend(sentence_tokens)
                current_length += sentence_length
        else:
            if current_length + paragraph_length > chunk_size:
                chunks.append(enc.decode(current_chunk))
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks
                current_chunk = current_chunk[-overlap:] if overlap else []
                current_length = len(current_chunk)

            current_chunk.extend(paragraph_tokens)
            current_length += paragraph_length

    if current_chunk:
        chunks.append(enc.decode(current_chunk))

    return chunks

def store_content_in_faiss(text_chunks: List[str], chunk_images: List[List[str]], image_metadata: List[Dict], file_hash: str):
    """Store content in FAISS index"""
    global faiss_index, DOCUMENT_STORE
    
    # Generate embeddings for text chunks
    batch_size = 50
    start_idx = len(DOCUMENT_STORE)  # Current index to start from
    
    # Remove existing documents with the same file_hash
    cleaned_docs = [doc for doc in DOCUMENT_STORE if doc.get("source") != str(file_hash)]
    if len(cleaned_docs) < len(DOCUMENT_STORE):
        print(f"Removed {len(DOCUMENT_STORE) - len(cleaned_docs)} existing entries for this file")
        DOCUMENT_STORE = cleaned_docs
        # Since we modified the document store, we need to rebuild the index
        recreate_faiss_index()
    
    for i in range(0, len(text_chunks), batch_size):
        end_idx = min(i + batch_size, len(text_chunks))
        batch = text_chunks[i:end_idx]
        
        try:
            # Generate embeddings
            embeddings = embedding_model.encode(batch)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            faiss_index.add(embeddings)
            
            # Store documents with metadata
            for j, chunk in enumerate(batch):
                doc_idx = i + j
                doc_metadata = {
                    "text": chunk,
                    "source": str(file_hash),
                    "images": chunk_images[doc_idx] if doc_idx < len(chunk_images) else [],
                    "index": start_idx + doc_idx
                }
                DOCUMENT_STORE.append(doc_metadata)
            
            print(f"Successfully added chunks {i+1}-{end_idx} of {len(text_chunks)}")
        except Exception as e:
            print(f"Error adding chunks {i+1}-{end_idx}: {e}")
        
        # Small delay to avoid system overload
        time.sleep(0.2)
    
    # Process unique images
    unique_image_urls = []
    unique_image_metadata = []
    
    seen_urls = set()
    for meta in image_metadata:
        url = meta.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_image_urls.append(url)
            unique_image_metadata.append({
                "page": meta.get("page", 0),
                "public_id": meta.get("public_id", ""),
                "source": str(file_hash)
            })
    
    if unique_image_urls:
        try:
            # Process images in smaller batches
            img_batch_size = 20
            for i in range(0, len(unique_image_urls), img_batch_size):
                end_idx = min(i + img_batch_size, len(unique_image_urls))
                batch_urls = unique_image_urls[i:end_idx]
                batch_metadata = unique_image_metadata[i:end_idx]
                
                try:
                    # Get image embeddings
                    image_embeddings = embedding_model.encode(batch_urls)
                    image_embeddings = np.array(image_embeddings).astype('float32')
                    faiss.normalize_L2(image_embeddings)
                    
                    # Add to index
                    faiss_index.add(image_embeddings)
                    
                    # Store image metadata
                    start_img_idx = len(DOCUMENT_STORE)
                    for j, url in enumerate(batch_urls):
                        meta = batch_metadata[j]
                        img_doc = {
                            "text": f"Image at page {meta.get('page', 0)}",
                            "source": str(file_hash),
                            "type": "image",
                            "url": url,
                            "page": meta.get("page", 0),
                            "public_id": meta.get("public_id", ""),
                            "index": start_img_idx + j
                        }
                        DOCUMENT_STORE.append(img_doc)
                    
                    print(f"Successfully added images {i+1}-{end_idx} of {len(unique_image_urls)}")
                except Exception as e:
                    print(f"Error adding images {i+1}-{end_idx}: {e}")
                
                # Small delay to avoid overloading
                time.sleep(0.2)
                
        except Exception as e:
            print(f"Error in image embedding process: {e}")

def recreate_faiss_index():
    """Recreate the FAISS index from the document store"""
    global faiss_index, DOCUMENT_STORE
    
    # Create a new index
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    
    if not DOCUMENT_STORE:
        print("Document store is empty, no need to recreate index")
        return
    
    # Process in batches
    batch_size = 50
    doc_texts = [doc["text"] for doc in DOCUMENT_STORE]
    
    for i in range(0, len(doc_texts), batch_size):
        end_idx = min(i + batch_size, len(doc_texts))
        batch = doc_texts[i:end_idx]
        
        # Generate embeddings
        embeddings = embedding_model.encode(batch)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        faiss_index.add(embeddings)
        
        print(f"Rebuilt index for entries {i+1}-{end_idx} of {len(doc_texts)}")
        
    print(f"Successfully recreated FAISS index with {faiss_index.ntotal} vectors")

def query_rag(query: str, n_results: int = 5) -> List[Dict]:
    """Enhanced multi-modal RAG query with linked images using FAISS"""
    global faiss_index, DOCUMENT_STORE
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = faiss_index.search(query_embedding, n_results)
        
        # Process results
        combined_results = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(DOCUMENT_STORE):  # Valid index
                doc = DOCUMENT_STORE[idx]
                
                result = {
                    "type": doc.get("type", "text"),
                    "content": doc["text"] if doc.get("type") != "image" else doc.get("url", ""),
                    "distance": float(distances[0][i])
                }
                
                # Add image URLs if available
                if "images" in doc and doc["images"]:
                    result["images"] = doc["images"]
                
                # For image results, add page information
                if doc.get("type") == "image":
                    result["page"] = doc.get("page", 0)
                
                combined_results.append(result)
        
        # Sort by relevance (distance)
        combined_results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
        
        return combined_results
    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        return []

def generate_response(query: str) -> str:
    """Generate response using Gemini with multi-modal context"""
    results = query_rag(query)
    
    if not results:
        return "I couldn't find relevant information in the documents."
    
    context_parts = []
    image_urls = []
    image_with_context = {}  # Map images to their context
    
    for item in results:
        if item["type"] == "text":
            text_content = item['content']
            context_parts.append(f"Text excerpt: {text_content}")
            
            # Collect linked images and their context
            if "images" in item and item["images"]:
                for img_url in item["images"]:
                    if img_url not in image_with_context:
                        image_with_context[img_url] = text_content
                    image_urls.append(img_url)
        
        elif item["type"] == "image":
            img_url = item["content"]
            image_urls.append(img_url)
            if img_url not in image_with_context:
                # If we have no context, note which page it's from
                page = item.get("page", 0)
                image_with_context[img_url] = f"Image from page {page}"
    
    # Remove duplicate images while preserving order
    unique_image_urls = []
    seen = set()
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            unique_image_urls.append(url)
    
    # Add image references with their context to the prompt
    for i, url in enumerate(unique_image_urls[:3]):  # Limit to first 3 images
        context_parts.append(f"Image {i+1}: {url}")
        context_parts.append(f"Image context: {image_with_context.get(url, 'No context available')}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Analyze this multi-modal context and answer the query:
    
    Context:
    {context}
    
    Query: {query}
    
    Provide a comprehensive answer considering both text and visual elements. 
    DO NOT include the image URLs in your response text."""
    
    try:
        response = model.generate_content(prompt)
        gemini_text = response.text.strip()
        
        # Add relevant images section with explicit URLs
        final_response = gemini_text
        
        if unique_image_urls:
            final_response += "\n\n--------------------\nRelevant images from the document:\n"
            for i, url in enumerate(unique_image_urls[:3]):
                final_response += f"\n[Image {i+1}] {url}\n"
        
        return final_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def verify_cloudinary_setup():
    """Verify that Cloudinary is properly configured"""
    try:
        # Test upload a simple image
        test_img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        test_img.save(img_io, format='PNG')
        img_io.seek(0)
        
        result = cloudinary.uploader.upload(
            img_io,
            folder="test",
            public_id="test_image"
        )
        
        print(f"Cloudinary test successful. Image URL: {result['secure_url']}")
        return True
    except Exception as e:
        print(f"Cloudinary configuration error: {e}")
        return False

def cleanup_session():
    """Clean up all resources after session ends"""
    global IMAGE_HASH_CACHE, IMAGE_METADATA_CACHE, DOCUMENT_STORE, faiss_index
    
    # Delete Cloudinary images
    deleted_count = 0
    try:
        for img_hash, meta in IMAGE_METADATA_CACHE.items():
            if "public_id" in meta:
                try:
                    cloudinary.uploader.destroy(meta["public_id"])
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting image {meta['public_id']}: {e}")
    except Exception as e:
        print(f"Error deleting images: {e}")
        