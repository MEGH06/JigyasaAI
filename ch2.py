import os
import re
import io
import fitz  # PyMuPDF
import chromadb
import cloudinary
import google.generativeai as genai
from PIL import Image
from imagehash import phash
from typing import List, Dict, Set, Optional
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from cloudinary.utils import cloudinary_url
import cloudinary.uploader
import tiktoken
import json
import time
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

# ChromaDB multi-modal setup
CHROMA_DB_PATH = "chroma_db"
PROCESSED_FILES_PATH = os.path.join(CHROMA_DB_PATH, "processed_files.json")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = OpenCLIPEmbeddingFunction()

print('All imports done')
PROCESSED_FILES = {}
IMAGE_HASH_CACHE = {}
IMAGE_METADATA_CACHE = {}  # New cache for image metadata

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
    
    # Initialize or get collection (without deleting existing data)
    collection_name = "multimodal_rag"
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"Error creating collection: {e}")
        # Try recreating the collection if it fails
        try:
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        except Exception as e2:
            print(f"Fatal error with ChromaDB: {e2}")
            raise
    
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
    
    # Store content in ChromaDB
    if all_text_chunks:
        store_content_in_chroma(all_text_chunks, all_chunk_images, all_images_with_metadata, file_hash)
    
    # Update processed files record
    PROCESSED_FILES[file_key] = {
        "hash": file_hash,
        "last_processed": datetime.now().isoformat(),
        "image_hashes": IMAGE_HASH_CACHE,
        "image_metadata": IMAGE_METADATA_CACHE
    }
    save_processed_files()

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

def store_content_in_chroma(text_chunks: List[str], chunk_images: List[List[str]], image_metadata: List[Dict], file_hash: str):
    collection = chroma_client.get_or_create_collection(
        name="multimodal_rag",
        embedding_function=embedding_function
    )
    
    # Delete existing entries from this PDF
    try:
        count = collection.count()
        collection.delete(where={"source": file_hash})
        new_count = collection.count()
        print(f"Deleted {count - new_count} entries with source {file_hash}")
    except Exception as e:
        print(f"Error deleting existing entries: {e}")
    
    # Generate unique IDs using index to guarantee uniqueness
    text_ids = [
        f"text_{file_hash}_{i}_{hash(chunk) & 0xFFFFFFFF}"
        for i, chunk in enumerate(text_chunks)
    ]
    
    # Create metadata for text chunks
    metadatas = []
    for images in chunk_images:
        # Store full image URLs and page information
        image_list = []
        for url in images:
            for meta in image_metadata:
                if meta.get("url") == url:
                    image_list.append({
                        "url": url,
                        "page": meta.get("page", 0),
                        "public_id": meta.get("public_id", "")
                    })
                    break
        
        metadatas.append({
            "images": json.dumps(image_list) if image_list else "",
            "source": str(file_hash),
            "image_urls": ",".join(images) if images else ""
        })
    
    # Check for duplicate IDs
    if len(text_ids) != len(set(text_ids)):
        print("Warning: Duplicate IDs detected in text_ids")
        text_ids = [f"{id}_{i}" for i, id in enumerate(text_ids)]
    
    # Upsert text chunks with batching to avoid memory issues
    batch_size = 50
    for i in range(0, len(text_chunks), batch_size):
        end_idx = min(i + batch_size, len(text_chunks))
        try:
            collection.upsert(
                ids=text_ids[i:end_idx],
                documents=text_chunks[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"Successfully upserted text chunks {i+1}-{end_idx} of {len(text_chunks)}")
        except Exception as e:
            print(f"Error upserting text chunks {i+1}-{end_idx}: {e}")
        
        # Small delay to avoid overloading
        time.sleep(0.5)
    
    # Process unique images
    unique_image_urls = []
    unique_image_metadatas = []
    
    seen_urls = set()
    for meta in image_metadata:
        url = meta.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_image_urls.append(url)
            unique_image_metadatas.append({
                "page": meta.get("page", 0),
                "public_id": meta.get("public_id", ""),
                "source": str(file_hash)
            })
    
    if unique_image_urls:
        try:
            # Generate image IDs
            image_ids = [f"img_{file_hash}_{i}_{hash(url) & 0xFFFFFFFF}" 
                         for i, url in enumerate(unique_image_urls)]
            
            # Check for duplicate image IDs
            if len(image_ids) != len(set(image_ids)):
                print("Warning: Duplicate IDs detected in image_ids")
                image_ids = [f"{id}_{i}" for i, id in enumerate(image_ids)]
            
            # Process images in smaller batches to avoid memory issues
            img_batch_size = 20
            for i in range(0, len(unique_image_urls), img_batch_size):
                end_idx = min(i + img_batch_size, len(unique_image_urls))
                batch_urls = unique_image_urls[i:end_idx]
                
                try:
                    # Get embeddings for image batch
                    image_embeddings = embedding_function(batch_urls)
                    
                    # Upsert image batch
                    collection.upsert(
                        ids=image_ids[i:end_idx],
                        embeddings=image_embeddings,
                        metadatas=unique_image_metadatas[i:end_idx],
                        documents=[f"Image at page {meta.get('page', 0)}" for meta in unique_image_metadatas[i:end_idx]]
                    )
                    print(f"Successfully upserted images {i+1}-{end_idx} of {len(unique_image_urls)}")
                except Exception as e:
                    print(f"Error upserting images {i+1}-{end_idx}: {e}")
                    print(f"URLs: {batch_urls}")
                
                # Small delay to avoid overloading
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error in image embedding process: {e}")

def query_rag(query: str, n_results: int = 5) -> List[Dict]:
    """Enhanced multi-modal RAG query with linked images"""
    try:
        collection = chroma_client.get_collection(
            name="multimodal_rag",
            embedding_function=embedding_function
        )
        
        # Query text first
        text_results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        combined_results = []
        
        # Process text results and their linked images
        if "documents" in text_results and text_results["documents"]:
            for i, doc in enumerate(text_results["documents"][0]):
                result = {
                    "type": "text",
                    "content": doc,
                    "distance": text_results["distances"][0][i] if "distances" in text_results else None
                }
                
                # Add linked images if available
                if "metadatas" in text_results and text_results["metadatas"][0]:
                    metadata = text_results["metadatas"][0][i]
                    if metadata and "image_urls" in metadata and metadata["image_urls"]:
                        result["images"] = metadata["image_urls"].split(",") if metadata["image_urls"] else []
                    
                    # Try the JSON format if available
                    if metadata and "images" in metadata and metadata["images"]:
                        try:
                            image_data = json.loads(metadata["images"])
                            result["image_data"] = image_data
                            result["images"] = [img["url"] for img in image_data]
                        except:
                            # Fallback if JSON parsing fails
                            if "images" not in result:
                                result["images"] = []
                
                combined_results.append(result)
        
        # Query for images directly using text-to-image search
        # Without using the $exists operator which is not supported
        image_query_results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        # Add image results from direct query
        if "metadatas" in image_query_results and image_query_results["metadatas"]:
            for i, metadata in enumerate(image_query_results["metadatas"][0]):
                if metadata and "public_id" in metadata:
                    # This is an image document
                    image_url = None
                    
                    # Find the URL for this public_id in our cache
                    for meta_url, meta in IMAGE_METADATA_CACHE.items():
                        if meta.get("public_id") == metadata["public_id"]:
                            image_url = meta.get("url")
                            break
                    
                    if image_url:
                        # Check if this image is already included in a text result
                        already_included = False
                        for result in combined_results:
                            if "images" in result and image_url in result["images"]:
                                already_included = True
                                break
                        
                        if not already_included:
                            combined_results.append({
                                "type": "image",
                                "content": image_url,
                                "distance": image_query_results["distances"][0][i] if "distances" in image_query_results else None,
                                "page": metadata.get("page", 0)
                            })
        
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
                    
            # Try to get more detailed image data if available
            if "image_data" in item and item["image_data"]:
                for img_data in item["image_data"]:
                    img_url = img_data.get("url")
                    if img_url and img_url not in image_with_context:
                        image_with_context[img_url] = text_content
        
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
        
        # Explicitly add the image URLs to the response
        final_response = gemini_text
        
        # Add relevant images section with explicit URLs
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

if __name__ == "__main__":
    try:
        # Verify Cloudinary setup first
        if not verify_cloudinary_setup():
            print("Please fix your Cloudinary configuration before continuing.")
            exit(1)
            
        pdf_path = input("Enter the path to your PDF file: ")
        if not pdf_path:
            pdf_path = r"C:\Users\Megh\Downloads\Let Us C.pdf"  # Default path
            
        process_pdf(pdf_path)
        
        print("Multi-modal RAG system ready. Ask me anything about the document!")
        while True:
            query = input("\nYour question: ")
            if query.lower() in ('exit', 'quit'):
                break
            
            response = generate_response(query)
            print(f"\nAssistant: {response}")
            
            # If you're using the structured response version, use this instead:
            # response = generate_response(query)
            # print(f"\nAssistant: {response['text']}")
            # if response['images']:
            #     print("\nRelevant images:")
            #     for i, img_url in enumerate(response['images']):
            #         print(f"{i+1}. {img_url}")
    except Exception as e:
        print(f"Error: {str(e)}")
    try:
        # Verify Cloudinary setup first
        if not verify_cloudinary_setup():
            print("Please fix your Cloudinary configuration before continuing.")
            exit(1)
            
        pdf_path = input("Enter the path to your PDF file: ")
        if not pdf_path:
            pdf_path = r"C:\Users\Megh\Downloads\Let Us C.pdf"  # Default path
            
        process_pdf(pdf_path)
        
        print("Multi-modal RAG system ready. Ask me anything about the document!")
        while True:
            query = input("\nYour question: ")
            if query.lower() in ('exit', 'quit'):
                break
            response = generate_response(query)
            print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")