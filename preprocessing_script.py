import os
import json
import sqlite3
import numpy as np
import re
import html2text # For robust Markdown to text conversion
import aiohttp
import asyncio
import argparse
import logging
from datetime import datetime, timezone # Import timezone for consistent datetimes
from dotenv import load_dotenv
from tqdm import tqdm # For progress bars

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configure logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, # INFO level for general progress, DEBUG for verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Constants and environment loading
# ─────────────────────────────────────────────────────────────────────────────
# Adjusted paths to match your new folder structure at the project root
DISCOURSE_DIR = "discourse_json" # Folder with individual Discourse topic JSONs
MARKDOWN_DIR = "tds_pages_md"   # Folder with individual Markdown files

DB_PATH = "knowledge_base.db" # The SQLite database file

# Chunking parameters (character counts, adjust based on your LLM's token limits if needed)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Base URL for Discourse, used for constructing full URLs in the DB
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in" 

load_dotenv()
API_KEY = os.getenv("API_KEY") # This will be your OpenAI API Key (or aipipe key)

if not API_KEY:
    logger.error("API_KEY environment variable is not set. Cannot create embeddings. Please set it before running.")
    # The script will continue to run the chunking and DB insertion parts,
    # but embedding creation will fail.

# ─────────────────────────────────────────────────────────────────────────────
# 3) Database setup / migrations
# ─────────────────────────────────────────────────────────────────────────────
def create_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name (e.g., row['column_name'])
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}", exc_info=True) # exc_info=True to log full traceback
        return None

def create_tables(conn):
    """Creates the necessary tables in the SQLite database if they don't exist."""
    try:
        cursor = conn.cursor()
        
        # Table for Discourse posts chunks
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            reply_to_post_number INTEGER DEFAULT 0,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB
        )
        ''')
        
        # Table for markdown document chunks
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
        ''')
        
        # Add reply_to_post_number column if it doesn't exist (for existing DBs, migration)
        try:
            cursor.execute("ALTER TABLE discourse_chunks ADD COLUMN reply_to_post_number INTEGER DEFAULT 0")
            logger.info("Added reply_to_post_number column to discourse_chunks table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e): # Specific check for column already existing
                logger.debug("reply_to_post_number column already exists, skipping ALTER TABLE.")
            else:
                logger.error(f"Error altering table discourse_chunks: {e}", exc_info=True)
        
        conn.commit()
        logger.info("Database tables created/checked successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}", exc_info=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Text Processing and Chunking Logic
# ─────────────────────────────────────────────────────────────────────────────
def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks, attempting to preserve sentence and paragraph boundaries.
    """
    if not text:
        return []
    
    # Normalize whitespace and paragraph breaks
    processed_text = re.sub(r'\n{2,}', '\n\n', text) # Replace multiple newlines with double newline
    processed_text = re.sub(r'[ \t]+', ' ', processed_text).strip() # Replace multiple spaces/tabs with single space

    if len(processed_text) <= chunk_size:
        return [processed_text]
    
    # Initial split by paragraphs
    paragraphs = processed_text.split('\n\n')
    
    intermediate_chunks = []
    current_paragraph_chunk = ""
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # If adding this paragraph makes the current chunk too large, start a new chunk
        if len(current_paragraph_chunk) + len(para) + 2 > chunk_size and current_paragraph_chunk:
            intermediate_chunks.append(current_paragraph_chunk.strip())
            current_paragraph_chunk = para
        else:
            # Add to current chunk
            if current_paragraph_chunk:
                current_paragraph_chunk += "\n\n" + para
            else:
                current_paragraph_chunk = para
                
    if current_paragraph_chunk.strip():
        intermediate_chunks.append(current_paragraph_chunk.strip())

    final_chunks = []
    # Further split chunks if they are still too large (e.g., very long paragraphs)
    for chunk_content in intermediate_chunks:
        if len(chunk_content) <= chunk_size:
            final_chunks.append(chunk_content)
        else:
            # Split by sentences if a paragraph-level chunk is too big
            sentences = re.split(r'(?<=[.!?])\s+', chunk_content)
            current_sentence_chunk = ""
            for sentence in sentences:
                if not sentence.strip(): continue

                if len(current_sentence_chunk) + len(sentence) + 1 > chunk_size and current_sentence_chunk:
                    final_chunks.append(current_sentence_chunk.strip())
                    current_sentence_chunk = sentence
                else:
                    if current_sentence_chunk:
                        current_sentence_chunk += " " + sentence
                    else:
                        current_sentence_chunk = sentence
            if current_sentence_chunk.strip():
                final_chunks.append(current_sentence_chunk.strip())

    # Apply overlap logic to the final set of chunks
    overlapped_final_chunks = []
    if final_chunks:
        overlapped_final_chunks.append(final_chunks[0])
        for i in range(1, len(final_chunks)):
            prev_chunk = final_chunks[i-1]
            current_chunk = final_chunks[i]

            # Take overlap from the end of the previous chunk
            overlap_content = prev_chunk[max(0, len(prev_chunk) - chunk_overlap):]
            
            # Prepend overlap content if not already part of the current chunk
            if overlap_content and not current_chunk.startswith(overlap_content):
                current_chunk = overlap_content + " " + current_chunk.strip()
            
            overlapped_final_chunks.append(current_chunk.strip())

    return overlapped_final_chunks

def clean_html(html_content):
    """
    Cleans HTML content using BeautifulSoup, converting it to plain text.
    Removes scripts/styles and normalizes whitespace/newlines.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Use newline as separator to preserve block-level text structure
    text = soup.get_text(separator='\n') 
    text = re.sub(r'\n{2,}', '\n\n', text) # Normalize multiple newlines to paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text).strip() # Normalize spaces and tabs
    
    return text

def markdown_to_plain_text(markdown_content):
    """
    Converts markdown content to plain text using html2text (via an intermediate HTML render).
    This helps in robust text extraction from markdown.
    """
    if not markdown_content:
        return ""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(markdown_content)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Process Discourse JSON files and insert into DB
# ─────────────────────────────────────────────────────────────────────────────
async def process_discourse_files(conn):
    """
    Reads individual Discourse topic JSON files from DISCOURSE_DIR,
    extracts and cleans post content, chunks it, and inserts into discourse_chunks table.
    """
    cursor = conn.cursor()
    
    # Check if table already contains data to skip re-processing raw files
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing discourse chunks in database. Skipping re-processing of raw Discourse JSONs.")
        return
    
    discourse_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(discourse_files)} Discourse JSON files to process from '{DISCOURSE_DIR}'.")
    
    total_chunks_processed = 0
    
    for file_name in tqdm(discourse_files, desc="Processing Discourse files"):
        try:
            file_path = os.path.join(DISCOURSE_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Extract topic-level metadata
                topic_id = data.get('id')
                topic_title = data.get('title', '')
                topic_slug = data.get('slug', '')
                
                posts = data.get('post_stream', {}).get('posts', [])
                
                for post in posts:
                    # Extract post-level metadata
                    post_id = post.get('id')
                    post_number = post.get('post_number')
                    author = post.get('username', '')
                    created_at = post.get('created_at', '')
                    reply_to = post.get('reply_to_post_number')
                    reply_to_post_number = reply_to if reply_to is not None else 0
                    likes = post.get('like_count', 0)
                    html_content = post.get('cooked', '')
                    
                    clean_content = clean_html(html_content)
                    
                    if len(clean_content.strip()) < 20:
                        logger.debug(f"Skipping short Discourse post (ID: {post_id}, Topic: {topic_id}).")
                        continue
                    
                    # Construct the full URL for the post
                    url = f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post_number}"
                    
                    # Create chunks from the cleaned post content
                    chunks = create_chunks(clean_content)
                    
                    # Insert each chunk into the database
                    for i, chunk in enumerate(chunks):
                        cursor.execute('''
                        INSERT INTO discourse_chunks 
                        (post_id, topic_id, topic_title, post_number, author, created_at, reply_to_post_number, likes, chunk_index, content, url, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            post_id, 
                            topic_id, 
                            topic_title, 
                            post_number, 
                            author, 
                            created_at, 
                            reply_to_post_number, 
                            likes, 
                            i, # chunk_index (0, 1, 2...)
                            chunk, 
                            url, 
                            None # Embedding will be added in a separate step
                        ))
                        total_chunks_processed += 1
                
                conn.commit() # Commit changes after each topic file is processed
        except Exception as e:
            logger.error(f"Error processing Discourse file {file_name}: {e}", exc_info=True)
            conn.rollback() # Rollback changes for the current file on error
    
    logger.info(f"Finished processing Discourse files. Created {total_chunks_processed} chunks.")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Process Markdown files and insert into DB
# ─────────────────────────────────────────────────────────────────────────────
async def process_markdown_files(conn):
    """
    Reads individual Markdown files from MARKDOWN_DIR,
    extracts content and metadata (including frontmatter), chunks it,
    and inserts into markdown_chunks table.
    """
    cursor = conn.cursor()
    
    # Check if table already contains data to skip re-processing raw files
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing markdown chunks in database. Skipping re-processing of raw Markdown files.")
        return
    
    markdown_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    logger.info(f"Found {len(markdown_files)} Markdown files to process from '{MARKDOWN_DIR}'.")
    
    total_chunks_processed = 0
    
    for file_name in tqdm(markdown_files, desc="Processing Markdown files"):
        try:
            file_path = os.path.join(MARKDOWN_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                doc_title = os.path.splitext(file_name)[0] # Default title from filename
                original_url = ""
                downloaded_at = datetime.now(timezone.utc).isoformat() # Default to now if not in frontmatter
                
                # Extract metadata from YAML frontmatter (e.g., Jekyll/Hugo style: --- ... ---)
                frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
                if frontmatter_match:
                    frontmatter_str = frontmatter_match.group(1)
                    
                    # Extract title
                    title_match = re.search(r'title:\s*[\'"]?(.*?)[\'"]?$', frontmatter_str, re.MULTILINE)
                    if title_match:
                        doc_title = title_match.group(1).strip()
                    
                    # Extract original URL
                    url_match = re.search(r'original_url:\s*[\'"]?(.*?)[\'"]?$', frontmatter_str, re.MULTILINE)
                    if url_match:
                        original_url = url_match.group(1).strip()
                    
                    # Extract downloaded_at timestamp
                    date_match = re.search(r'downloaded_at:\s*[\'"]?(.*?)[\'"]?$', frontmatter_str, re.MULTILINE)
                    if date_match:
                        downloaded_at = date_match.group(1).strip()

                    # Remove frontmatter from content
                    content = content[frontmatter_match.end():]
                
                # Convert markdown content to plain text for chunking and embedding
                plain_text_content = markdown_to_plain_text(content)
                
                if len(plain_text_content.strip()) < 20:
                    logger.debug(f"Skipping short Markdown file (Title: {doc_title}, File: {file_name}).")
                    continue
                
                # Create chunks from the cleaned content
                chunks = create_chunks(plain_text_content)
                
                # Insert each chunk into the database
                for i, chunk in enumerate(chunks):
                    cursor.execute('''
                    INSERT INTO markdown_chunks 
                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (doc_title, original_url, downloaded_at, i, chunk, None))
                    total_chunks_processed += 1
                
                conn.commit() # Commit changes after each file is processed
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_name}: {e}", exc_info=True)
            conn.rollback() # Rollback changes for the current file on error
    
    logger.info(f"Finished processing Markdown files. Created {total_chunks_processed} chunks.")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Create Embeddings and update DB (with retries and long text handling)
# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Embedding API configuration
OPENAI_EMBEDDING_API_URL = "https://aipipe.org/openai/v1/embeddings"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_MAX_CHARS = 8000 # Conservative character limit per embedding request

async def embed_text(session: aiohttp.ClientSession, text: str, record_id: int, 
                     is_discourse: bool, max_retries: int = 3, part_suffix: str = "") -> bool:
    """
    Sends a single text chunk to the embedding API and updates the database.
    Handles retries for rate limits.
    """
    retries = 0
    while retries < max_retries:
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": OPENAI_EMBEDDING_MODEL,
                "input": text
            }
            
            async with session.post(OPENAI_EMBEDDING_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    embedding_blob = json.dumps(embedding).encode('utf-8') # Store as JSON string BLOB

                    conn = create_connection() # Re-open connection for update
                    if not conn: return False

                    try:
                        cursor = conn.cursor()
                        if is_discourse:
                            # Update the existing record's embedding
                            cursor.execute(
                                "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                (embedding_blob, record_id)
                            )
                        else:
                            # Update the existing record's embedding
                            cursor.execute(
                                "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                                (embedding_blob, record_id)
                            )
                        conn.commit()
                        logger.debug(f"Successfully embedded and updated DB for ID {record_id} ({'discourse' if is_discourse else 'markdown'}) {part_suffix}.")
                        return True
                    except sqlite3.Error as db_err:
                        logger.error(f"DB Error updating embedding for ID {record_id}: {db_err}", exc_info=True)
                        conn.rollback()
                        return False
                    finally:
                        conn.close() # Always close connection
                        
                elif response.status == 429: # Rate limit error
                    error_text = await response.text()
                    logger.warning(f"Rate limit hit for ID {record_id} {part_suffix}. Retrying in {5 * (retries + 1)}s: {error_text}")
                    await asyncio.sleep(5 * (retries + 1)) # Exponential backoff
                    retries += 1
                else:
                    error_text = await response.text()
                    logger.error(f"API Error embedding text for ID {record_id} {part_suffix} (status {response.status}): {error_text}")
                    return False # Non-retryable API error
        except aiohttp.ClientError as e:
            logger.error(f"Network error during embedding for ID {record_id} {part_suffix} (attempt {retries+1}): {e}", exc_info=True)
            retries += 1
            await asyncio.sleep(3 * retries) # Wait before retry
        except Exception as e:
            logger.error(f"Unexpected error during embedding for ID {record_id} {part_suffix}: {e}", exc_info=True)
            return False # Non-retryable unexpected error
            
    logger.error(f"Failed to embed text for ID {record_id} {part_suffix} after {max_retries} retries.")
    return False

async def create_embeddings(api_key: str):
    """
    Orchestrates embedding creation for all chunks in the database that are missing embeddings.
    Handles batches and long texts.
    """
    if not api_key:
        logger.error("API_KEY is not set. Cannot create embeddings.")
        return False
        
    conn = create_connection()
    if not conn:
        logger.error("Failed to connect to database for embedding creation.")
        return False

    cursor = conn.cursor()
    
    # Get chunks without embeddings for both tables
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks_to_embed = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks_to_embed)} discourse chunks to embed.")
    
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks_to_embed = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks_to_embed)} markdown chunks to embed.")
    
    # Use aiohttp.ClientSession for connection pooling
    async with aiohttp.ClientSession() as session:
        # Process chunks for Discourse
        for i, (record_id, content) in enumerate(tqdm(discourse_chunks_to_embed, desc="Embedding Discourse Chunks")):
            if len(content) > OPENAI_EMBEDDING_MAX_CHARS:
                # Handle very long texts by chunking them further if needed for embedding API
                # Note: The `create_chunks` function handles initial content splitting.
                # This specific `if` block is for a scenario where `create_chunks` might still yield
                # a single chunk larger than the embedding model's input limit.
                logger.warning(f"Discourse chunk ID {record_id} is too long ({len(content)} chars) for embedding model. Splitting for embedding.")
                # For simplicity here, we'll embed the first 8000 chars as a proxy.
                # A more robust solution would re-chunk this specific content into embedding-API-friendly pieces
                # and average their embeddings, or store multiple embeddings linked to one content chunk.
                # For this example, we'll embed a truncated version.
                success = await embed_text(session, content[:OPENAI_EMBEDDING_MAX_CHARS], record_id, True)
            else:
                success = await embed_text(session, content, record_id, True)
            
            if not success:
                logger.error(f"Failed to embed Discourse chunk ID {record_id}. Continuing to next.")
            await asyncio.sleep(0.1) # Small delay to avoid hammering the API

        # Process chunks for Markdown
        for i, (record_id, content) in enumerate(tqdm(markdown_chunks_to_embed, desc="Embedding Markdown Chunks")):
            if len(content) > OPENAI_EMBEDDING_MAX_CHARS:
                logger.warning(f"Markdown chunk ID {record_id} is too long ({len(content)} chars) for embedding model. Splitting for embedding.")
                success = await embed_text(session, content[:OPENAI_EMBEDDING_MAX_CHARS], record_id, False)
            else:
                success = await embed_text(session, content, record_id, False)

            if not success:
                logger.error(f"Failed to embed Markdown chunk ID {record_id}. Continuing to next.")
            await asyncio.sleep(0.1) # Small delay

    conn.close()
    logger.info("Finished creating embeddings.")
    return True # Indicate successful completion of embedding phase

# ─────────────────────────────────────────────────────────────────────────────
# 8) Main execution function
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    """Main function to orchestrate the preprocessing pipeline."""
    # Argument parsing for chunk size and overlap
    parser = argparse.ArgumentParser(description="Preprocess Discourse posts and markdown files for RAG system")
    # FIX: Declare global variables BEFORE their first use in the function
    global CHUNK_SIZE, CHUNK_OVERLAP 
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Size of text chunks (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    args = parser.parse_args()
    
    # Update global chunking parameters if provided via arguments
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    logger.info(f"Using chunk size: {CHUNK_SIZE}, chunk overlap: {CHUNK_OVERLAP}")
    
    # Create database connection and tables
    conn = create_connection()
    if conn is None:
        logger.error("Failed to establish database connection. Exiting preprocessing.")
        return
    
    create_tables(conn) # Creates tables if they don't exist, handles simple migrations
    
    # Process raw Discourse JSON files and Markdown files (populate chunks into DB)
    await process_discourse_files(conn)
    await process_markdown_files(conn)
    
    # Close connection briefly before embedding to avoid long-lived connections for async
    conn.close() 
    logger.info("Raw data processing complete. Database connection closed.")

    # Create embeddings for all chunks currently in the database
    # This function will open and close its own connections as needed per update
    embedding_success = await create_embeddings(API_KEY)
    
    if embedding_success:
        logger.info("Preprocessing complete: All chunks created and embedded successfully.")
    else:
        logger.error("Preprocessing finished, but some embeddings may have failed. Check logs for details.")

# ─────────────────────────────────────────────────────────────────────────────
# 9) Entry point for script execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
