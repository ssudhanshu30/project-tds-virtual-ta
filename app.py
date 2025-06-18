import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configure logging (DEBUG level for visibility)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Constants and environment loading
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 15
MAX_CONTEXT_CHUNKS = 4
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# ─────────────────────────────────────────────────────────────────────────────
# 4) Initialize FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Database setup / migrations
# ─────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable name-based access
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# If DB file does not exist, create tables. Otherwise, ensure the column exists.
if not os.path.exists(DB_PATH):
    logger.debug("Database file not found. Creating new SQLite database with required tables.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB,
        reply_to_post_number INTEGER DEFAULT 0
    )
    ''')
    c.execute('''
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
    conn.commit()
    conn.close()
else:
    logger.debug("Database file exists. Checking for missing columns.")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("ALTER TABLE discourse_chunks ADD COLUMN reply_to_post_number INTEGER DEFAULT 0")
        logger.info("Added reply_to_post_number column to discourse_chunks")
    except sqlite3.OperationalError:
        logger.debug("reply_to_post_number column already exists, skipping ALTER TABLE.")
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6) Cosine similarity helper
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 7) Function to get embedding from AIPipe proxy (with retries)
# ─────────────────────────────────────────────────────────────────────────────
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    retries = 0
    while retries < max_retries:
        try:
            logger.debug(f"Getting embedding for text of length {len(text)} (attempt {retries+1}/{max_retries}).")
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug("Successfully received embedding from API.")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limited. Retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception in get_embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)

# ─────────────────────────────────────────────────────────────────────────────
# 8) Find similar content in both tables (discourse + markdown)
# ─────────────────────────────────────────────────────────────────────────────
async def find_similar_content(query_embedding, conn):
    try:
        logger.debug("Finding similar content in database.")
        cursor = conn.cursor()
        results = []

        # --- Discourse chunks ---
        logger.debug("Querying discourse_chunks table for non-null embeddings.")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, reply_to_post_number, author, created_at,
               likes, chunk_index, content, url, embedding
        FROM discourse_chunks
        WHERE embedding IS NOT NULL
        """)
        discourse_chunks = cursor.fetchall()
        logger.debug(f"Fetched {len(discourse_chunks)} rows from discourse_chunks.")

        processed_count = 0
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "post_number": chunk["post_number"],
                        "reply_to_post_number": chunk["reply_to_post_number"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.debug(f"Processed {processed_count} / {len(discourse_chunks)} discourse rows.")
            except Exception as e:
                logger.error(f"Error processing discourse chunk ID {chunk['id']}: {e}")

        # --- Markdown chunks ---
        logger.debug("Querying markdown_chunks table for non-null embeddings.")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding
        FROM markdown_chunks
        WHERE embedding IS NOT NULL
        """)
        markdown_chunks = cursor.fetchall()
        logger.debug(f"Fetched {len(markdown_chunks)} rows from markdown_chunks.")

        processed_count = 0
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.debug(f"Processed {processed_count} / {len(markdown_chunks)} markdown rows.")
            except Exception as e:
                logger.error(f"Error processing markdown chunk ID {chunk['id']}: {e}")

        # Sort and group by document/post
        logger.debug(f"Total matching chunks before grouping: {len(results)}.")
        results.sort(key=lambda x: x["similarity"], reverse=True)

        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            key = f"{r['source']}_{r.get('post_id', r.get('title'))}"
            grouped_results.setdefault(key, []).append(r)

        final_results = []
        for key, chunks in grouped_results.items():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])

        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        to_return = final_results[:MAX_RESULTS]
        logger.debug(f"Returning {len(to_return)} results after grouping and truncation.")
        return to_return

    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 9) New: fetch replies for a given post_number (all chunks of each reply)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_replies_for_post(conn: sqlite3.Connection, topic_id: int, post_number: int) -> List[Dict[str, Any]]:
    """
    Given a topic_id and a post_number, return all reply‐posts’ combined content
    where reply_to_post_number == post_number.

    For each distinct reply post_id, fetch all chunks (ordered by chunk_index)
    and concatenate them into a single 'content' string.
    """
    logger.debug(f"fetch_replies_for_post called with topic_id={topic_id}, post_number={post_number}")

    cursor = conn.cursor()

    # 1) Find all distinct post_id values that reply to this post_number
    cursor.execute("""
        SELECT DISTINCT post_id
        FROM discourse_chunks
        WHERE topic_id = ? AND reply_to_post_number = ?
    """, (topic_id, post_number))
    rows = cursor.fetchall()
    reply_post_ids = [row["post_id"] for row in rows]
    logger.debug(f"  Found {len(reply_post_ids)} distinct reply post_id(s) for post_number={post_number}")

    replies = []
    for reply_post_id in reply_post_ids:
        # 2) Fetch *all* chunks for that reply_post_id, ordered by chunk_index
        cursor.execute("""
            SELECT chunk_index, author, content, url
            FROM discourse_chunks
            WHERE post_id = ?
            ORDER BY chunk_index ASC
        """, (reply_post_id,))
        chunk_rows = cursor.fetchall()
        if not chunk_rows:
            continue

        # 3) Concatenate chunk contents in order
        full_content = ""
        reply_author = None
        reply_url = None
        for cr in chunk_rows:
            full_content += cr["content"] + "\n"
            reply_author = cr["author"]        # same across chunks
            reply_url = cr["url"]              # same across chunks

        replies.append({
            "post_id": reply_post_id,
            "author": reply_author,
            "content": full_content.strip(),
            "url": reply_url
        })

        logger.debug(f"   → Built full reply for post_id={reply_post_id}, author={reply_author}, chunk_count={len(chunk_rows)}")

    return replies

# ─────────────────────────────────────────────────────────────────────────────
# 10) Enrich content with adjacent chunks and replies (uses new fetch_replies_for_post)
# ─────────────────────────────────────────────────────────────────────────────
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.debug(f"enrich_with_adjacent_chunks called with {len(results)} result(s).")
        cursor = conn.cursor()
        enriched_results = []

        for result in results:
            enriched_result = result.copy()
            additional_content = ""
            logger.debug(f"Processing result: source={result['source']}, post_id={result.get('post_id')}, chunk_index={result.get('chunk_index')}")

            if result["source"] == "discourse":
                post_id = result["post_id"]
                current_chunk_index = result["chunk_index"]
                topic_id = result["topic_id"]
                post_number = result["post_number"]
                logger.debug(f"  → Discourse chunk: topic_id={topic_id}, post_number={post_number}, chunk_index={current_chunk_index}")

                # 10.1) Adjacent chunk (previous)
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content
                        FROM discourse_chunks
                        WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        logger.debug(f"    Found previous chunk at index {current_chunk_index - 1}")
                        additional_content += prev_chunk["content"] + "\n"
                    else:
                        logger.debug(f"    No previous chunk at index {current_chunk_index - 1}")

                # 10.2) Adjacent chunk (next)
                cursor.execute("""
                    SELECT content
                    FROM discourse_chunks
                    WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    logger.debug(f"    Found next chunk at index {current_chunk_index + 1}")
                    additional_content += next_chunk["content"] + "\n"
                else:
                    logger.debug(f"    No next chunk at index {current_chunk_index + 1}")

                # 10.3) Fetch ALL replies for this post_number
                replies = fetch_replies_for_post(conn, topic_id, post_number)
                logger.debug(f"    fetch_replies_for_post returned {len(replies)} reply‐posts")
                if replies:
                    additional_content += "\n\n---\nReplies:\n"
                    for reply in replies:
                        logger.debug(f"      Appending reply from post_id={reply['post_id']}, author={reply['author']}")
                        additional_content += (
                            f"\n[Reply by {reply['author']}]:\n"
                            f"{reply['content']}\n"
                            f"Source URL: {reply['url']}\n"
                        )

            elif result["source"] == "markdown":
                title = result["title"]
                current_chunk_index = result["chunk_index"]
                logger.debug(f"  → Markdown chunk: title='{title}', chunk_index={current_chunk_index}")

                # Previous markdown chunk
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content
                        FROM markdown_chunks
                        WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        logger.debug(f"    Found previous markdown chunk at index {current_chunk_index - 1}")
                        additional_content += prev_chunk["content"] + "\n"
                    else:
                        logger.debug(f"    No previous markdown chunk at index {current_chunk_index - 1}")

                # Next markdown chunk
                cursor.execute("""
                    SELECT content
                    FROM markdown_chunks
                    WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    logger.debug(f"    Found next markdown chunk at index {current_chunk_index + 1}")
                    additional_content += next_chunk["content"] + "\n"
                else:
                    logger.debug(f"    No next markdown chunk at index {current_chunk_index + 1}")

            # Append additional_content if any was found
            if additional_content:
                logger.debug("    Appending additional content to original chunk.")
                enriched_result["content"] = f"{result['content']}\n\n{additional_content.strip()}"
            else:
                logger.debug("    No additional content found for this chunk.")

            enriched_results.append(enriched_result)

        logger.debug(f"Finished enriching. Total enriched results: {len(enriched_results)}.")
        return enriched_results

    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 11) Generate an answer via LLM (with sources)
# ─────────────────────────────────────────────────────────────────────────────
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    retries = 0
    while retries < max_retries:
        try:
            logger.debug(f"Preparing to generate answer. Question: '{question[:50]}…', Results: {len(relevant_results)}")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                snippet = result["content"][:1500]
                context += f"\n\n{source_type} (URL: {result['url']}):\n{snippet}"
            logger.debug(f"Combined context (first 500 chars): {context[:500].replace(chr(10), ' ')}")

            prompt = f"""Answer the following question based ONLY on the provided context. 
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Return your response in this exact format:
1. A comprehensive yet concise answer
2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer

Sources must be in this exact format:
Sources:
1. URL: [exact_url_1], Text: [brief quote or description]
2. URL: [exact_url_2], Text: [brief quote or description]

Make sure the URLs are copied exactly from the context without any changes.
"""
            logger.debug("Sending payload to LLM API.")
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug("Received answer from LLM.")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"LLM rate limit. Retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception in generate_answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

# ─────────────────────────────────────────────────────────────────────────────
# 12) Process multimodal query (text + optional image)
# ─────────────────────────────────────────────────────────────────────────────
async def process_multimodal_query(question, image_base64):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        logger.debug(f"process_multimodal_query called: question='{question[:50]}…', image_provided={image_base64 is not None}")
        if not image_base64:
            logger.debug("No image provided. Getting text-only embedding.")
            return await get_embedding(question)

        logger.debug("Image provided. Calling Vision-capable LLM for description.")
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
        image_data = f"data:image/jpeg;base64,{image_base64}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.debug(f"Received image description (first 200 chars): {image_description[:200].replace(chr(10), ' ')}")
                    combined_query = f"{question}\nImage context: {image_description}"
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}")
                    logger.debug("Falling back to text-only embedding.")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception in process_multimodal_query: {e}")
        logger.error(traceback.format_exc())
        logger.debug("Falling back to text-only embedding due to exception.")
        return await get_embedding(question)

# ─────────────────────────────────────────────────────────────────────────────
# 13) Parse LLM response (extract answer + sources)
# ─────────────────────────────────────────────────────────────────────────────
def parse_llm_response(response: str) -> Dict[str, Any]:
    try:
        logger.debug("Parsing LLM response to extract answer and sources.")
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break

        answer = parts[0].strip()
        links: List[Dict[str, str]] = []

        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)

                url_match = re.search(
                    r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)',
                    line,
                    re.IGNORECASE
                )
                text_match = re.search(
                    r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"',
                    line,
                    re.IGNORECASE
                )

                if url_match:
                    url = next((g for g in url_match.groups() if g), "").strip()
                    text = "Source reference"
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})

        logger.debug(f"Parsed answer length={len(answer)}, sources found={len(links)}.")
        return {"answer": answer, "links": links}

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.error(traceback.format_exc())
        return {"answer": "Error parsing the response from the language model.", "links": []}

# ─────────────────────────────────────────────────────────────────────────────
# 14) API route: /query
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        logger.debug(f"Received /query call: question='{request.question[:50]}…', image_provided={request.image is not None}")
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(status_code=500, content={"error": error_msg})

        conn = get_db_connection()
        try:
            # 1) Get embedding (text-only or multimodal)
            logger.debug("Calling process_multimodal_query to get embedding.")
            query_embedding = await process_multimodal_query(request.question, request.image)

            # 2) Find similar content
            logger.debug("Calling find_similar_content with embedding.")
            relevant_results = await find_similar_content(query_embedding, conn)
            logger.debug(f"find_similar_content returned {len(relevant_results)} result(s).")

            if not relevant_results:
                logger.debug("No relevant results found; returning early.")
                return {"answer": "I couldn't find any relevant information in my knowledge base.", "links": []}

            # 3) Enrich with adjacent chunks + replies
            logger.debug("Calling enrich_with_adjacent_chunks to add context and replies.")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            logger.debug(f"enrich_with_adjacent_chunks returned {len(enriched_results)} enriched result(s).")

            # 4) Generate answer via LLM
            logger.debug("Calling generate_answer to get final answer from LLM.")
            llm_response = await generate_answer(request.question, enriched_results)

            # 5) Parse LLM response into answer + sources
            logger.debug("Calling parse_llm_response to split out answer and sources.")
            result = parse_llm_response(llm_response)

            # 6) If no links from LLM, create fallback links from top‐5 relevant_results
            if not result["links"]:
                logger.debug("No sources extracted from LLM. Building fallback links from top relevant_results.")
                links: List[Dict[str, str]] = []
                unique_urls = set()
                for res in relevant_results[:5]:
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = (res["content"][:100] + "...") if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                result["links"] = links

            logger.debug(f"Returning final result: answer length={len(result['answer'])}, number of links={len(result['links'])}.")
            return result

        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": error_msg})
        finally:
            conn.close()

    except Exception as e:
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 15) Health check endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)})

# ─────────────────────────────────────────────────────────────────────────────
# 16) Run the server
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
