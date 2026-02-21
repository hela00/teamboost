import asyncio
import asyncpg
import aiohttp
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()

# Database config
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "teamboost")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding API
EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

POLL_INTERVAL = 60
MAX_TASKS_PER_RUN = 10
DELAY_BETWEEN_REQUESTS = 5


#Connect to PostgreSQL
async def connect_database() -> asyncpg.Pool:
    pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
        min_size=2,
        max_size=5
    )
    logger.info("Connected to database")
    return pool

#Fetch tasks without embeddings
async def fetch_tasks_without_embeddings(pool: asyncpg.Pool, limit: int) -> List[Dict[str, Any]]:
    
    query = """
        SELECT id, title, description
        FROM tasks
        WHERE embedding IS NULL
        LIMIT $1
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit)
    return [dict(row) for row in rows]


def prepare_text(title: str, description: Optional[str]) -> str:
    return f"{title}\n\n{description}" if description else title


async def generate_embedding(session: aiohttp.ClientSession, text: str) -> Optional[List[float]]:
    url = f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:embedContent"
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_API_KEY}
    
    try:
        async with session.post(url, json=payload, headers=headers, params=params) as response:
            if response.status != 200:
                logger.error(f"API error: {await response.text()}")
                return None
            result = await response.json()
            embedding = result.get('embedding', {}).get('values', [])
            return embedding if embedding else None
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def save_embedding(pool: asyncpg.Pool, task_id: str, embedding: List[float]):

    vector_str = '[' + ','.join(map(str, embedding)) + ']'
    async with pool.acquire() as conn:
        await conn.execute("UPDATE tasks SET embedding = $1::vector WHERE id = $2", vector_str, task_id)


async def process_pending_tasks(pool: asyncpg.Pool, session: aiohttp.ClientSession):
    """Process tasks needing embeddings"""
    tasks = await fetch_tasks_without_embeddings(pool, MAX_TASKS_PER_RUN)
    if not tasks:
        logger.info("No tasks need embeddings")
        return 0
    
    logger.info(f"Processing {len(tasks)} tasks")
    successful = 0
    for i, task in enumerate(tasks, 1):
        logger.info(f"Task {i}/{len(tasks)}: {task['title'][:50]}")
        text = prepare_text(task['title'], task.get('description'))
        embedding = await generate_embedding(session, text)
        if embedding:
            await save_embedding(pool, task['id'], embedding)
            successful += 1
        else:
            logger.error(f"Failed to generate embedding for task {task['id']}")
        if i < len(tasks):
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    logger.info(f"Processed {successful}/{len(tasks)} tasks successfully")
    return successful


async def run_once():
    pool = await connect_database()
    session = aiohttp.ClientSession()
    try:
        await process_pending_tasks(pool, session)
    finally:
        await session.close()
        await pool.close()


async def run_continuous():
    pool = await connect_database()
    session = aiohttp.ClientSession()
    try:
        while True:
            await process_pending_tasks(pool, session)
            logger.info(f"Sleeping for {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)
    finally:
        await session.close()
        await pool.close()


async def main():
    import sys
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set")
        return
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    if mode in ("continuous", "-c"):
        logger.info("Starting continuous mode")
        await run_continuous()
    else:
        logger.info("Running single cycle")
        await run_once()


if __name__ == "__main__":
    asyncio.run(main())
