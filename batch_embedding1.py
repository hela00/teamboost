import asyncio
import asyncpg
import aiohttp
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from datetime import datetime
import time

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "teamboost")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


BATCH_SIZE = 50
REQUESTS_PER_MINUTE = 6
DELAY_BETWEEN_BATCHES = 60 / REQUESTS_PER_MINUTE
MAX_RETRIES = 8
INITIAL_RETRY_DELAY = 15
MAX_RETRY_DELAY = 120


async def connect_database() -> asyncpg.Pool:
    """Connect to PostgreSQL"""
    try:
        pool = await asyncpg.create_pool(
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connected")
        return pool
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise


async def fetch_tasks(pool: asyncpg.Pool, limit: Optional[int] = None) -> List[Dict]:
    """Fetch tasks without embeddings"""
    query = "SELECT id, title, description FROM tasks WHERE embedding IS NULL"
    if limit:
        query += f" LIMIT {limit}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    return [dict(row) for row in rows]


def prepare_text(title: str, description: Optional[str]) -> str:
    return f"{title}\n\n{description}" if description else title


async def generate_batch_embeddings(session: aiohttp.ClientSession, tasks: List[Dict]) -> List[Dict]:
    """Generate embeddings for a batch"""
    url = f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:batchEmbedContents"
    requests_payload = [
        {"model": f"models/{EMBEDDING_MODEL}",
         "content": {"parts": [{"text": prepare_text(t["title"], t.get("description"))}]},
         "taskType": "RETRIEVAL_DOCUMENT"} for t in tasks
    ]
    payload = {"requests": requests_payload}
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_API_KEY}

    retry_delay = INITIAL_RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, json=payload, headers=headers, params=params) as response:
                resp_text = await response.text()
                if response.status == 429: #rate limit error code
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limit (429), retrying in {retry_delay}s (Attempt {attempt+1})")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                        continue
                    else:
                        logger.error("Rate limit exceeded")
                        return []
                if response.status != 200:
                    logger.error(f"API error {response.status}: {resp_text}")
                    return []

                result = json.loads(resp_text)
                embeddings = result.get("embeddings", [])
                results = []
                for task, emb_data in zip(tasks, embeddings):
                    embedding = emb_data.get("values", [])
                    if embedding:
                        results.append({"task_id": task["id"], "embedding": embedding})
                    else:
                        logger.warning(f"No embedding for task {task['id']}")
                return results

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Error: {e}, retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                logger.error(f"Failed to generate embeddings: {e}")
                return []
    return []


async def save_embeddings(pool: asyncpg.Pool, embeddings: List[Dict]):
    """Save embeddings to DB"""
    if not embeddings:
        return
    async with pool.acquire() as conn:
        for item in embeddings:
            vector_str = "[" + ",".join(map(str, item["embedding"])) + "]"
            await conn.execute("UPDATE tasks SET embedding = $1::vector WHERE id = $2", vector_str, item["task_id"])
    logger.info(f"Saved {len(embeddings)} embeddings")


async def create_vector_index(pool: asyncpg.Pool):
    """Create vector index"""
    async with pool.acquire() as conn:
        index_exists = await conn.fetchval("""
            SELECT COUNT(*) FROM pg_indexes WHERE tablename='tasks' AND indexname='tasks_embedding_idx'
        """)
        if index_exists == 0:
            dim_check = await conn.fetchval("SELECT vector_dims(embedding) FROM tasks WHERE embedding IS NOT NULL LIMIT 1")
            if dim_check and dim_check <= 2000:
                await conn.execute("""
                    CREATE INDEX tasks_embedding_idx ON tasks USING hnsw (embedding vector_cosine_ops)
                    WITH (m=16, ef_construction=64)
                """)
            else:
                count = await conn.fetchval("SELECT COUNT(*) FROM tasks WHERE embedding IS NOT NULL")
                lists = max(int(count / 1000), 10)
                await conn.execute(f"""
                    CREATE INDEX tasks_embedding_idx ON tasks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists={lists})
                """)


async def process_tasks(max_tasks: Optional[int] = None):
    """Main processing function"""
    start_time = datetime.now()
    pool = await connect_database()
    tasks = await fetch_tasks(pool, limit=max_tasks)
    if not tasks:
        logger.info("No tasks to process")
        await pool.close()
        return

    session = aiohttp.ClientSession()
    successful, failed = 0, 0
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(tasks), desc="Processing tasks", unit="task") as pbar:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i + BATCH_SIZE]
            batch_start = time.time()
            embeddings = await generate_batch_embeddings(session, batch)
            if embeddings:
                await save_embeddings(pool, embeddings)
                successful += len(embeddings)
                failed += len(batch) - len(embeddings)
            else:
                failed += len(batch)
            pbar.update(len(batch))
            elapsed = time.time() - batch_start
            if i + BATCH_SIZE < len(tasks) and elapsed < DELAY_BETWEEN_BATCHES:
                await asyncio.sleep(DELAY_BETWEEN_BATCHES - elapsed)

    if successful > 0:
        await create_vector_index(pool)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total: {len(tasks)}, Success: {successful}, Failed: {failed}, Duration: {duration:.2f}s")

    await session.close()
    await pool.close()


async def main():
    import sys
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found")
        return

    max_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else None
    await process_tasks(max_tasks=max_tasks)


if __name__ == "__main__":
    asyncio.run(main())
