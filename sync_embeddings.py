import asyncio
import asyncpg
import aiohttp
import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


TB_TEST_DB_USER=os.getenv("TB_TEST_DB_USER")
TB_TEST_DB_PASSWORD=os.getenv("TB_TEST_DB_PASSWORD")
TB_TEST_DB_NAME=os.getenv("TB_TEST_DB_NAME")
TB_TEST_DB_HOST=os.getenv("TB_TEST_DB_HOST")
TB_TEST_DB_PORT=os.getenv("TB_TEST_DB_PORT")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

BATCH_SIZE = 100

LOCAL_DB = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
    "host": DB_HOST,
    "port": DB_PORT,
}

REMOTE_DB = {
    "user": TB_TEST_DB_USER,
    "password": TB_TEST_DB_PASSWORD,
    "database": TB_TEST_DB_NAME,
    "host": TB_TEST_DB_HOST,
    "port": TB_TEST_DB_PORT,
}

async def connect_db(config: dict) -> asyncpg.Pool:
    return await asyncpg.create_pool(**config, min_size=1, max_size=5)


async def fetch_embeddings(pool: asyncpg.Pool, offset: int) -> List[Dict]:
    query = """
        SELECT id, organization_id, embedding
        FROM tasks
        WHERE embedding IS NOT NULL
        ORDER BY updated_at
        LIMIT $1 OFFSET $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, BATCH_SIZE, offset)
    return [dict(row) for row in rows]


async def insert_global_embeddings(pool: asyncpg.Pool, rows: List[Dict]):
    query = """
        INSERT INTO global_embeddings (id, element_type, organization_id, embedding)
        VALUES ($1, 'TASK', $2, $3)
        ON CONFLICT (id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            organization_id = EXCLUDED.organization_id,
            updated_at = now();
    """
    async with pool.acquire() as conn:
        for row in rows:
            await conn.execute(
                query,
                row["id"],
                row["organization_id"],
                row["embedding"]
            )


async def sync_embeddings():
    local_pool = await connect_db(LOCAL_DB)
    remote_pool = await connect_db(REMOTE_DB)

    offset = 0
    total = 0

    try:
        while True:
            rows = await fetch_embeddings(local_pool, offset)

            if not rows:
                logger.info("No more rows to sync.")
                break

            await insert_global_embeddings(remote_pool, rows)

            total += len(rows)
            logger.info(f"Synced batch: {len(rows)} rows (total={total})")

            offset += BATCH_SIZE

        logger.info(" Sync completed successfully")

    finally:
        await local_pool.close()
        await remote_pool.close()


if __name__ == "__main__":
    asyncio.run(sync_embeddings())