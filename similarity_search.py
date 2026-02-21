import asyncpg
import aiohttp
import asyncio
import os
from dotenv import load_dotenv
from typing import List
import json

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "teamboost")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


async def get_query_embedding(text: str) -> List[float]:
    url = f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:embedContent"

    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [{"text": text}]
        },
        "taskType": "RETRIEVAL_QUERY"
    }

    params = {"key": GOOGLE_API_KEY}
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, params=params) as resp:
            data = await resp.json()
            return data["embedding"]["values"]


async def search_tasks(query: str, limit: int = 5):
    embedding = await get_query_embedding(query)
    vector_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )

    sql = """
    SELECT id, title, description, organization_id,
           1 - (embedding <=> $1::vector) AS similarity
    FROM tasks
    WHERE embedding IS NOT NULL AND organization_id = 590789
    AND 1 - (embedding <=> $1::vector) < 0.99
    ORDER BY embedding <=> $1::vector 
    LIMIT $2;
    """

    rows = await conn.fetch(sql, vector_str, limit)
    await conn.close()

    return rows


async def main():
    query = input("Enter search query: ")

    results = await search_tasks(query, limit=5)

    print("\nTop similar tasks:\n")
    for r in results:
        print(f"[{r['similarity']:.3f}] {r['title']}")
        print(f"    {r['description']}\n")
        print(f"    {r['organization_id']}\n")


if __name__ == "__main__":
    asyncio.run(main())
