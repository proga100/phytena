import asyncio
import os
import sys
sys.path.append(os.getcwd())
from app.clients.embeddings import EmbeddingsClient

async def check_dim():
    api_key = os.getenv("GEMINI_API_KEY")
    client = EmbeddingsClient(api_key=api_key, model="gemini-embedding-2")
    
    # 1. Default
    emb = await client.get_embedding("test")
    print(f"Default dim: {len(emb)}")
    
    # 2. Requested 768
    emb_768 = await client.get_embedding("test", output_dimensionality=768)
    print(f"Requested 768 dim: {len(emb_768)}")

if __name__ == "__main__":
    asyncio.run(check_dim())
