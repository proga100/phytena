import httpx
import os
import asyncio

async def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        print(response.json())

if __name__ == "__main__":
    asyncio.run(list_models())
