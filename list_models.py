"""Print the model names a provider offers.

Usage:  python list_models.py        (main LLM)
        python list_models.py 2      (fallback LLM_2_*, and so on)
Pick a chat model from the list and put it in .env.local.
"""
import asyncio
import sys

from rag import PROVIDERS, client_for


async def main() -> None:
    wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    provider = next((p for p in PROVIDERS if p["id"] == wanted), None)
    if provider is None:
        sys.exit(f"LLM {wanted} is not configured in .env.local")
    page = await client_for(provider).models.list()
    for name in sorted(model.id for model in page.data):
        print(name)


if __name__ == "__main__":
    asyncio.run(main())
