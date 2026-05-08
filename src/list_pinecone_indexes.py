"""
List all Pinecone indexes for the API key in .env (read-only).

Usage (from repo root `Chatbot/src`):
    uv run python src/list_pinecone_indexes.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("PINECONE_API_KEY"):
    print("PINECONE_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
indexes = pc.list_indexes()

# SDK versions differ: sometimes a list of models, sometimes a response with .indexes
items = getattr(indexes, "indexes", indexes)
if items is None:
    items = []

for idx in items:
    name = getattr(idx, "name", None) or str(idx)
    row = {"name": name}
    for attr in ("dimension", "metric", "host", "spec", "status"):
        if hasattr(idx, attr):
            val = getattr(idx, attr)
            md = getattr(val, "model_dump", None)
            if callable(md):
                val = md()
            row[attr] = val
    print(json.dumps(row, default=str, indent=2))

print(f"\nTotal: {len(list(items))}")
