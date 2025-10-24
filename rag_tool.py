import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI

# Load secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Init clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
COLLECTION_NAME = "imdb_movies"  # sama dengan yang kamu buat di load_imdb_to_qdrant.py


def get_embedding(text: str):
    """Generate embedding pakai model OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text.replace("\n", " ")
    )
    return response.data[0].embedding


def get_relevant_movies(query: str, top_k: int = 5):
    """Ambil film relevan dari Qdrant berdasarkan query"""
    query_vector = get_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    if not results:
        return "No relevant movies found."

    response_lines = []
    for hit in results:
        payload = hit.payload
        title = payload.get("title", "Unknown")
        year = payload.get("year", "?")
        genre = payload.get("genre", "?")
        rating = payload.get("rating", "?")
        overview = payload.get("overview", "")
        score = hit.score

        response_lines.append(
            f"🎬 **{title}** ({year}) — {genre}, ⭐ {rating}\n> {overview}\n"
        )

    return "\n\n".join(response_lines)
