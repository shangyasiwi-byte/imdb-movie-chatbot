import pandas as pd
import numpy as np
import uuid
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import streamlit as st

# ================================
# 1️⃣ Load API Keys (dari Streamlit secrets)
# ================================
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ================================
# 2️⃣ Inisialisasi Client
# ================================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Nama koleksi (collection) Qdrant
COLLECTION_NAME = "imdb_movies"


# ================================
# 3️⃣ Membuat / Reset Collection
# ================================
def create_collection(vector_size=1536):
    """Membuat ulang collection Qdrant dengan konfigurasi cosine distance."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' berhasil dibuat di Qdrant.")


# ================================
# 4️⃣ Load Dataset
# ================================
def load_dataset(csv_path="imdb_top_1000.csv"):
    """Membaca dan membersihkan dataset IMDb."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Overview"])
    print(f"📊 Jumlah data terbaca: {len(df)} baris")
    return df


# ================================
# 5️⃣ Generate Embedding
# ================================
def get_embedding(text):
    """Membuat embedding dari teks menggunakan OpenAI."""
    text = text.replace("\n", " ")
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Gagal membuat embedding: {e}")
        return np.zeros(1536).tolist()  # fallback


# ================================
# 6️⃣ Insert Data ke Qdrant
# ================================
def insert_data(limit=300):
    """Membuat embedding dan upload data IMDb ke Qdrant."""
    df = load_dataset().head(limit)
    points = []

    print(f"🚀 Membuat embedding & mengunggah {len(df)} data ke Qdrant...")
    for i, row in df.iterrows():
        text = (
            f"Title: {row['Series_Title']} | "
            f"Year: {row['Released_Year']} | "
            f"Genre: {row['Genre']} | "
            f"Rating: {row['IMDB_Rating']} | "
            f"Overview: {row['Overview']} | "
            f"Director: {row['Director']}"
        )

        embedding = get_embedding(text)
        payload = {
            "title": row["Series_Title"],
            "year": row["Released_Year"],
            "genre": row["Genre"],
            "rating": row["IMDB_Rating"],
            "overview": row["Overview"],
            "director": row["Director"]
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)

        # Optional: jeda kecil agar tidak kena rate-limit
        time.sleep(0.25)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Berhasil upload {len(points)} item ke Qdrant.")


# ================================
# 7️⃣ Tes Pencarian
# ================================
def test_search(query="romantic movie"):
    """Menguji pencarian di Qdrant."""
    query_vector = get_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    print("\n🔍 Hasil pencarian:")
    for r in results:
        p = r.payload or {}
        print(f"🎬 {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}) | "
              f"{p.get('genre', '')} | ⭐ {p.get('rating', 'N/A')}")
        print(f"   {p.get('overview', '')[:200]}...\n")


# ================================
# 8️⃣ Jalankan Langsung
# ================================
if __name__ == "__main__":
    create_collection()
    insert_data(limit=100)   # bisa ubah ke 1000 kalau mau semua
    test_search(query="science fiction about space")
