import streamlit as st
from langchain_openai import ChatOpenAI
from rag_tool import get_relevant_movies
import json

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Inisialisasi LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

def chat_movie_agent(question: str, history: str = ""):
    """
    Jalankan MovieGPT: jika pertanyaan soal film, ambil data dari Qdrant lalu dijawab oleh GPT.
    """
    tool_messages = []  # untuk ditampilkan di Streamlit

    # deteksi apakah pertanyaan berhubungan dengan film
    keywords = ["movie", "film", "actor", "actress", "director", "rating", "imdb", "genre"]
    if any(word in question.lower() for word in keywords):
        tool_result = get_relevant_movies(question)
        tool_messages.append(tool_result)  # biar muncul di UI

        prompt = (
            "You are MovieGPT, an expert in IMDb movie data.\n"
            "You have access to Qdrant search results containing relevant movie information.\n\n"
            f"User question: {question}\n"
            f"Relevant data from IMDb:\n{tool_result}\n\n"
            "Now, provide a concise and engaging answer in Indonesian using that data."
        )
    else:
        prompt = (
            "The user asked something unrelated to movies. "
            "Politely refuse to answer and tell them you only discuss movie-related topics."
        )

    # jalankan LLM
    response = llm.invoke(prompt)
    answer = response.content

    # estimasi token & harga sederhana
    total_input_tokens = len(prompt.split())
    total_output_tokens = len(answer.split())
    price = 17000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    # return untuk app.py
    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages or ["No Qdrant results found."]
    }
