# app.py
import streamlit as st
from agent_imdb import chat_movie_agent

# =======================
# 🎨 UI Setup
# =======================
st.set_page_config(page_title="🎬 IMDb Movie Chatbot", page_icon="🎥", layout="wide")
st.title("🎬 IMDb Movie Chatbot")
st.caption("Ask me anything about movies — powered by LangChain, Qdrant, and GPT-4o-mini.")
st.image("./assets/cinemayeay.jpg", use_column_width=True)

# =======================
# 💬 Session State
# =======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =======================
# 🎤 Chat Input
# =======================
if prompt := st.chat_input("Ask me about movies, actors, or directors..."):
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Riwayat chat singkat (opsional)
    history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]]
    )

    # =======================
    # 🤖 Jalankan Agent
    # =======================
    with st.chat_message("assistant"):
        with st.spinner("🎬 Searching the IMDb vault..."):
            try:
                response = chat_movie_agent(prompt, history)
                answer = response.get("answer", "⚠️ No answer returned.")

                # Tampilkan hasil
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Tampilkan detail tambahan
                with st.expander("🔧 Tool Calls"):
                    tool_msgs = response.get("tool_messages", [])
                    if tool_msgs:
                        for t in tool_msgs:
                            st.code(t, language="python")
                    else:
                        st.write("No tool calls recorded.")

                with st.expander("💰 Usage Details"):
                    st.code(
                        f"Input tokens: {response.get('total_input_tokens', 0)}\n"
                        f"Output tokens: {response.get('total_output_tokens', 0)}\n"
                        f"Estimated cost (IDR): {response.get('price', 0):.4f}"
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
