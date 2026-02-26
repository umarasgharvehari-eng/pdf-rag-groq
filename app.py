import os, json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="PDF RAG Chat (Groq)", page_icon="📄")
st.title("📄 PDF RAG Chat (Groq)")

# ---- Secrets / Sidebar ----
# Prefer Streamlit Secrets in deployment, but sidebar is helpful for local testing.
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API Key", type="password")
    model_name = st.text_input("Groq Model", value="llama-3.1-8b-instant")
    top_k = st.slider("Top K Chunks", 2, 10, 4)

# Load from secrets if set
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_store():
    index = faiss.read_index("index.faiss")
    with open("chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return index, data["chunks"], data["meta"]

def retrieve(query: str, k: int = 4):
    embedder = load_embedder()
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")

    index, chunks, meta = load_store()
    D, I = index.search(q, k)

    results = []
    for rank, idx in enumerate(I[0], start=1):
        results.append({
            "rank": rank,
            "text": chunks[idx],
            "meta": meta[idx],
            "distance": float(D[0][rank-1]),
        })
    return results

def groq_chat(model_name: str, system: str, user: str) -> str:
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about your PDFs...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not os.getenv("GROQ_API_KEY"):
        with st.chat_message("assistant"):
            st.error("Add GROQ_API_KEY in Streamlit Secrets or paste it in the sidebar.")
        st.stop()

    docs = retrieve(prompt, k=top_k)

    context_blocks = []
    for d in docs:
        src = d["meta"].get("source", "unknown")
        page = d["meta"].get("page", "unknown")
        context_blocks.append(
            f"[{d['rank']}] source={src}, page={page}\n{d['text']}"
        )

    context = "\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Use citations like [1], [2] referring to the context blocks."
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"

    answer = groq_chat(model_name=model_name, system=system, user=user)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("🔎 Retrieved context"):
        for d in docs:
            src = d["meta"].get("source", "unknown")
            page = d["meta"].get("page", "unknown")
            st.markdown(f"**[{d['rank']}]** {src} — page {page}")
            st.write(d["text"])
            st.divider()
