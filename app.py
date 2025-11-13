"""
Streamlit YouTube Transcription + RAG Chat (Groq)
- Expects GROQ_API_KEY in environment (and optional GROQ_MODEL)
- Uses youtube-transcript-api to fetch transcripts (no YouTube Data API key required).
- Uses TF-IDF + cosine similarity to pick relevant chunks, then calls Groq Chat Completions.
"""

import os
import re
import json
import time
import requests
from typing import List, Tuple

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Configuration / Defaults
# ----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # We'll still let the app run, but API calls will fail until the key is set.
    pass

GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/compound-mini")
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Chunking parameters
CHUNK_SIZE = 900    # characters per chunk
CHUNK_OVERLAP = 200  # overlap characters

# ----------------------------
# Helpers
# ----------------------------

def extract_video_id(url: str) -> str:
    """
    Get YouTube video id from a variety of URL formats.
    """
    # Common patterns
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([A-Za-z0-9_-]{10,})",
        r"([A-Za-z0-9_-]{11})"  # fallback - 11 char id
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract video id from the URL. Provide a full YouTube URL.")

def fetch_transcript(video_id: str, languages=None) -> List[dict]:
    """
    Fetch transcript using youtube-transcript-api.
    Returns a list of {'text':..., 'start':..., 'duration':...}
    """
    try:
        if languages:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list
    except VideoUnavailable:
        raise RuntimeError("Video is unavailable.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Transcript fetch error: {e}")

def transcript_to_text(transcript: List[dict]) -> str:
    """
    Convert the list-of-segments to a single text string.
    """
    return " ".join(seg.get("text", "").strip() for seg in transcript).strip()

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text into overlapping chunks for retrieval.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_tfidf_index(chunks: List[str]):
    """
    Fit a TF-IDF vectorizer on chunks and return (vectorizer, vectors).
    """
    vectorizer = TfidfVectorizer(strip_accents="unicode", stop_words="english")
    vectors = vectorizer.fit_transform(chunks) if chunks else None
    return vectorizer, vectors

def retrieve_top_chunks(question: str, chunks: List[str], vectorizer, vectors, top_k=4) -> List[Tuple[int, str]]:
    """
    Return top_k (index, chunk) most relevant to question by cosine similarity.
    """
    if not chunks:
        return []
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, vectors)[0]  # shape (n_chunks,)
    top_idx = sims.argsort()[::-1][:top_k]
    results = [(int(i), chunks[i]) for i in top_idx if sims[i] > 0]
    return results

def call_groq_chat(question: str, selected_chunks: List[Tuple[int, str]], api_key: str, model: str = GROQ_MODEL, max_tokens: int = 512):
    """
    Call Groq Chat Completions (OpenAI-compatible endpoint).
    We pass the selected transcript chunks as 'documents' to the API (simple RAG).
    """
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Set environment variable GROQ_API_KEY before using the Groq API.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build documents array from chunks (document id + text)
    documents = [{"id": str(i), "text": text} for i, text in selected_chunks]

    system_message = (
        "You are a helpful assistant. Use ONLY the provided documents (transcript snippets) to answer user questions about the video. "
        "If the answer is not contained in the documents, say you don't know and suggest looking at the video."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        # Groq supports 'documents' (per API reference) to pass context for RAG-like usage.
        "documents": documents,
        "max_output_tokens": max_tokens,
        "temperature": 0.12,
    }

    resp = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        # Try to extract error
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Groq API error {resp.status_code}: {err}")

    data = resp.json()
    # Groq's OpenAI-compatible response object is similar: check choices / message
    # Different Groq SDKs return slightly different wrappers; attempt common patterns:
    # Try "choices"[0]["message"]["content"] or "choices"[0]["text"] or top-level "text"
    try:
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            # OpenAI-like
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            if "text" in choice:
                return choice["text"]
        # fallback: some Groq endpoints might return top-level fields
        if "text" in data:
            return data["text"]
        # fallback: try 'output_text' (Responses API)
        if "output_text" in data:
            return data["output_text"]
    except Exception:
        pass

    # If nothing matched, return full JSON
    return json.dumps(data, indent=2)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="YouTube → Transcript + RAG (Groq)", layout="wide")
st.title("YouTube Transcript + Q&A (Streamlit + Groq RAG)")

st.markdown(
    """
Enter a **YouTube video URL**, click **Get Transcript**.  
Then ask questions about the video — the app will retrieve the most relevant transcript chunks and query Groq (RAG).
"""
)

col1, col2 = st.columns([2, 1])

with col1:
    video_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    get_button = st.button("Get Transcript")

with col2:
    st.write("Groq settings")
    st.text("Model (env GROQ_MODEL):")
    st.code(GROQ_MODEL)
    st.write("Make sure GROQ_API_KEY is set in environment.")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Set it as an environment variable before asking questions to Groq.")

# Session caching of transcript + index
if "transcript_text" not in st.session_state:
    st.session_state["transcript_text"] = ""
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "video_id" not in st.session_state:
    st.session_state["video_id"] = None

if get_button and video_url:
    try:
        vid = extract_video_id(video_url)
        st.session_state["video_id"] = vid
        with st.spinner("Fetching transcript..."):
            segments = fetch_transcript(vid)
            text = transcript_to_text(segments)
            st.session_state["transcript_text"] = text
            chunks = chunk_text(text)
            st.session_state["chunks"] = chunks
            vect, vecs = build_tfidf_index(chunks)
            st.session_state["vectorizer"] = vect
            st.session_state["vectors"] = vecs
        st.success("Transcript fetched and indexed.")
    except Exception as e:
        st.error(f"Could not fetch transcript: {e}")

# Show transcript preview
with st.expander("Transcript (preview)"):
    if st.session_state["transcript_text"]:
        st.write(st.session_state["transcript_text"][:5000] + ("..." if len(st.session_state["transcript_text"])>5000 else ""))
        if st.button("Show full transcript"):
            st.write(st.session_state["transcript_text"])
    else:
        st.info("No transcript yet. Enter a YouTube URL and click 'Get Transcript'.")

# Q&A UI
st.markdown("---")
st.header("Ask questions about the video")
question = st.text_input("Your question about the video")
ask = st.button("Ask Groq")

if ask:
    if not st.session_state["transcript_text"]:
        st.warning("Please fetch a transcript first.")
    elif not question.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            chunks = st.session_state["chunks"]
            vect = st.session_state["vectorizer"]
            vecs = st.session_state["vectors"]
            selected = retrieve_top_chunks(question, chunks, vect, vecs, top_k=5)
            # If similarity returned nothing, fall back to top sequential chunks
            if not selected:
                selected = list(enumerate(chunks[:5]))
        # Show what we will send as context
        st.subheader("Context provided to the model (top chunks)")
        for idx, chunk in selected:
            st.text_area(f"chunk {idx}", value=chunk[:2000], height=120)

        # Call Groq
        try:
            with st.spinner("Calling Groq..."):
                answer = call_groq_chat(question, selected, api_key=GROQ_API_KEY, model=GROQ_MODEL)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Groq call failed: {e}")

# Footer / credits
st.markdown("---")
st.markdown(
    """
**Notes**
- Transcript is fetched with the `youtube-transcript-api` package (auto-generated or uploaded captions). Accuracy depends on YouTube captions.  
- This app uses a simple TF-IDF retrieval for RAG; for production, consider embeddings + vector DB for better recall.  
- Set `GROQ_API_KEY` (and optionally `GROQ_MODEL`) in your environment before using the Groq integration.
"""
)
