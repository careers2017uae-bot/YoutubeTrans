"""
Streamlit YouTube Transcript + Groq RAG Chat (Fixed Version)
- Works with latest youtube-transcript-api
"""

import os
import re
import json
import requests
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/llama3-8b-8192")  # any available model
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


# --- UTILITIES ---
def extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video ID."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)


def fetch_transcript(video_id: str) -> str:
    """Fetch transcript; supports both new and old youtube-transcript-api versions."""
    try:
        # prefer modern API
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = list_obj.find_transcript(['en']).fetch()
            except Exception:
                transcript = list(list_obj._manually_created_transcripts.values())[0].fetch()
        else:
            # fallback for legacy versions
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        return " ".join(seg['text'] for seg in transcript)
    except VideoUnavailable:
        raise RuntimeError("Video unavailable.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found.")
    except Exception as e:
        raise RuntimeError(f"Transcript fetch error: {e}")



def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_tfidf(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X


def retrieve_chunks(question, chunks, vectorizer, X, k=4):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top_idx]


def call_groq(question, context_chunks):
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    docs = [{"id": str(i), "text": c} for i, c in enumerate(context_chunks)]
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Answer the user's question using only the provided transcript."},
            {"role": "user", "content": question},
        ],
        "documents": docs,
        "max_output_tokens": 512,
    }
    r = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=60)
    data = r.json()
    return data["choices"][0]["message"]["content"]


# --- STREAMLIT UI ---
st.set_page_config(page_title="🎬 YouTube Transcript Chat", layout="wide")
st.title("🎥 YouTube Transcript + Groq Chat (RAG)")

url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=example")
if st.button("Get Transcript"):
    try:
        vid = extract_video_id(url)
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(vid)
            st.session_state['transcript'] = transcript
            chunks = chunk_text(transcript)
            vect, X = build_tfidf(chunks)
            st.session_state['chunks'] = chunks
            st.session_state['vect'] = vect
            st.session_state['X'] = X
        st.success("✅ Transcript fetched successfully!")
    except Exception as e:
        st.error(str(e))

if 'transcript' in st.session_state:
    with st.expander("View Transcript"):
        st.write(st.session_state['transcript'][:4000] + "...")

st.markdown("---")
st.subheader("💬 Ask Questions About the Video")

question = st.text_input("Your Question:")
if st.button("Ask Groq"):
    try:
        with st.spinner("Retrieving answer..."):
            rel_chunks = retrieve_chunks(question, st.session_state['chunks'], st.session_state['vect'], st.session_state['X'])
            answer = call_groq(question, rel_chunks)
        st.success("### Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Groq API + youtube-transcript-api")
