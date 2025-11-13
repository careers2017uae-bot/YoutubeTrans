"""
Streamlit YouTube Transcript + RAG Chat (robust method using yt-dlp)
- Uses yt_dlp to fetch subtitle URLs (manual preferred, then auto captions)
- Downloads and parses VTT/SRT to text in memory (no fragile library calls)
- Uses a simple TF-IDF retriever and Groq Chat Completions for RAG
- Requires GROQ_API_KEY in env
"""

import os
import re
import io
import json
import requests
import streamlit as st
from typing import Optional, Dict, List, Tuple

# yt_dlp for robust subtitle URL extraction
from yt_dlp import YoutubeDL

# simple retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Configuration
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
TOP_K = 5
REQUEST_TIMEOUT = 20  # seconds for subtitle fetch

# -------------------------
# Utilities: subtitle extraction & parsing
# -------------------------
def extract_info_with_ytdlp(url: str) -> Dict:
    """
    Use yt_dlp to extract video info (metadata includes 'subtitles' and 'automatic_captions').
    Returns the info dict from yt_dlp.extract_info(download=False).
    """
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

def choose_subtitle_url(info: Dict, prefer_langs: List[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Choose the best subtitle track URL.
    Preference order:
      1. Manual subtitles (info['subtitles'])
      2. Automatic captions (info['automatic_captions'])
    prefer_langs is a list like ['en','en-US'] to prefer English; if None, try 'en' then any.
    Returns (subtitle_url, ext) or (None, None) if not found.
    """
    prefer_langs = prefer_langs or ['en', 'en-US', 'en-GB']
    # Helper to inspect a dict like info['subtitles'] or info['automatic_captions']
    def pick_from(subdict):
        if not isinstance(subdict, dict):
            return None
        # Try preferred languages first
        for lang in prefer_langs:
            if lang in subdict and subdict[lang]:
                # choose a format that is VTT or SRT if available
                formats = subdict[lang]
                for f in formats:
                    ext = f.get('ext', '').lower()
                    url = f.get('url')
                    if ext in ('vtt', 'webvtt', 'srt', 'srv3', 'ttml'):
                        return url, ext
                # fallback to first available
                f = formats[0]
                return f.get('url'), f.get('ext')
        # no preferred languages -> pick any one
        for lang, formats in subdict.items():
            if formats:
                f = formats[0]
                return f.get('url'), f.get('ext')
        return None

    # Manual subtitles first
    url_ext = pick_from(info.get('subtitles') or {})
    if url_ext:
        return url_ext
    # fall back to automatic captions
    url_ext = pick_from(info.get('automatic_captions') or {})
    if url_ext:
        return url_ext
    return None, None

def download_subtitle_text(sub_url: str, ext: Optional[str]) -> str:
    """
    Download subtitle file from URL and parse into plain text.
    Supports VTT and SRT and basic XML/TTML like formats heuristically.
    """
    if not sub_url:
        raise RuntimeError("No subtitle URL provided.")
    try:
        resp = requests.get(sub_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.text
    except Exception as e:
        raise RuntimeError(f"Failed to download subtitle: {e}")

    # Try to detect VTT
    if raw.lstrip().startswith("WEBVTT") or (ext and ext.lower() in ('vtt','webvtt')):
        return parse_vtt_to_text(raw)
    # SRT (common)
    if re.search(r"\d+\s*\n\d{2}:\d{2}:\d{2}", raw) or (ext and ext.lower() == 'srt'):
        return parse_srt_to_text(raw)
    # TTML / XML-like
    if raw.lstrip().startswith('<?xml') or '<tt' in raw[:200].lower():
        return parse_xml_subs_to_text(raw)

    # As a last resort, strip timestamps and tags heuristically
    return heuristic_strip_subtitles(raw)

def parse_vtt_to_text(vtt: str) -> str:
    # Remove WEBVTT header and timestamps and cue settings
    lines = vtt.splitlines()
    cleaned = []
    for line in lines:
        # skip header / cue identifiers (numbers or 'NOTE')
        if line.strip().upper().startswith("WEBVTT"):
            continue
        # skip cue settings like "00:00:01.000 --> 00:00:04.000"
        if re.match(r'^\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->', line) or re.match(r'^\s*\d{2}:\d{2}\.\d{3}\s*-->', line):
            continue
        # common arrow format with milliseconds
        if re.match(r'^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', line):
            continue
        # skip empty lines that separate cues
        cleaned.append(line)
    text = "\n".join(cleaned)
    # Remove any HTML tags left (some subtitles contain <b>, <i> etc.)
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_srt_to_text(srt: str) -> str:
    # Remove numeric indexes and timestamps
    # Remove lines that are just numbers or timestamps
    lines = srt.splitlines()
    cleaned = []
    for line in lines:
        if line.strip().isdigit():
            continue
        if re.match(r'^\s*\d{2}:\d{2}:\d{2}', line):
            continue
        cleaned.append(line)
    text = " ".join(cleaned)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_xml_subs_to_text(xmltext: str) -> str:
    # crude XML tag removal; keep only text
    text = re.sub(r'<[^>]+>', ' ', xmltext)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def heuristic_strip_subtitles(raw: str) -> str:
    # Remove timestamps like 00:01:02,000 --> 00:01:05,000 and numbers
    cleaned = re.sub(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}', ' ', raw)
    cleaned = re.sub(r'\d+\n', ' ', cleaned)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def fetch_transcript_via_ytdlp(url: str) -> str:
    """
    End-to-end: use yt-dlp to extract subtitle URL, download and parse into plain text.
    Raises RuntimeError on failure.
    """
    try:
        info = extract_info_with_ytdlp(url)
    except Exception as e:
        raise RuntimeError(f"yt-dlp extract error: {e}")

    sub_url, ext = choose_subtitle_url(info)
    if not sub_url:
        # No subtitles found; give helpful message including whether video has closed captions metadata
        has_caps = bool(info.get('automatic_captions') or info.get('subtitles'))
        if not has_caps:
            raise RuntimeError("No subtitles or automatic captions found for this video.")
        else:
            raise RuntimeError("Subtitle info present but no downloadable URL found.")
    # Download and parse
    try:
        text = download_subtitle_text(sub_url, ext)
        if not text.strip():
            raise RuntimeError("Downloaded subtitle was empty after parsing.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to download/parse subtitle: {e}")

# -------------------------
# Retrieval & model call
# -------------------------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_index(chunks: List[str]):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X

def retrieve_top_chunks(question: str, chunks: List[str], vectorizer, X, top_k=TOP_K):
    if not chunks:
        return []
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X)[0]
    idx = sims.argsort()[::-1][:top_k]
    results = [(int(i), chunks[i]) for i in idx if sims[i] > 0]
    # if none have positive sim, return top_k first chunks as fallback
    if not results:
        results = list(enumerate(chunks[:top_k]))
    return results

def call_groq_chat(question: str, selected_chunks: List[Tuple[int, str]]):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    documents = [{"id": str(i), "text": txt} for i, txt in selected_chunks]
    system_message = (
        "You are a helpful assistant. Use ONLY the provided documents (transcript snippets) to answer the user's question "
        "about the YouTube video. If you cannot find the answer in the documents, say you don't know and suggest watching the video."
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        "documents": documents,
        "max_output_tokens": 512,
        "temperature": 0.1,
    }
    resp = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Groq API error {resp.status_code}: {err}")
    data = resp.json()
    # try to extract typical fields
    if "choices" in data and data["choices"]:
        ch = data["choices"][0]
        if "message" in ch and "content" in ch["message"]:
            return ch["message"]["content"]
        if "text" in ch:
            return ch["text"]
    if "output_text" in data:
        return data["output_text"]
    return json.dumps(data, indent=2)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="YouTube → Transcript + RAG (yt-dlp)", layout="wide")
st.title("YouTube Transcript + Q&A using RAG (by Engr. Bilal)")

st.markdown(
    """
This app extracts subtitles using **yt-dlp** (manual subtitles preferred, otherwise auto captions),
parses them into text in memory, then answers questions using a simple RAG pipeline with Groq.
"""
)

col1, col2 = st.columns([3,1])
with col1:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    btn_fetch = st.button("Get Transcript (yt-dlp)")
with col2:
    st.write("Groq settings")
    st.code(GROQ_MODEL)
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Set environment variable before asking questions.")

if btn_fetch and url:
    try:
        with st.spinner("Extracting subtitle URL and downloading subtitles..."):
            transcript_text = fetch_transcript_via_ytdlp(url)
            st.session_state['transcript_text'] = transcript_text
            # build chunks & index
            chunks = chunk_text(transcript_text)
            vect, X = build_index(chunks)
            st.session_state['chunks'] = chunks
            st.session_state['vectorizer'] = vect
            st.session_state['X'] = X
        st.success("Transcript extracted and indexed ✅")
    except Exception as e:
        st.error(f"Transcript fetch error: {e}")

if 'transcript_text' in st.session_state:
    with st.expander("Transcript Preview"):
        text = st.session_state['transcript_text']
        if len(text) > 5000:
            st.write(text[:5000] + "...")
        else:
            st.write(text)

st.markdown("---")
st.header("Ask a question about the video")
question = st.text_input("Question about the video")
if st.button("Ask"):
    if 'transcript_text' not in st.session_state:
        st.warning("Fetch a transcript first.")
    elif not question.strip():
        st.warning("Type a question.")
    else:
        try:
            with st.spinner("Retrieving relevant chunks..."):
                chunks = st.session_state['chunks']
                vectorizer = st.session_state['vectorizer']
                X = st.session_state['X']
                selected = retrieve_top_chunks(question, chunks, vectorizer, X, top_k=TOP_K)
            st.subheader("Context sent to model (top chunks)")
            for idx, chunk in selected:
                st.text_area(f"chunk {idx}", value=chunk[:2000], height=120)

            with st.spinner("Calling Groq..."):
                answer = call_groq_chat(question, selected)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while answering: {e}")

st.markdown("---")
st.caption("If subtitles aren't available, the app cannot generate a transcript. yt-dlp relies on YouTube's available captions (manual or auto).")
