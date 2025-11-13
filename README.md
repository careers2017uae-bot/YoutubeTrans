# YoutubeTrans
The app:

Accepts a YouTube URL and fetches the transcript (auto or manual captions) using youtube-transcript-api (no YouTube Data API key required). 
PyPI

Chunks the transcript and builds a lightweight TF–IDF retrieval index (scikit-learn).

On a user question, finds the most relevant transcript chunks and calls Groq’s chat completions endpoint (OpenAI-compatible endpoint) with those chunks as documents (simple RAG). Groq docs show chat completions endpoint and models (we default to groq/compound-mini but you can override via env var). 
GroqCloud
+1
