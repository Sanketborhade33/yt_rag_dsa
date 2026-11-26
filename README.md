# YT Lecture RAG: find the exact second a topic was explained

A semantic search engine over a 68.7-hour YouTube DSA lecture playlist. Ask a doubt in Hinglish or English and it returns the exact moments in the exact lectures where that topic was explained. Click a result and the video plays inside the page, starting at that second.

> Most RAG demos return a blob of text. Mine returns *a place in a video*.


```
Query:  "memoization aur tabulation ka difference kahan explain kiya hai?"

 1  DP Lecture 3 - Memoization Deep Dive      @ 12:04   (click to play)
 2  DP Lecture 5 - Tabulation Recipe          @ 04:31   (click to play)
```

**Scale:** 126 lectures, 68.7 hours, ~2,900 searchable chunks, about one second per search.

---

## What I built

I designed and implemented the full pipeline, end to end:

1. **Ingestion:** playlist listing and audio download with `yt-dlp`, with retry, backoff and bot-detection handling.
2. **Transcription:** batched `faster-whisper large-v3` on GPU, with an atomic transcript cache so a crash never corrupts data.
3. **Chunking:** my own time-window chunker that preserves timestamps and filters Whisper hallucinations.
4. **Indexing:** embeddings into Qdrant with deterministic IDs, so re-running never duplicates.
5. **Retrieval:** vector search plus a title-aware re-ranker I added after measuring where plain dense search failed.
6. **Grounded answers (optional):** an LLM summary with three guards against ungrounded output.
7. **API and UI:** a FastAPI backend and a single-page frontend with an in-page YouTube player.
8. **Tooling:** a CLI, a preflight checker and a golden-set evaluation harness.

---

## Architecture

```
                         INGEST (run once, GPU)
Playlist URL
   |  yt-dlp
   v
Audio (m4a) --> faster-whisper large-v3 (batched) --> transcripts/<id>.json   (cached, atomic writes)
                                                          |
                                         time-window chunking (75 s, 15 s overlap)
                                                          |
                                     sentence-transformers (all-MiniLM-L6-v2, 384-d)
                                                          |
                                                       Qdrant

                              QUERY (every search)
Question --> embed --> Qdrant top-20 --> distance filter (<= 0.6)
         --> title-boost re-rank --> top-6
         --> /search : timestamps + previews (no LLM)
         --> /ask    : LLM summary with [n] citations (optional)
```

| Layer | Choice |
|---|---|
| Playlist / audio | `yt-dlp` |
| Transcription | `faster-whisper` (CTranslate2), `large-v3`, batch 8 |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` |
| Vector store | Qdrant (embedded local mode or Qdrant Cloud) |
| LLM (optional) | Gemini or Groq |
| API | FastAPI + Uvicorn |
| Frontend | HTML/CSS/JS + YouTube IFrame API |
| CLI / packaging | Typer + Rich, `uv` |

---

## Engineering decisions and what I measured

### 1. Chunk by time, not by characters
Generic text splitters concatenate everything into one string and lose the timestamps, and the timestamp is the whole product here. My chunker works on the segment list: 75-second windows with 15 seconds of overlap, it never splits a segment, and it prefixes the lecture title to each chunk so a chunk from minute 34 still carries its topic.

### 2. Transcripts are the expensive artifact, so cache them safely
Transcribing 68 hours takes about 8.5 GPU-hours. Everything after it takes minutes. Transcripts are written to a temp file and renamed atomically, so a Ctrl-C or power cut can never leave a truncated file that the cache trusts. Changing the chunk size or embedding model never requires re-transcribing.

### 3. Batched Whisper: 3x faster, same quality

| Config | Speed | 68.7 h playlist |
|---|---|---|
| `large-v3` sequential, beam 5 | 2.5x realtime | ~27 h |
| `large-v3` batch 8, beam 5 (default) | 8.1x | ~8.5 h |
| `large-v3` batch 8, beam 1 | 10.5x | ~6.5 h |

I kept beam 5. Greedy decoding is the likeliest to slip on unusual technical terms in accented speech.

### 4. A small model plus better ranking beat a big model

| Model | Download | Reindex (CPU) | Top-1 | Top-5 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` (default) | 87 MB | 1.6 min | 11/12 | 12/12 |
| `BAAI/bge-m3` | 4.35 GB | 55 min | 12/12 | 12/12 |

Dense similarity over a 75-second ramble dilutes the topic: a chunk about ASCII values outranked the "Number of Islands" lecture for the query "number of islands". I added a title-boost re-ranker (`distance - 0.06 x matched title words`, with a Hinglish stopword list and light stemming). On my 12-question test set, top-1 accuracy went **from 9/12 to 12/12**, which let me ship the 87 MB model instead of the 4 GB one.

### 5. Retrieval first, generation optional
`/search` never calls an LLM, so it is instant, free, unlimited and cannot hallucinate. The "Summarise" button is optional, with three guards:
1. If no chunk passes the distance cutoff, return the refusal **without calling the LLM**.
2. If the model says the excerpts don't cover the question, honour that.
3. If the answer has no citations, mark it ungrounded and attach no links.

Citations are renumbered so only the sources actually used are shown.

### 6. Tuning the cutoff with data, not guesses
I tested `MAX_DISTANCE` against 20 in-syllabus and 10 off-topic questions. 0.75 never fired, and 0.5 refused real questions like "hashmap kab use karna chahiye". The two score populations overlap, so no cutoff separates them perfectly. I settled on **0.6**, which kept 20/20 real questions and refused 10/10 off-topic ones, and treat it as a coarse pre-filter with the model's refusal as the semantic check.

### 7. Handling Whisper loop hallucinations
On silence or a music sting, Whisper can repeat a phrase indefinitely. VAD filtering prevents most of it, and `is_repetitive()` drops chunks where one sentence makes up 60% or more of the text, since those match every query and say nothing.

### 8. Small details that matter
- Citation links rewind **5 seconds**, because the explanation usually starts just before the matched chunk.
- Deterministic point IDs (`uuid5` of `video_id:start_sec`) make ingestion idempotent.
- The collection name carries the embedding dimension, so changing models never causes a dimension-mismatch error.
- The player queues clicks until the YouTube API loads, and falls back to a plain YouTube link if an ad-blocker blocks it.
- The embedding model preloads at server startup, so the first user doesn't wait 14 seconds.

---

## Quick start (no GPU needed)

Transcripts and a prebuilt vector index are included in the repo.

```bash
git clone <https://github.com/Sanketborhade33/yt_rag>
cd ytscraper
uv sync                     # Python 3.11 or 3.12; `pip install uv` if needed
uv run ytrag load           # loads the prebuilt index (~20 s, nothing is re-embedded)
uv run ytrag serve
```

Open <http://127.0.0.1:8000>. Next time, only `uv run ytrag serve` is needed.

Search works with **no API keys**. For the optional "Summarise" button, copy `.env.example` to `.env` and add a free `GEMINI_API_KEY` ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) or `GROQ_API_KEY` ([console.groq.com/keys](https://console.groq.com/keys)).

> The local vector store allows one process at a time, so stop `ytrag serve` before running `ytrag ask`. Setting `QDRANT_URL` to a hosted Qdrant removes this limit.

---

## CLI

```bash
uv run ytrag search "sliding window"     # retrieval only, shows distances
uv run ytrag ask "kadane ka intuition"   # retrieval + LLM answer
uv run ytrag stats                       # index size per video
uv run ytrag eval --verbose              # run the golden set
uv run ytrag preflight                   # check keys, store, models, LLM
uv run ytrag reindex [--replace]         # rebuild index from cached transcripts
uv run ytrag serve --port 8001           # custom port
uv run ytrag clean-audio                 # delete downloaded audio
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Ranked timestamps, no LLM. Body: `{"question": "...", "top_k": 6}` |
| `POST` | `/ask` | LLM summary with citations (rate-limited per IP) |
| `GET` | `/meta` | Lecture count and hours indexed |
| `GET` | `/stats` | Collection size and embedding model |
| `GET` | `/health` | Health check |

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "dijkstra kaise kaam karta hai"}'
```

---

## Project structure

```
ytscraper/
├── api/
│   ├── main.py             FastAPI app (endpoints, rate limit, model preload)
│   └── static/index.html   Frontend (HTML + CSS + JS)
├── ytrag/
│   ├── config.py           Env-driven settings
│   ├── models.py           Video / Segment / Chunk dataclasses
│   ├── playlist.py         yt-dlp listing + download
│   ├── transcribe.py       faster-whisper + transcript cache
│   ├── chunk.py            Time-window chunking + hallucination filter
│   ├── embed.py            Embedder
│   ├── index.py            Qdrant, search, title-boost, import/export
│   ├── answer.py           Grounding guards, LLM call, search_only()
│   ├── evaluate.py         Golden-set evaluation
│   ├── util.py             Retry + network-wait helpers
│   └── cli.py              `ytrag` commands
├── transcripts/            126 cached Whisper transcripts
├── index/vectors.npz       Prebuilt vectors (384-d, float16)
├── eval/golden.json        Evaluation set
└── pyproject.toml
```

---

## Configuration

Defaults live in [`ytrag/config.py`](ytrag/config.py).

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` | none | Only for Summarise |
| `YTRAG_LLM_BACKEND` | auto | `gemini`, `groq` or `none` |
| `YTRAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Must match the index; `reindex` after changing |
| `YTRAG_CHUNK_SECONDS` / `YTRAG_CHUNK_OVERLAP` | `75` / `15` | Chunk window and overlap |
| `YTRAG_TOP_K` | `6` | Results returned |
| `YTRAG_MAX_DISTANCE` | `0.6` | Cosine-distance pre-filter |
| `YTRAG_TITLE_BOOST` | `0.06` | Per matched title word; `0` disables |
| `YTRAG_RATE_LIMIT_REQUESTS` / `_WINDOW` | `20` / `60` | `/ask` limit per IP |
| `QDRANT_URL` / `QDRANT_API_KEY` | empty | Empty means local embedded store |

---

## Running it on your own playlist (needs a GPU)

```bash
uv sync --extra cuda
uv run ytrag preflight --playlist "<PLAYLIST_URL>"
uv run ytrag langtest "<ONE_VIDEO_URL>"                      # pick the Whisper language
uv run ytrag ingest --playlist "<PLAYLIST_URL>" --limit 1    # test on one video
uv run ytrag ingest --playlist "<PLAYLIST_URL>"              # full playlist
```

Per video: download, transcribe (cached), delete audio, chunk, upsert. A failing video is logged and the run continues, and the same command is safe to re-run after a crash or network drop. Downloads and upserts retry with exponential backoff, and a rate limit triggers a long wait instead of failure.

## Evaluation

```bash
uv run ytrag eval --verbose
```

`eval/golden.json` supports retrieval entries (expected video plus a time range) and refusal entries (off-topic questions that must be declined).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `already open in another process` | Stop `ytrag serve` first |
| `These vectors were built with ...` | Unset `YTRAG_EMBED_MODEL` or run `reindex` |
| `Port 8000 is already in use` | `uv run ytrag serve --port 8001` |
| `ImportError` inside `.venv` (Windows) | Delete `.venv`, add an antivirus exclusion, run `uv sync` |
| LLM quota error | Free tiers are limited; `/search` needs no LLM |

## Limitations and next steps

- Add hybrid search (BM25 + dense) and a cross-encoder re-ranker to improve rare technical terms.
- Expand the golden set with real video IDs for a meaningful hit-rate.
- Move the rate limiter to Redis for multi-worker deployment.
- Quality depends on Whisper's accuracy on accented Hinglish technical terms.

## Acknowledgements

- Lecture content: *Padho with Pratyush* (YouTube). All rights belong to the creator; transcripts are included for search and educational purposes.
- Built with `yt-dlp`, `faster-whisper`, `sentence-transformers`, Qdrant and FastAPI.

