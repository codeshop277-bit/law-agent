import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CLAUDE_CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Pinecone settings
PINECONE_INDEX_NAME = "study-materials"
PINECONE_NAMESPACE = "current-session"
PINECONE_DIMENSION = 1536         # text-embedding-3-small output dimension
PINECONE_METRIC = "cosine"

# Chunking settings
CHUNK_SIZE = 512                  # tokens per chunk
CHUNK_OVERLAP = 100               # overlapping tokens between chunks
MIN_CHUNK_LENGTH = 50             # discard chunks shorter than this (characters)

# Embedding settings
EMBEDDING_BATCH_SIZE = 150        # vectors per batch upsert to Pinecone
CLAUDE_HAIKU_MODEL  = "claude-haiku-4-5-20251001"   # query refine, rerank
CLAUDE_SONNET_MODEL = "claude-sonnet-4-6"    
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
PINECONE_DIMENSION   = 384   

# ── Phase 2 settings ──────────────────────────────────────────────────────

# LLM settings
LLM_MODEL = "gpt-4o"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.3             # low = focused, factual responses

# Retrieval settings
RETRIEVAL_TOP_K = 20              # candidates fetched from Pinecone before reranking
RERANK_TOP_N = 5                  # chunks passed to LLM after reranking
RELEVANCE_THRESHOLD = 0.75        # cosine score below this triggers web fallback

# Conversation memory
MAX_HISTORY_TURNS = 6             # number of user/assistant turn pairs to keep

# Web fallback
WEB_FALLBACK_MAX_RESULTS = 3      # number of web search results to include as context
