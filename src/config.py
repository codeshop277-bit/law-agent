import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

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
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 150        # vectors per batch upsert to Pinecone
