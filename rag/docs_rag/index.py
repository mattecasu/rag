import os

from chroma_indexer import ChromaIndexer

RAGS_DOCS_FOLDER = os.path.expanduser('~') + "/Desktop/rag/"
CHROMA_DIR = os.path.expanduser('~') + "/Desktop/chroma"
CHROMA_COLLECTION_NAME = "rag_demo"

indexer = ChromaIndexer(CHROMA_DIR, CHROMA_COLLECTION_NAME)

indexer.index(docs_folder=RAGS_DOCS_FOLDER)
