import glob
import logging
from typing import List

import chromadb
from chromadb import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaIndexer:
    # __MODEL = "all-MiniLM-L6-v2"
    __MODEL = "all-mpnet-base-v2"

    def __init__(self, chroma_dir, collection_name):
        self.chroma_dir = chroma_dir
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection_name = collection_name
        self.embedder = SentenceTransformerEmbeddings(
            model_name=self.__MODEL
        )
        self.chroma_store = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedder
        )

    def already_indexed(self, doc_name: str):
        maybe_doc = self.chroma_client.get_collection(self.collection_name).get(where={"source": doc_name})
        return ("documents" in maybe_doc) and (len(maybe_doc.get("documents")) >= 1)

    def embed(self, chunks, embedding_model, collection_name):
        chroma = Chroma.from_documents(
            client=self.chroma_client,
            collection_name=collection_name,
            documents=chunks,
            embedding=embedding_model,
            persist_directory=self.chroma_dir)
        # chroma.persist()

    def get_chunks(self, separators: List[str], file_path, chunk_size=300, chunk_overlap=20) -> List[
        Document]:
        loader = PyPDFLoader(file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
        )
        pages = loader.load_and_split(text_splitter=splitter)
        logger.info(f"We have split the file {file_path} into {len(pages)} chunks")
        return pages

    def index(self, docs_folder):
        for doc in glob.glob(docs_folder + "*.pdf"):
            if self.already_indexed(doc):
                logger.info(f"Skipping {doc}, already indexed")
                continue
            logger.info(f"Indexing {doc}")
            chunks = self.get_chunks(
                separators=["\n\n", "\n", " ", "", "-", "—"],
                file_path=doc,
                chunk_size=500,
                chunk_overlap=10
            )
            logger.info(f"Embedding {len(chunks)} chunks.")
            self.embed(chunks, self.embedder, self.collection_name)
            logger.info(f"Indexed {doc}")
        logger.info(
            f"The collection {self.collection_name} contains {self.chroma_client.get_collection(self.collection_name).count()} docs"
        )
