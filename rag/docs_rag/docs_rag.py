import logging
import os
from operator import itemgetter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.docs_rag import utils, history
from rag.docs_rag.chroma_indexer import ChromaIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.expanduser('~') + "/Desktop/chroma"
CHROMA_COLLECTION_NAME = "rag_demo"

indexer = ChromaIndexer(CHROMA_DIR, CHROMA_COLLECTION_NAME)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the Question at the end, briefly, based only on the following context.

You are given Chunks of text collected from different files.
The Chat History is also provided (it can be useful to resolve anaphoras).
Never answer with queries or code. Never make up answers

** Chunks **

{chunks}

** Question **

{question}

**** Chat History ****

{history}

""")

chat_llm = ChatOllama(
    model="mistral",
    temperature=1,
    cache=False
)

chunks_retriever = indexer.chroma_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 20}
)

rag_chain = (
        {
            "chunks": itemgetter("input")
                      | utils.get_compressor_wrapper(chunks_retriever, 8)
                      | utils.format_chunks_by_file,
            "question": itemgetter("input")
                        | RunnablePassthrough(),
            "history": itemgetter("history")
                       | RunnablePassthrough()
                       | utils.format_history
        }
        | rag_prompt
        | chat_llm
        | StrOutputParser()
)

memory_history = ChatMessageHistory()
chain_with_history = history.get_history_wrapper(rag_chain, memory_history)
