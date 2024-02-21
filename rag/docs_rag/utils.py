import itertools
import logging
from typing import Dict, List, Any, Iterable

import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ragatouille import RAGPretrainedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptsPrinterHandler(BaseCallbackHandler):
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        print(f"Prompt:\n{formatted_prompts}")


def wrap_in_colbert(base_retriever, k=5) -> BaseRetriever:
    return ContextualCompressionRetriever(
        base_compressor=RAGPretrainedModel
        .from_pretrained("colbert-ir/colbertv2.0")
        .as_langchain_document_compressor(k=k),
        base_retriever=base_retriever
    )


def get_ui(generate_output):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    text = st.chat_input("Enter text:")
    if text:
        text = text.strip()
        st.chat_message("user").write(text)
        with st.spinner("🤔..."):
            answer, answer_to_store = itertools.tee(generate_output(text))
            st.chat_message("assistant").write_stream(answer)
        stored_answer = ''.join(answer_to_store)
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.messages.append({"role": "assistant", "content": stored_answer})


def docs_group_by(docs: Iterable[Document], metadata_field: str) -> dict[str, Iterable[Document]]:
    sorted_list = sorted(docs, key=lambda d: d.metadata[metadata_field])
    groupdict = {k: list(g) for k, g in itertools.groupby(sorted_list, lambda d: d.metadata[metadata_field])}
    return groupdict


def get_file_name(path: str) -> str:
    return path.rsplit('/')[-1]


def format_chunks_by_file(chunks: Iterable[Document], field="source") -> str:
    grouped = docs_group_by(chunks, field)
    formatted_chunks = ""
    for (source_key, chunk_docs) in grouped.items():
        file_name = get_file_name(source_key)
        chunks = "\n".join(["\t - chunk - " + chunk.page_content.replace('\n', '  ') for chunk in chunk_docs])
        formatted_chunks += f"* CHUNKS from {file_name} *:\n{chunks}\n\n"
    return formatted_chunks


def format_history(docs):
    return '\n'.join(f"{doc.type.capitalize()}: {doc.content.strip()}" for doc in docs)
