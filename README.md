
# RAG over PDFs

This repo lets you index unstructured documents in a Chroma store, and then query them through an offline LLM served by Ollama. A simple UI is provided through streamlit.

## Prerequisite
- Install [Ollama](https://ollama.ai) or [run it with Docker](https://hub.docker.com/r/ollama/ollama)
- pull the needed models (e.g. `ollama pull mistral`)

## Run the app
- run `poetry run python3 rag/docs_rag/index.py` to index PDFs into a Chroma index.
- run the chat with `poetry run streamlit run rag/docs_rag/docs_chat.py`
