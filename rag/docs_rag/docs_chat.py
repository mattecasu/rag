import streamlit as st

from rag.docs_rag import history, utils
from rag.docs_rag.docs_rag import chain_with_history, memory_history

# st.set_page_config(layout="wide")
st.title("🥳  Fun with :blue[_RAG_]s!")
st.chat_message("assistant").write("Ask me something!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

utils.get_ui(
    lambda question: chain_with_history.stream(
        input={
            "input": history.add_condensed_history(question, memory_history),
            "history": memory_history
        },
        config={
            "configurable": {"session_id": "foobar"},
            "callbacks": [
                utils.PromptsPrinterHandler()
            ]
        }
    )
)
