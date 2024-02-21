import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag.docs_rag import utils

docs = [
    Document(page_content="my second text content", metadata={"source": "my/path/fileName2.pdf"}),
    Document(page_content="my text content", metadata={"source": "my/path/fileName1.pdf"}),
    Document(page_content="my third text content", metadata={"source": "my/path/fileName2.pdf"})
]


def test_rsplit():
    assert utils.get_file_name("my/path/toFile.pdf") == "toFile.pdf"


def test_re_split():
    assert re.split("[#/]", "my/Path")[-1] == "Path"
    assert re.split("[#/]", "my/path#Path")[-1] == "Path"


def test_group_by():
    grouped = utils.docs_group_by(docs, "source")
    first_doc = grouped.get("my/path/fileName1.pdf")[0]
    assert first_doc.page_content == "my text content"


def test_format_chunks():
    expected_format = """* CHUNKS from fileName1.pdf *:
\t - chunk - my text content

* CHUNKS from fileName2.pdf *:
\t - chunk - my second text content
\t - chunk - my third text content

"""
    formatted = utils.format_chunks_by_file(docs, "source")
    assert formatted == expected_format


def test_history():
    h = ChatMessageHistory()
    h.add_messages([AIMessage("hello!"), HumanMessage("Hello there!")])
    assert len(h.messages) == 2
    h.clear()
    assert len(h.messages) == 0
