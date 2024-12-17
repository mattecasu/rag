from operator import itemgetter

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from rag.docs_rag import utils


def get_history_wrapper(rag_chain, memory_history: BaseChatMessageHistory):
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=lambda session_id: memory_history,
        input_messages_key="input",
        history_messages_key="history"
    )


condensing_prompt = ChatPromptTemplate.from_template(
    """
    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    
    **** Question ****:
    
    {question}
    
    **** History ****:
    
    {history}
    """
)

# https://python.langchain.com/docs/use_cases/question_answering/chat_history
# (RAG doesn't work well without a history re-summary)
condenser_chain = (
        {
            "question": itemgetter("input")
                        | RunnablePassthrough(),
            "history": itemgetter("history")
                       | RunnablePassthrough()
                       | utils.format_history
        }
        | condensing_prompt
        | ChatOllama(model="mistral", temperature=0, cache=False)
        | StrOutputParser()
)


def add_condensed_history(question: str, memory_history: BaseChatMessageHistory) -> str:
    if not memory_history.messages:
        return question
    else:
        condensed_question = condenser_chain.invoke(
            input={
                "input": question,
                "history": memory_history.messages
            }
        )
        memory_history.clear()
        return condensed_question


if __name__ == "__main__":
    memory_history = ChatMessageHistory()
    memory_history.add_user_message("What is a history?")
    memory_history.add_ai_message("A history is something important")
    answer = condenser_chain.invoke(
        input={
            "input": "what candidates have worked with SQL for more than 3 years?",
            "history": memory_history.messages
        },
        config={
            "configurable": {"session_id": "foobar"},
            "callbacks": [
                utils.PromptsPrinterHandler()
            ]
        }
    )
    print(answer)
