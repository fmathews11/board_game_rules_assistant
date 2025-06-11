from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_tavily import TavilySearch
from utils.prompts import SPIRIT_ISLAND_SEARCH_PROMPT, STANDALONE_QUESTION_PROMPT
import os
import dotenv

dotenv.load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_object = TavilySearch(max_results=3,
                                    include_raw_content=True,
                                    include_domains=['https://spiritislandwiki.com/'])


def _extract_raw_text_from_message_history(message_list: list) -> str:
    """
    Extracts raw text from a history of message objects.

    :param message_list: A list of message objects, consisting of instances of
        HumanMessage and AIMessage, whose text content will be extracted and
        formatted into a single raw text block.
    :return: A string containing the formatted raw text of the message history,
        where each message is prefixed by its sender type ('Human:' or 'AI:')
    """
    return "\n".join([f"Human: {message.content}"
                      if isinstance(message, HumanMessage)
                      else f"AI: {message.content}" for message
                      in message_list if isinstance(message, (HumanMessage, AIMessage))])


def get_tavily_search_spirit_island_text(question: str) -> str:
    """
    Retrieves and processes the search results for the specified question using
    the Tavily search tool, returning a concatenated string of raw content.

    :param question: The search query to be processed.
    """
    response = tavily_search_object.invoke(question)
    return " ".join([i['raw_content'] + f"Found in {i['url']}"
                     for i in response['results']
                     if i.get('raw_content')])


def create_spirit_island_search_chain(llm):
    """
    A chain is constructed with components including:
    - A context processing lambda function to search Spirit Island-related text.
    - A passthrough for handling question queries.
    - A specific search prompt for querying Spirit Island content.
    - The provided LLM for response generation

    """

    chain = (
            {'context': RunnableLambda(get_tavily_search_spirit_island_text),
             'question': RunnablePassthrough()}
            | SPIRIT_ISLAND_SEARCH_PROMPT
            | llm
            | StrOutputParser())

    return chain


def create_standalone_question_from_chat_history(chat_history: list,
                                                 users_question: str,
                                                 llm) -> str:

    formatted_chat_history = _extract_raw_text_from_message_history(chat_history)
    chain = (
            {"chat_history": itemgetter("chat_history"), "user_question": itemgetter("user_question")}
            | STANDALONE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
    )
    return chain.invoke({'chat_history': formatted_chat_history, 'user_question': users_question})


def rephrase_query_for_better_search(original_query: str, previous_answer: str, llm) -> str:
    """
    Rephrases a query to get better search results when the initial search was insufficient.

    :param original_query: The original query that was used for search
    :param previous_answer: The answer from the previous search attempt
    :param llm: The language model to use for rephrasing
    :return: A rephrased query for better search results
    """
    prompt = f"""
    You are a search query optimization expert. Your task is to rephrase a query to get better search results.

    The original query was: "{original_query}"

    The search results for this query were insufficient to fully answer the question. 
    Here's the previous answer: "{previous_answer}"

    Please rephrase the query to be more specific, use different keywords, or approach the question from a different angle.
    Focus on extracting the core information need and expressing it in a way that would yield better search results.

    Rephrased query:
    """

    response = llm.invoke(prompt)
    return response.content.strip()
