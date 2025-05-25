from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_tavily import TavilySearch
from utils.prompts import SPIRIT_ISLAND_SEARCH_PROMPT
import os
import dotenv

dotenv.load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_object = TavilySearch(max_results=10,
                                    include_raw_content=True,
                                    include_domains=['https://spiritislandwiki.com/'])


def _get_tavily_search_spirit_island_text(question: str) -> str:
    """
    Retrieves and processes the search results for the specified question using
    the Tavily search tool, returning a concatenated string of raw content.

    :param question: The search query to be processed.
    """
    response = tavily_search_object.invoke(question)
    return " ".join([i['raw_content'] or "" for i in response['results']])


def create_spirit_island_search_chain(llm):
    """
    A chain is constructed with components including:
    - A context processing lambda function to search Spirit Island-related text.
    - A passthrough for handling question queries.
    - A specific search prompt for querying Spirit Island content.
    - The provided LLM for response generation

    :param llm: LLM model istance
    """

    chain = (
            {'context': RunnableLambda(_get_tavily_search_spirit_island_text),
             'question': RunnablePassthrough()}
            | SPIRIT_ISLAND_SEARCH_PROMPT
            | llm
            | StrOutputParser())

    return chain
