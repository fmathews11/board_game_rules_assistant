import json
from operator import itemgetter
import langchain_tavily
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_tavily import TavilySearch
from langgraph.graph import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
qa_llm = ChatGoogleGenerativeAI(model=QA_MODEL_NAME,
                                temperature=0,
                                google_api_key=GEMINI_API_KEY,
                                max_tokens=50000)


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]


def _load_game_manual(game_name: str) -> str:
    try:
        with open(f"text/{game_name}.txt", encoding='utf-8') as f:
            output = f.read()
            return output
    except FileNotFoundError:
        with open(f"../text/spirit_island.txt", encoding='utf-8') as f:
            output = f.read()
            return output


game_names = Literal['moonrakers', 'spirit_island', 'scythe', 'perch', 'wingspan', 'tokaido']


@tool
def search_board_game_text(question: str,
                           game_name: game_names):
    """
    Search and provide answers to questions related to board game manual.
    The inputs to this function should always be phrased as a question.  If the inbound query is a question,
    do not alter it.

    :param question: The question to be answered.
    :param game_name: The name of the game for which the manual is to be searched.
    """
    manual = _load_game_manual(game_name)
    prompt_template = """
    You are a helpful assistant. You are proficient in answering questions about board game rules.

    Use the following rulebook to answer a user's question

    ---RULES START---
    {manual}
    ---RULES END---

    ## Answer guidelines
     - Answer the question directly, as you are a subject matter expert.
     - DO NOT include phrases such as "based on the context" or "based on the provide manual".  Simply provide an answer.
     - Provide page number references to cite where the user can find this information.
     - If URL's are present, cite the specific URLs where you found the information.
     - Use as many words as necessary to answer the question
     - Use bullet points and/or markdown format to make the answer as easily-interpreted as possible.

    Now, use ONLY this information to answer a user's question. Do not use any other information.
    Here is the user's question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
            {"user_question": itemgetter('user_question'), "manual": itemgetter('manual')}
            | prompt
            | qa_llm
            | StrOutputParser()
    )
    return chain.invoke({'user_question': question, 'manual': manual})


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


def get_tavily_search_text(question: str,
                           tavily_search_object: langchain_tavily.TavilySearch) -> str:
    """
    Retrieves and processes the search results for the specified question using
    the Tavily search tool, returning a concatenated string of raw content.  This should only be used
    when search_spirit_island_basic is not sufficient.

    :param tavily_search_object:
    :param question: The search query to be processed.
    """
    response = tavily_search_object.invoke(question)
    if isinstance(response, str):
        return ""
    return " ".join([i['content'] + f" Found in {i['url']}"
                     for i in response['results']
                     if i.get('content')])


_search_source_mappings = {"spirit_island": 'https://spiritislandwiki.com/',
                           "scythe": 'https://www.reddit.com/r/SCYTHE/'
                           }


@tool
def augment_search_context(question: str, game: game_names):
    """Use this tool to get information about a board game."""
    manual = _load_game_manual(game)
    # If we don't have any external source for searching, return the manual as is
    if not _search_source_mappings.get(game):
        return manual
    router_prompt_template = """You are a helpful assistant.  Your job is to look at the board game manual and 
    chat history to determine if you have enough information to answer a user's question in it's entirety.

    If so, response in a json format with:
    ```
    "answer":"I can answer this question"
    ```

    If not, respond with
    ```
    "more_information_needed": ARRAY_OF_SEARCH_TERMS_FOR_EXTERNAL_SOURCE
    ```

    If you respond with "more_information_needed", you MUST respond with at least search term, but no more than 3.
    The search terms should be phrased as if they're being input into a search engine for optimal text retrieval.
    
    Only output the JSON.  Do not output any other text.
    

    Here is the manual:
    {manual}
    
    Here is the chat history:
    

    Now, here is the user's question:
    {question}

    Begin!
    """
    router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

    router_chain = (
            {
                'question': RunnablePassthrough(),
                'manual': lambda x: manual
            }
            | router_prompt
            | qa_llm
            | JsonOutputParser()
    )
    response = router_chain.invoke(question)
    if "more_information_needed" in response:
        tavily_search_object = TavilySearch(max_results=10,
                                            include_raw_content=False,
                                            include_domains=[_search_source_mappings.get(game)])

        for question in response["more_information_needed"]:
            print(f"Getting answer for {question}")
            answer = get_tavily_search_text(question, tavily_search_object)
            manual += answer
            print(len(manual))

    return manual


tools = [augment_search_context]
qa_llm = qa_llm.bind_tools(tools=tools)

tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


SYSTEM_PROMPT = f"""
Assistant is a large language model trained by Google. 
It is capable of understanding natural language and providing accurate and informative responses.
Assistant is able to answer questions about board games, and should pass the values of all tools directly to the user
with no modification.  Assistant must always use a tool and never general knowledge.

You must always pass the values of all tools directly to the user with no modification.

You must always cite page numbers

You have access to the following tools:
{[tool.name for tool in tools]}
Pass the user's question directly to the tool.


Format your answers using bullet points and markdown for ease of interpretation.

Suggest some follow up questions to the user based on the provided context from the tools.
For instance, would you like to know more about *INSERT SUGGESTION(S) HERE*?

--- Rules
 - DO NOT include phrases such as "based on the context" or "based on the provide manual".  Simply provide an answer.
 - Do not answer any questions which are not about board game rules or strategy. Inform the user that you are an assistant for board games.
 - If at any point you do not know which board game a user is asking about, ask for clarification.
 - Do not make any assumptions about the board game the user is inquiring about unless it's explicitly mentioned.
 - If a game is mentioned, always pass the question to the tool(s) to answer the question.
"""


# Define the node that calls the model
def call_model(
        state: AgentState,
        config: RunnableConfig,
):
    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage(SYSTEM_PROMPT)
    response = qa_llm.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END}
)
workflow.add_edge("tools", "agent")
graph = workflow.compile()

if __name__ == '__main__':
    state = AgentState(messages=[])

    while True:

        inputs = input("Enter a question:")
        if inputs == "exit":
            break
        state['messages'].append(HumanMessage(content=inputs))
        for s in graph.stream(state, stream_mode="values", debug=True):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            state['messages'] = s["messages"]
