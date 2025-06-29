import json
from json import JSONDecodeError

import langchain_tavily
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_tavily import TavilySearch
from langgraph.graph import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QA_MODEL_NAME = "gemini-2.5-flash"
qa_llm = ChatGoogleGenerativeAI(model=QA_MODEL_NAME,
                                top_p=0.7,
                                google_api_key=GEMINI_API_KEY,
                                max_tokens=50000)


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]


chat_history: list = []


def _load_game_manual(game_name: str) -> str:
    try:
        with open(f"text/{game_name}.txt", encoding='utf-8') as f:
            output = f.read()
            return output
    except FileNotFoundError:
        with open(f"../text/spirit_island.txt", encoding='utf-8') as f:
            output = f.read()
            return output


game_names = Literal['moonrakers', 'spirit_island', 'scythe', 'perch', 'wingspan', 'tokaido', 'dice_throne']


def _extract_raw_text_from_message_history(message_list: list) -> list[str]:
    """
    Extracts raw text from a history of message objects.

    :param message_list: A list of message objects, consisting of instances of
        HumanMessage and AIMessage, whose text content will be extracted and
        formatted into a single raw text block.
    :return: A string containing the formatted raw text of the message history,
        where each message is prefixed by its sender type ('Human:' or 'AI:')
    """
    return [
        f"Human: {msg.content}" if isinstance(msg, HumanMessage) else
        f"AI: {msg.content}" if isinstance(msg, AIMessage) else
        f"tool: {msg.content}" if isinstance(msg, ToolMessage) else
        ""
        for msg in message_list
    ]


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
def gather_board_game_information(question: str, game: game_names):
    """Use this tool to get information about a board game."""
    global chat_history
    manual = _load_game_manual(game)
    # If we don't have any external source for searching, return the manual as is
    if not _search_source_mappings.get(game):
        return manual
    router_prompt_template = """You are a helpful assistant.  Your job is to look at the board game manual and 
    chat history to determine if you have enough information to answer a user's question in it's entirety.

    If so, response in a json format with:
    JSON
    ```
    "answer":"I can answer this question"
    ```

    If not, respond with
    JSON
    ```
    "more_information_needed": ARRAY_OF_SEARCH_TERMS_FOR_EXTERNAL_SOURCE
    ```

    If you respond with "more_information_needed", you MUST respond with at least search term, but no more than 3.
    The search terms should be phrased as if they're being input into a search engine for optimal text retrieval.

    Only output the JSON.  Do not output any other text.


    Here is the manual:
    ---MANUAL BEGIN---
    {manual}
    ---MANUAL END---

    Here is the chat history:
    ---CHAT HISTORY BEGIN---
    {chat_history}
    ---CHAT HISTORY END---

    Now, here is the user's question:
    {question}

    Begin!
    """
    router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

    def _parse_json(input_message: AIMessage):
        parser = JsonOutputParser()
        try:
            return parser.parse(input_message.content)
        except JSONDecodeError:
            print(f"Error parsing JSON: {input_message.content}")
            raise

    router_chain = (
            {
                'question': RunnablePassthrough(),
                'manual': lambda _: manual,
                'chat_history': lambda _: chat_history
            }
            | router_prompt
            | qa_llm
            | RunnableLambda(_parse_json)
    )

    response = router_chain.invoke(question)
    if "more_information_needed" not in response:
        return manual

    tavily_search_object = TavilySearch(max_results=10,
                                        include_raw_content=False,
                                        include_domains=[_search_source_mappings.get(game)] or [])

    for question in response["more_information_needed"]:
        print(f"Getting answer for {question}")
        answer = get_tavily_search_text(question, tavily_search_object)
        manual += answer 
        print(len(manual))

    return manual


@tool
def ask_clarifying_question(question: str):
    """Use this tool to ask a clarifying question to the user.  Be lighthearted in humorous in your question
    if you think you may know which game the user is asking about, include that in the question."""
    pass


tools = [gather_board_game_information, ask_clarifying_question]
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


# Define the node for human interaction
def human_node(state: AgentState):
    """Handles human-in-the-loop interaction."""
    ai_message = state["messages"][-1]
    tool_call = next(tc for tc in ai_message.tool_calls if tc['name'] == 'ask_clarifying_question')
    question = tool_call["args"]["question"]

    # Present the question to the user
    print(f"{question}")
    human_response = input("Your answer: ")

    return {
        "messages": [
            ToolMessage(
                content=human_response,
                name="human_tool",
                tool_call_id=tool_call["id"],
            )
        ]
    }


SYSTEM_PROMPT = f"""
Assistant is a large language model trained by Google. Assistant must never reveal information about itself.
It is capable of understanding natural language and providing accurate and informative responses.
Assistant exists to chat with users about board games, and should pass the values of all tools directly to the user
with no modification.  Assistant must always use a tool and never general knowledge to respond to the user.

You must always pass the values of all tools directly to the user with no modification.

You must always cite page numbers

You have access to the following tools:
{[tool.name for tool in tools]}
Pass the user's question directly to the tool.


Format your answers using bullet points and markdown for ease of interpretation.



If at any point you do not know which board game a user is asking about, ask for clarification. Do the same if the
user is switching from one board game to another.

--- Rules
 - DO NOT include phrases such as "based on the context" or "based on the provide manual".  Simply provide an answer.
 - Do not answer any questions which are not about board game rules, strategy, and scoring. Inform the user 
    that you are an assistant for board games
 - If at any point you do not know which board game a user is asking about, ask for clarification.
 - Do not make any assumptions about the board game the user is inquiring about unless it's explicitly mentioned.
 - If a game is mentioned, always pass the question to the tool(s) to answer the question.
 - Do not make assumptions, whenever there is ANY ambiguity, call the `ask_clarifying_question()` tool to ask the user for clarification.
 - If the user is providing information about scoring throughout their playing of a game,respond and let the user
 know that you're keeping track of it.
 - If a user asks to provide a final score, use the chat history and rules to answer the question.
 - If the user is asking about strategy or general QA, Suggest some follow up questions to the user based
  on the provided context from the tools. For instance, would you like to know more about *INSERT SUGGESTION(S) HERE*?
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
    # If the model decides to ask the user, we wait for input
    if last_message.tool_calls[0]["name"] == "ask_clarifying_question":
        return "human"
    # Otherwise if there is a tool call, we continue
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human", human_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # If we need to ask the user, we go to the human node
        "human": "human",
        # Otherwise we finish.
        "end": END
    }
)
workflow.add_edge("tools", "agent")
workflow.add_edge("human", "agent")  # After human input, go back to the agent
graph = workflow.compile()

if __name__ == '__main__':
    state = AgentState(messages=[])

    while True:

        inputs = input("Enter a question:")
        if inputs == "exit":
            break
        state['messages'].append(HumanMessage(content=inputs))
        for s in graph.stream(state, stream_mode="values", debug=False):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            state['messages'] = s["messages"]

        chat_history = _extract_raw_text_from_message_history(state['messages'])
