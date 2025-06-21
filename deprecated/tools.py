from _operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from agent import game_names, _load_game_manual, qa_llm


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
