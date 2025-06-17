import os
import numpy as np
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage

from deprecated.board_game_agent import BoardGameAgentState

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
client = genai.Client(api_key=GEMINI_API_KEY)


def get_gemini_embeddings(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY'))
    return np.array(result.embeddings[0].values)


def _load_spirit_island_manual():
    with open("../text/spirit_island.txt", "r", encoding="utf-8") as f:
        return f.read()


SPIRIT_ISLAND_TEXT = _load_spirit_island_manual()


def cosine_similarity(vector1, vector2) -> float | None:
    """
  Computes the cosine similarity between two vectors using NumPy.
  """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:
        return None  # Undefined
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def create_spirit_island_state(user_message_content: str,
                               previous_messages: list = None) -> BoardGameAgentState:
    messages = []
    if previous_messages:
        messages.extend(previous_messages)
    messages.append(HumanMessage(content=user_message_content))

    return BoardGameAgentState(
        messages=messages,
        current_game_name="spirit_island",
        current_game_manual=SPIRIT_ISLAND_TEXT,
        identified_game_in_query=None,
        info_message_for_user=None
    )
