import unittest
from custom_react_agent import graph as react_graph, AgentState as react_state
from langchain_core.messages import HumanMessage, ToolMessage
from answer_benchmarks import RAZE_ANSWER,BLIGHT_LOSS_ANSWER
from tests.test_utility_functions import get_gemini_embeddings, cosine_similarity
import time


class TestReactAgent(unittest.TestCase):

    def test_assert_raze_answer_is_accurate(self):
        complicated_question = """It's the end of the Fast Phase. I just used 'Raze' (a unique power from Bringer of 
        Dream-of-Flower) to destroy a town on a Blighted land with 2 Blight. This generated 1 Fear. I also have 
        'Terrifying Nightmares' (Fear card) in play, which states 'After Fear is generated, 1 Damage to Invaders in a 
        land with no Blight.' I previously placed 2 Blight on the island from an event, bringing the total Blight to 
        5. What is the precise sequence of events for resolving the Fear and Blight, and what are the final states of 
        the land, the Blight pool, and the Terror Level, assuming no other Blight sources or Fear generation?"""
        state = react_state(messages=[HumanMessage(content=complicated_question)])
        answer = react_graph.invoke(state)
        self.assertTrue(any([isinstance(i, ToolMessage) for i in answer['messages']]))
        known_answer_embedding = get_gemini_embeddings(RAZE_ANSWER)
        time.sleep(10)
        candidate_answer_embedding = get_gemini_embeddings(answer['messages'][-1].content)
        similarity = cosine_similarity(known_answer_embedding, candidate_answer_embedding)
        print(f"Similarity for RAZE question: {similarity}")
        self.assertGreater(similarity, 0.9)

    def test_assert_blight_loss_answer_is_accurate(self):
        complicated_question = """The game is in the late stages. We are facing a Blight loss condition. The island 
        has 7 Blight in total, and the Blight card states '3 Blight per player before loss.' We are playing a 
        2-player game. The current turn's Invader card is 'Coastal Lands Ravage.' There are 3 Coastal lands, 
        each with 1 Town. One of these Coastal lands also has 1 Blight. We have enough Energy to play one Major Power 
        between us. We have 'Thematic Corruption' (Major Power: 3 Energy, 3 Sun, 2 Fire, Range 0, Target 1 Blighted 
        land. Deal 4 Damage to Invaders. If this destroys any Invaders, remove 1 Blight) and 'Quake' (Major Power: 4 
        Energy, 2 Earth, 1 Air, Range 1, Target 1 land. Deal 3 Damage to Invaders. Replace 1 Town with 1 Explorer). 
        Which Major Power should we prioritize playing, and why, if our primary goal is to avoid the Blight loss 
        condition this turn?"""
        state = react_state(messages=[HumanMessage(content=complicated_question)])
        answer = react_graph.invoke(state)
        self.assertTrue(any([isinstance(i, ToolMessage) for i in answer['messages']]))
        known_answer_embedding = get_gemini_embeddings(BLIGHT_LOSS_ANSWER)
        time.sleep(10)
        candidate_answer_embedding = get_gemini_embeddings(answer['messages'][-1].content)
        similarity = cosine_similarity(known_answer_embedding, candidate_answer_embedding)
        print(f"Similarity for blight loss question: {similarity}")
        self.assertGreater(similarity, 0.9)


if __name__ == "__main__":
    unittest.main()
