import logging

import numpy as np

import utils

log = logging.getLogger(__name__)


def generate_values_from_distribution(mean_value, std_dev) -> np.ndarray:
    return np.random.normal(loc=mean_value, scale=std_dev, size=(2,))


class Agent:
    def __init__(self, agent_id: int, team_id: int, mental_values_mean: np.ndarray, mental_values_std_dev: float,
                 similarity_threshold: float):
        log.info("Init agent with agentId: {}, teamId: {}, mental_values: {}, std: {}"
                 .format(agent_id, team_id, mental_values_mean, mental_values_std_dev))
        self.agent_id = agent_id
        self.team_id = team_id
        self.mental_values = generate_values_from_distribution(mental_values_mean, mental_values_std_dev)
        self.similarity_threshold = similarity_threshold

    def get_similarity_ratio(self, neighborhood: list):
        if len(neighborhood) > 0:
            similarity_ratio = np.average(
                [utils.calulate_similarity_to_nehighbor(self.mental_values, neighbor.agent.mental_values) for neighbor in
                 neighborhood]
            )
        else:
            similarity_ratio = 1

        return similarity_ratio

    def is_unhappy(self, neighborhood: list) -> bool:
        '''

        :return:
        '''
        is_unhappy = (self.get_similarity_ratio(neighborhood) > self.similarity_threshold)

        return is_unhappy

    def __repr__(self):
        return f"Agent(team={self.team_id}, id={self.agent_id}, val={[str(round(num, 1)) for num in self.mental_values]})"
