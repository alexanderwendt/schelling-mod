import logging

import numpy as np

log = logging.getLogger(__name__)


def generate_values_from_distribution(mean_value, std_dev) -> np.ndarray:
    return np.random.normal(loc=mean_value, scale=std_dev, size=(2,))


class Agent:
    def __init__(self, agent_id: int, team_id: int, mental_values_mean: np.ndarray, mental_values_std_dev: float):
        log.info("Init agent with agentId: {}, teamId: {}, mental_values: {}, std: {}"
                 .format(agent_id, team_id, mental_values_mean, mental_values_std_dev))
        self.agent_id = agent_id
        self.team_id = team_id
        self.mental_values = generate_values_from_distribution(mental_values_mean, mental_values_std_dev)

    def __repr__(self):
        return f"Agent(team={self.team_id}, id={self.agent_id}, val={[round(num, 1) for num in self.mental_values]})"

