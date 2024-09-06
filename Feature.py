from enum import Enum

from Agent import Agent


class FeatureType(Enum):
    HOUSE = "h",
    BARRIER = "b"


class Feature:
    '''
    House, barrier

    '''

    def __init__(self, feature_type: FeatureType, agent: Agent = None):
        self.type = feature_type
        self.agent = agent

    def is_empty(self) -> bool:
        return self.agent is None

    def __repr__(self):
        if self.agent is not None:
            value = f"({self.type.name}, aid={self.agent.agent_id})"
        else:
            value = f"({self.type.name}, aid=None)"
        return value
