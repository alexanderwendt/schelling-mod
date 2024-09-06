import numpy as np

from Agent import Agent
from Feature import Feature, FeatureType


class City:
    def __init__(self):
        self.raw_map = None
        self.city = None

    def set_map(self, city_map: np.ndarray):
        self.raw_map = city_map

    def generate_map(self, size: int, empty_ratio: float, races: list[int]):
        p = [(1 - empty_ratio) / 2, (1 - empty_ratio) / 2, empty_ratio]
        city_size = int(np.sqrt(size)) ** 2

        raw_map_preparation = np.random.default_rng(0).choice(races, size=city_size, p=p)
        self.raw_map = np.reshape(raw_map_preparation, (int(np.sqrt(city_size)), int(np.sqrt(city_size))))

    def instantiate_city(self, agent_values: dict, agent_values_std: float):
        # For each field of the map, generate feature that is a house for 'H'
        # If house, put an agent if not team 0

        self.city = np.empty(self.raw_map.shape, dtype=Agent)
        current_agent_id = 0

        for (i, j), value in np.ndenumerate(self.raw_map):
            team_id = self.raw_map[i][j].item()

            if team_id != 0:
                agent = Agent(current_agent_id, team_id, agent_values.get(team_id), agent_values_std)
            else:
                agent = None

            feature = Feature(FeatureType.HOUSE, agent)
            self.city[i][j] = feature

            current_agent_id = current_agent_id+1

    def __repr__(self):
        return "City"



