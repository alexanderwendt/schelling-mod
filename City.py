import numpy as np

from Agent import Agent
from Feature import Feature, FeatureType


class City:
    def __init__(self):
        self.raw_map = None
        self.city = None

    def set_map(self, city_map: np.ndarray):
        self.raw_map = city_map

    def generate_map(self, size: int, empty_ratio: float, races: list[int], teams_distribution: dict):
        p = [empty_ratio]
        if teams_distribution is not None:
            p.extend(np.array(list(teams_distribution.values()))*(1-empty_ratio))
        else:
            p.extend(([((1 - empty_ratio) / (len(races) - 1)) for _ in range(len(races) - 1)]))

        city_size = int(np.sqrt(size)) ** 2

        raw_map_preparation = np.random.default_rng(0).choice(races, size=city_size, p=p)
        self.raw_map = np.reshape(raw_map_preparation, (int(np.sqrt(city_size)), int(np.sqrt(city_size))))

    def instantiate_city(self, agent_values: dict, agent_values_std: float, similarity_threshold: float):
        # For each field of the map, generate feature that is a house for 'H'
        # If house, put an agent if not team 0

        self.city = np.empty(self.raw_map.shape, dtype=Agent)
        current_agent_id = 0

        for (i, j), value in np.ndenumerate(self.raw_map):
            team_id = self.raw_map[i][j].item()

            if team_id != 0:
                agent = Agent(current_agent_id, team_id, agent_values.get(team_id), agent_values_std,
                              similarity_threshold)
            else:
                agent = None

            feature = Feature(FeatureType.HOUSE, [i, j], agent)
            self.city[i][j] = feature

            current_agent_id = current_agent_id + 1

    def get_team_map(self):
        current_team_map = np.zeros(self.raw_map.shape, dtype=int)
        for (i, j), value in np.ndenumerate(self.raw_map):
            if self.city[i][j].agent is not None:
                current_team_map[i][j] = self.city[i][j].agent.team_id
            else:
                current_team_map[i][j] = 0

        return current_team_map

    def get_neighbors(self, row, col, n_neighbors):
        '''

        :return:
        '''
        neighborhood = []
        for i in range(row - n_neighbors, row + n_neighbors + 1):
            for j in range(col - n_neighbors, col + n_neighbors + 1):
                if 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1] and [i, j] != [row, col]:
                    feature = self.city[i, j]
                    if feature.agent is not None and feature.type == FeatureType.HOUSE is not None:
                        neighborhood.append(feature)

        return neighborhood

    def get_random_empty_house_position(self):
        empty_houses = list()
        for (i, j), value in np.ndenumerate(self.city):
            feature = self.city[i, j]
            if feature.type == FeatureType.HOUSE and feature.agent is None:
                empty_houses.append([i, j])

        random_house = empty_houses[np.random.choice(len(empty_houses))]
        return random_house

    def get_mean_similarity_ratio(self, n_neighbors):
        similarities = list()
        for (i, j), feature in np.ndenumerate(self.city):
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                neighborhood = self.get_neighbors(i, j, n_neighbors)
                similarities.append(feature.agent.get_similarity_ratio(neighborhood))

        return np.average(similarities) #0: all are the same, 1: all are completly different

    def __repr__(self):
        return "City"
