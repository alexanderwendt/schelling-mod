'''
MIT License

Copyright (c) 2024 Alexander Wendt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

run with streamlit run main.py
'''

import argparse
import logging
import math
import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2024'
__credits__ = ['Credits to Adil Moujahid and his version from https://github.com/adilmoujahid/streamlit-schelling']
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@gmx.at'
__status__ = 'Experimental'

from Agent import Agent
from City import City
from Feature import FeatureType

parser = argparse.ArgumentParser(description='Schelling Model Modified')
parser.add_argument("-r", "--run_simulation", action='store_true', help="Run simulation with default parameters",
                    required=False)
parser.add_argument("--population_size", help="Population size", type=int, default=2500, required=False)
parser.add_argument("--empty_ratio", help="Empty houses ratio", type=float, default=0.2, required=False)
parser.add_argument("--similarity_threshold", help="Similarity threshold", type=float, default=0.4, required=False)
parser.add_argument("--iterations", help="Number of iterations", type=int, default=10, required=False)

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info(args)

#city_map = np.array([[1, 0, 0, 2, 1], [1, 2, 0, 2, 1], [1, 2, 0, 2, 1], [1, 2, 0, 2, 1]])
city_map = np.array([[1, 1, 1, 0, 0],
                     [1, 2, 1, 2, 1],
                     [1, 1, 1, 1, 2],
                     [0, 0, 0, 0, 0],
                     [2, 1, 2, 1, 2]])

n_neighbors = 2
mental_values_std_dev = 0.0
mental_values_map = {1: [0, 0], 2: [0.707, 0.707], 3: [0.2, 0]}


class Schelling:

    def __init__(self, size, empty_ratio, similarity_threshold, n_neighbors, load_map: bool):
        self.races = [1, 2, 0]  # races 1 and 2 and 0 is empty places
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold

        self.city = City()

        if load_map:
            self.city.set_map(city_map)
        else:
            self.city.generate_map(size, empty_ratio, self.races)

        self.city.instantiate_city(mental_values_map, mental_values_std_dev, self.similarity_threshold)

        log.info("City: \n{}".format(self.city.city))
        log.info("Team map: \n{}".format(self.city.get_team_map()))

    def run(self):
        for (row, col), value in np.ndenumerate(self.city.city):
            feature = self.city.city[row, col]
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                current_agent: Agent = feature.agent
                log.debug("Processing feature: {}".format(feature))
                # Get all plots in the neighborhood
                neighborhood = self.city.get_neighbors(row, col, self.n_neighbors)
                log.debug("Neighborhood: {}".format(neighborhood))
                is_unhappy = current_agent.is_unhappy(neighborhood)

                if is_unhappy:
                    random_house_position = self.city.get_random_empty_house_position()
                    self.city.city[random_house_position[0], random_house_position[1]].agent = feature.agent
                    self.city.city[row, col].agent = None
                    log.debug("Agent {} is unhappy. Move from {} to {}".format(current_agent, [row, col],
                                                                               random_house_position))
                else:
                    log.debug("Agent {} is happy. Do nothing".format(current_agent))
            else:
                log.debug("Feature {} is not a house or house is empty".format(feature))


# Streamlit App
if __name__ == "__main__":
    st.title("Schelling's Model of Segregation")

    if args.run_simulation:
        schelling = Schelling(args.population_size, args.empty_ratio, args.similarity_threshold, 1, True)

        for i in range(args.iterations):
            schelling.run()
            mean_similarity_ratio = schelling.city.get_mean_similarity_ratio(n_neighbors)
    else:
        population_size = st.sidebar.slider("Population Size", 9, 10000, 1000)
        empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0., 1., .2)
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .3)
        n_iterations = st.sidebar.number_input("Number of Iterations", 10)

        schelling = Schelling(population_size, empty_ratio, similarity_threshold, n_neighbors, False)

        mean_similarity_ratio = [schelling.city.get_mean_similarity_ratio(n_neighbors)]

        # Plot the graphs at initial stage
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 4))

        # Left hand side graph with Schelling simulation plot
        cmap = ListedColormap(['white', 'red', 'royalblue'])
        plt.subplot(121)
        plt.axis('off')
        plt.pcolor(schelling.city.get_team_map(), cmap=cmap, edgecolors='w', linewidths=1)

        # Right hand side graph with Mean Similarity Ratio graph
        plt.subplot(122)
        plt.xlabel("Iterations")
        plt.xlim([0, n_iterations+1])
        plt.ylim([0.0, 1])
        plt.title("Mean Similarity Ratio", fontsize=15)
        plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.city.get_mean_similarity_ratio(n_neighbors), fontsize=10)

        city_plot = st.pyplot(plt)

        progress_bar = st.progress(0)

        if st.sidebar.button('Run Simulation'):
            for i in range(n_iterations+1):
                schelling.run()
                mean_similarity_ratio.append(schelling.city.get_mean_similarity_ratio(n_neighbors))
                plt.figure(figsize=(8, 4))

                plt.subplot(121)
                plt.axis('off')
                plt.pcolor(schelling.city.get_team_map(), cmap=cmap, edgecolors='w', linewidths=1)

                plt.subplot(122)
                plt.xlabel("Iterations")
                plt.xlim([0, n_iterations+1])
                plt.ylim([0.0, 1])
                plt.title("Mean Similarity Ratio", fontsize=15)
                plt.plot(range(1, len(mean_similarity_ratio) + 1), mean_similarity_ratio)
                plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.city.get_mean_similarity_ratio(n_neighbors),
                         fontsize=10)

                city_plot.pyplot(plt)
                plt.close("all")
                progress_bar.progress((i + 1.) / (n_iterations+1))
