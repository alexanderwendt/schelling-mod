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
'''

import argparse
import logging
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

parser = argparse.ArgumentParser(description='Schelling Model Modified')
parser.add_argument("-r", "--run_simulation", action='store_true', help="Run simulation with default parameters", required=False)
parser.add_argument("--population_size", help="Population size", type=int, default=2500, required=False)
parser.add_argument("--empty_ratio", help="Empty houses ratio", type=float, default=0.2, required=False)
parser.add_argument("--similarity_threshold", help="Similarity threshold", type=float, default=0.4, required=False)
parser.add_argument("--iterations", help="Number of iterations", type=int, default=10, required=False)

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info(args)


class Schelling:

    def __init__(self, size, empty_ratio, similarity_threshold, n_neighbors):
        self.size = size
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors

        # Ratio of races (-1, 1) and empty houses (0)
        p = [(1 - empty_ratio) / 2, (1 - empty_ratio) / 2, empty_ratio]
        city_size = int(np.sqrt(self.size)) ** 2
        self.city = np.random.choice([-1, 1, 0], size=city_size, p=p)
        self.city = np.reshape(self.city, (int(np.sqrt(city_size)), int(np.sqrt(city_size))))

    def run(self):
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            if race != 0:
                neighborhood = self.city[
                               row - self.n_neighbors:row + self.n_neighbors,
                               col - self.n_neighbors:col + self.n_neighbors]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.nonzero(neighborhood == 0)[0])
                if neighborhood_size != n_empty_houses + 1:
                    n_similar = len(np.nonzero(neighborhood == race)[0]) - 1
                    similarity_ratio = n_similar / (neighborhood_size - n_empty_houses - 1.)
                    is_unhappy = (similarity_ratio < self.similarity_threshold)
                    if is_unhappy:
                        empty_houses = list(zip(np.nonzero(self.city == 0)[0], np.nonzero(self.city == 0)[1]))
                        random_house = random.choice(empty_houses)
                        self.city[random_house] = race
                        self.city[row, col] = 0

    def get_mean_similarity_ratio(self):
        count = 0
        similarity_ratio = 0
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            if race != 0:
                neighborhood = self.city[row - self.n_neighbors:row + self.n_neighbors,
                               col - self.n_neighbors:col + self.n_neighbors]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.nonzero(neighborhood == 0)[0])
                if neighborhood_size != n_empty_houses + 1:
                    n_similar = len(np.nonzero(neighborhood == race)[0]) - 1
                    similarity_ratio += n_similar / (neighborhood_size - n_empty_houses - 1.)
                    count += 1
        return similarity_ratio / count


# Streamlit App
if __name__ == "__main__":
    st.title("Schelling's Model of Segregation")

    if args.run_simulation:
        schelling = Schelling(args.population_size, args.empty_ratio, args.similarity_threshold, 3)
        for i in range(args.iterations):
            schelling.run()
    else:
        population_size = st.sidebar.slider("Population Size", 9, 10000, 2500)
        empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0., 1., .2)
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .4)
        n_iterations = st.sidebar.number_input("Number of Iterations", 10)

        schelling = Schelling(population_size, empty_ratio, similarity_threshold, 3)

        mean_similarity_ratio = [schelling.get_mean_similarity_ratio()]

        # Plot the graphs at initial stage
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 4))

        # Left hand side graph with Schelling simulation plot
        cmap = ListedColormap(['red', 'white', 'royalblue'])
        plt.subplot(121)
        plt.axis('off')
        plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

        # Right hand side graph with Mean Similarity Ratio graph
        plt.subplot(122)
        plt.xlabel("Iterations")
        plt.xlim([0, n_iterations])
        plt.ylim([0.4, 1])
        plt.title("Mean Similarity Ratio", fontsize=15)
        plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

        city_plot = st.pyplot(plt)

        progress_bar = st.progress(0)

        if st.sidebar.button('Run Simulation'):
            for i in range(n_iterations):
                schelling.run()
                mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())
                plt.figure(figsize=(8, 4))

                plt.subplot(121)
                plt.axis('off')
                plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

                plt.subplot(122)
                plt.xlabel("Iterations")
                plt.xlim([0, n_iterations])
                plt.ylim([0.4, 1])
                plt.title("Mean Similarity Ratio", fontsize=15)
                plt.plot(range(1, len(mean_similarity_ratio) + 1), mean_similarity_ratio)
                plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

                city_plot.pyplot(plt)
                plt.close("all")
                progress_bar.progress((i + 1.) / n_iterations)
