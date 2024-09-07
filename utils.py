import numpy as np

def calulate_similarity_to_nehighbor(vector1, vector2):
    '''
    Calculate euclidian distance to neighbor

    :return:
    '''

    return np.linalg.norm(np.array(vector1) - np.array(vector2))