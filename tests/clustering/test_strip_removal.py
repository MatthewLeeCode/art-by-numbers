import numpy as np
from clustering import create_distance_matrix


def test_create_distance_matrix():
    """ Creates a distance matrix comparing each color
    to every other color. 
    """
    color_labels = {
        1: np.array([0, 0, 0]),
        0: np.array([255, 255, 255]),
        2: np.array([255, 0, 0]),
        3: np.array([0, 255, 0]),
        4: np.array([0, 0, 255])
    }
    distance_matrix = create_distance_matrix(color_labels)

    expected = np.array([[0., 441.67295593, 360.62445841, 360.62445841, 360.62445841],
                         [441.67295593,   0., 255., 255., 255.],
                         [360.62445841, 255.,   0., 360.62445841, 360.62445841],
                         [360.62445841, 255., 360.62445841,   0., 360.62445841],
                         [360.62445841, 255., 360.62445841, 360.62445841, 0.]])
    
    assert np.allclose(distance_matrix, expected), "Distance matrix is incorrect"
