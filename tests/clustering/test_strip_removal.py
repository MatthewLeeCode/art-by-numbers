import numpy as np
import clustering


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
    distance_matrix = clustering.create_distance_matrix(color_labels)

    expected = np.array([[0., 441.67295593, 360.62445841, 360.62445841, 360.62445841],
                         [441.67295593,   0., 255., 255., 255.],
                         [360.62445841, 255.,   0., 360.62445841, 360.62445841],
                         [360.62445841, 255., 360.62445841,   0., 360.62445841],
                         [360.62445841, 255., 360.62445841, 360.62445841, 0.]])
    
    assert np.allclose(distance_matrix, expected), "Distance matrix is incorrect"


def test_remove_strips():
    """ Removes strips of pixels from a 2D numpy array.
    If a pixel has only 1 neighbour of the same color it is replaced with the next closest color.
    """
    image = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 2, 2, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]])
    empty_arr = np.zeros(image.shape)
    # Add 2 channels
    image = np.stack((image, empty_arr, empty_arr), axis=2)
    
    color_labels = {
        1: np.array([0, 0, 0]),
        0: np.array([1, 0, 0]),
        2: np.array([2, 0, 0])
    }
    distance_matrix = clustering.create_distance_matrix(color_labels)
    image = clustering.remove_strips(image, color_labels, distance_matrix)

    expected = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 1, 1],
                         [1, 0, 1, 1, 1],
                         [1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1]])
    expected = np.stack((expected, empty_arr, empty_arr), axis=2)
    
    assert np.equal(image, expected).all(), "Image is incorrect"