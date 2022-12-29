"""
Removes strips of pixels from a 2D numpy array.

If a pixel only has 1 neighbour, we can assume it is part of a 'strip'.
We replace the pixel with a color that is the next closest out of the colors
provided.

For example, if we have a strip of red pixels, we can replace them with
pink pixels, which are the next closest color to red. Running this a few times
will remove strips of pixels (though not entirely)
"""
import numpy as np


def create_distance_matrix(color_labels:dict) -> np.ndarray:
    """ Creates a distance matrix comparing each color
    to every other color. 
    
    Args:
        color_labels: Color labels and their RGB values.
    """
    # Sort the dictionary by key
    color_labels = dict(sorted(color_labels.items()))
    
    # Ensure there are no gaps in the label values
    assert list(color_labels.keys()) == list(range(len(color_labels))), \
        "Color labels must be sequential integers starting from 0"
    
    colors = list(color_labels.values())
    distance_matrix = np.zeros((len(colors), len(colors)))
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors):
            distance_matrix[i, j] = np.linalg.norm(color1 - color2)
    return distance_matrix