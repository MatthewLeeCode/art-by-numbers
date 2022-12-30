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


def create_distance_matrix(color_labels: dict) -> np.ndarray:
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
            color1 = np.array(color1)
            color2 = np.array(color2)
            distance_matrix[i, j] = np.linalg.norm(color1 - color2)
    return distance_matrix


def find_closest_color(color_key: int, distance_matrix: list[list]) -> int:
    """ Finds the next closest color to the color
    with the given key.

    Args:
        color_key: The key of the color to find the next closest color for.
        distance_matrix: A distance matrix comparing each color
            to every other color.

    Returns:
        The key of the next closest color.
    """
    # Finds the closest color in the distance matrix
    closest_color_index = 0
    for i, distance in enumerate(distance_matrix[color_key]):
        if distance == 0:
            continue
        if distance < distance_matrix[color_key][closest_color_index]:
            closest_color_index = i
    
    return closest_color_index


def remove_strips(image: np.ndarray, color_labels:dict, distance_matrix: np.ndarray) -> np.ndarray:
    """ Removes strips of pixels from a 2D numpy array.

    If a pixel has only 1 neighbour of the same color it is replaced with the next closest color.

    Args:
        image (np.ndarray): The image to remove strips from.
        color_labels (dict): Color labels and their RGB values.
        distance_matrix (np.ndarray): A distance matrix comparing each color
            to every other color.

    Returns:
        np.ndarray: The image with strips removed.
    """
    new_image = image.copy()
    
    # Loops through the image except for the first and last pixel
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            pixel = np.array(image[i, j, :])
            neighbours = np.array([
                image[i - 1, j, :],
                image[i + 1, j, :],
                image[i, j - 1, :],
                image[i, j + 1, :]
            ])
            num_of_neighbours = len([n for n in neighbours if np.equal(pixel, n).all()])
            
            # If the pixel has no neighbours, skip it
            if num_of_neighbours == 0:
                continue
            
            # If the pixel has only 1 neighbour, replace it with the next closest color
            if num_of_neighbours == 1:
                # Find the key of this pixel
                current_color_key = 0
                for key, value in color_labels.items():
                    if np.array_equal(pixel, value):
                        current_color_key = key
                        break
                # Find the index of this pixel
                next_color_index = find_closest_color(current_color_key, distance_matrix)
                # Replace the pixel with the next closest color
                new_image[i, j] = color_labels[next_color_index]
    return new_image