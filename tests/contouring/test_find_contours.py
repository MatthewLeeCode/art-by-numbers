"""
Tests the functions of the contouring module
"""
import contouring
import numpy as np


def create_test_image_square() -> tuple[np.ndarray, np.ndarray]:
    """ Creates a test image with a square in the middle.
    
    Is a 5x5 image with a 3x3 square in the middle. Values
    of the square are '1' while the rest are '0'.
    
    Returns:
        np.ndarray: The test image.
        np.ndarray: The expected contours.
    """
    # Create a test image
    image = np.zeros((5, 5), dtype=np.uint8)
    
    # Create the square in the center
    image[1:4, 1:4] = 1

    # The expected contours to compare against. Contour will wrap clockwise at 
    # the edge of the square. Only the required points are given, not each
    # value along the boarder.
    expected_contours = np.array([
        [[1, 1]],
        [[1, 3]],
        [[3, 3]],
        [[3, 1]]
    ])
    
    return image, expected_contours


def create_test_image_multiple_squares() -> tuple[np.ndarray, np.ndarray]:
    """ Creates a test image with multiple squares.
    
    is a 10x10 image with 2 squares, one 3x3 top left and one 2x2 bottom right.
    Values of the squares are '1' while the rest are '0'.
    
    returns:
        np.ndarray: the test image.
    """
    # create a test image
    image = np.zeros((10, 10), dtype=np.uint8)
    
    # create the squares
    image[1:4, 1:4] = 1
    image[7:9, 7:9] = 1
    
    # the expected contours to compare against. contour will wrap clockwise at
    # the edge of the square. only the required points are given, not each
    # value along the boarder.
    expected_contours = []
    
    # Top left square
    expected_contours.append([
        [[1, 1]],
        [[1, 3]],
        [[3, 3]],
        [[3, 1]]
    ])
    
    # Bottom right square
    expected_contours.append([
        [[7, 7]],
        [[7, 8]],
        [[8, 8]],
        [[8, 7]]
    ])
    
    expected_contours = np.array(expected_contours)
    
    return image, expected_contours


def test_get_mask_contours_square():
    """ Tests the get_mask_contours function with a square. """
    
    # Create a test image
    image, expected_contours = create_test_image_square()
    
    # Get the contours
    contours, hierarchy = contouring.get_mask_contours(image)
    
    # Check the number of contours
    assert len(contours) == 1
    
    # Check the contour
    assert np.array_equal(contours[0], expected_contours)
    
    # Check the hierarchy
    # There should only be one contour so their is no hierarchy. All values will be -1 
    # for the one contour that we have.
    assert np.array_equal(hierarchy, np.array([[[-1, -1, -1, -1]]]))