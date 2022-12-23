"""
Tests the functions of the contouring module.

This testing module could be considered redundant as the contour functions
really just use opencv. However, it is useful to test that the functions    
are working as expected and that the expected contours are being found.

You never know when opencv might change their functions and break our code.
Or maybe at some stage we will want to change the contouring functions to   
use a different library. This testing module will help us to ensure that
the functions are working as expected.

Additionally, these tests helped me better understand the output format.
Especially the hierarchy.
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
    # value along the boarder (This means for a square, just the corners).
    expected_contours = np.array([
        [[1, 1]], # Top Left
        [[1, 2]],
        [[1, 3]], # Top Right
        [[2, 3]],
        [[3, 3]], # Bottom Right
        [[3, 2]],
        [[3, 1]], # Bottom Left
        [[2, 1]]
    ])
    
    return image, expected_contours


def create_test_image_multiple_squares() -> tuple[np.ndarray, np.ndarray]:
    """ Creates a test image with multiple squares.
    
    is a 10x10 image with 2 squares, one 3x3 top left and one 2x2 bottom right.
    Values of the squares are '1' while the rest are '0'.
    
    returns:
        np.ndarray: the test image.
        np.ndarray: The expected contours.
    """
    # create a test image
    image = np.zeros((10, 10), dtype=np.uint8)
    
    # create the squares
    image[1:4, 1:4] = 1
    image[7:9, 7:9] = 1
    
    # the expected contours to compare against. Contour will wrap clockwise at
    # the edge of the square. only the required points are given, not each
    # value along the boarder (This means for a square, just the corners).
    expected_contours = []
    
    # Bottom right square
    expected_contours.append(
        np.array([
            [[7, 7]],
            [[7, 8]],
            [[8, 8]],
            [[8, 7]]
        ])
    )
    
    # Top left square
    expected_contours.append(
        np.array([
            [[1, 1]], # Top Left
            [[1, 2]],
            [[1, 3]], # Top Right
            [[2, 3]],
            [[3, 3]], # Bottom Right
            [[3, 2]],
            [[3, 1]], # Bottom Left
            [[2, 1]]
        ])
    )
    
    return image, expected_contours


def create_test_image_with_embedded_squares() -> tuple[np.ndarray, np.ndarray]:
    """ Creates a test image with embedded squares.
    
    Is a 50x50 image with three squares:
    A: 40x40 square in the center with a thickness of 2 pixels (not filled)
    B: 30x30 square in the center with a thickness of 2 pixels (not filled)
    C: 20x20 square in the center filled
    
    This will result in outer and inner contours for box A and B. We can call outer
    contours A1 and B1 and inner contours A2 and B2. The inner contours will be
    embedded in the outer contours.
    
    In total, there will be 5 contours:
    A1, B1, A2, B2, C
    
    returns:
        np.ndarray: the test image.
        np.ndarray: The expected contours.
    """
    # create a test image
    image = np.zeros((50, 50), dtype=np.uint8)
    
    # create the squares
    
    # A: 40x40 square in the center with a thickness of 2 pixels (not filled)
    image[5:45, 5:45] = 1
    image[7:43, 7:43] = 0
    
    # B: 30x30 square in the center with a thickness of 2 pixels (not filled)
    image[10:40, 10:40] = 1
    image[12:38, 12:38] = 0
    
    # C: 20x20 square in the center filled
    image[15:35, 15:35] = 1
    
    # The expected hierarchy to compare against.
    # We are using 'RETR_CCOMP' which will return the contours in 2 levels (Parent, children) 
    expected_hierachy = []
    
    # C Has no children so the hierarchy is -1. 
    expected_hierachy.append(np.array([1, -1, -1, -1]))
    
    # B1 has one child (B2)
    expected_hierachy.append(np.array([3, 0, 2, -1]))
    
    # B2 has no children
    expected_hierachy.append(np.array([-1, -1, -1, 1]))
    
    # A1 has one child (A2)
    expected_hierachy.append(np.array([-1, 1, 4, -1]))
    
    # A2 has no children
    expected_hierachy.append(np.array([-1, -1, -1, 3]))
            
    expected_hierachy = np.array(expected_hierachy)
    
    return image, expected_hierachy
    

def test_get_mask_contours_square() -> None:
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
    

def test_get_mask_contours_multiple_squares() -> None:
    """ Tests the get_mask_contours function with multiple squares. """
    # Create a test image
    image, expected_contours = create_test_image_multiple_squares()
    
    # Get the contours
    contours, hierarchy = contouring.get_mask_contours(image)
    
    # Check the number of contours
    assert len(contours) == 2
    
    # Check the contours
    assert np.array_equal(contours[0], expected_contours[0])
    assert np.array_equal(contours[1], expected_contours[1])
    
    # Check the hierarchy
    # There are no parent-child relationships with the two squares.
    # However, the first value in the hierarchy array is the next contour
    # The second value is the previous contour
    # So the first contour should have the second contour as the next contour
    # and the second contour should have the first contour as the previous contour
    expected_hierarchy = np.array([[[1, -1, -1, -1], [-1, 0, -1, -1]]])
    assert np.array_equal(hierarchy, expected_hierarchy)
    
    
def test_get_mask_contours_embedded_squares() -> None:
    """ Tests the get_mask_contours function with embedded squares. """    
    # Create a test image
    image, expected_hierarchy = create_test_image_with_embedded_squares()
    
    # Get the contours
    contours, hierarchy = contouring.get_mask_contours(image)
    
    # There is only one hierarchy tree so we can just get the first one
    hierarchy = hierarchy[0]
    
    # Check the number of contours
    assert len(hierarchy) == 5

    # Check the hierarchy
    assert np.array_equal(hierarchy, expected_hierarchy)