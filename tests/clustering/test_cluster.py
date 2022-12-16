"""
Tests the functions of the clustering module.
"""
import clustering
import numpy as np
from PIL import Image


def create_test_image():
    """ Creates a test image
    
    Is a 3 pixel RGB image with the following colors:
        - Red
        - Blue
        - Black
    
    Returns:
        np.ndarray: The test image.
    """
    # Create a test image
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    
    # Set the top left pixel to red
    image[0, 0, :] = [255, 0, 0]
    
    # Set the last pixel to blue
    image[0, 2, :] = [0, 0, 255]
    
    return image


def test_kmeans_cluster():
    """ Tests the kmeans_cluster function. """
    # Create a test image
    image = create_test_image()
    
    # Perform the K-means clustering
    cluster_dict, clustered_image = clustering.kmeans_cluster(image, 3)
    
    # Check that the cluster dictionary contains the correct number of clusters
    assert len(cluster_dict) == 3
    
    # Check that the cluster dictionary contains the correct colors
    # The order of the colors is not guaranteed
    assert (0, 0, 0) in cluster_dict.values()
    assert (0, 0, 255) in cluster_dict.values()
    assert (255, 0, 0) in cluster_dict.values()

    # Check that no other colors are in the cluster dictionary
    assert len(cluster_dict.values()) == 3
    
    # Check that clustered image matches the original image
    assert np.array_equal(image, np.array(clustered_image))
    
    
def test_create_mask():
    """ Tests the create_mask function. """
    # Create a test image
    image = create_test_image()
    
    # Test the create_mask function for the red color
    expected_image = np.zeros((1, 3), dtype=np.uint8)
    
    # Expects the first pixel to be '1' and the rest to be '0'
    expected_image[0, 0] = 1
    
    mask = clustering.create_mask(image, (255, 0, 0))

    # Check that the mask matches the expected image
    assert np.array_equal(expected_image, np.array(mask))
    
    # Test the create_mask function for the blue color
    expected_image = np.zeros((1, 3), dtype=np.uint8)

    # Expects the last pixel to be '1' and the rest to be '0'
    expected_image[0, 2] = 1

    mask = clustering.create_mask(image, (0, 0, 255))
    
    # Check that the mask matches the expected image
    assert np.array_equal(expected_image, np.array(mask))
