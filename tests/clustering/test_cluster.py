"""
Tests the functions of the clustering module.
"""
import clustering
import numpy as np


def create_test_image():
    """ Creates a test image
    
    Is a 3x3 pixel RGB image with the following colors:
        - Red
        - Blue
        - Green
        - White
        - Black
    
    Returns:
        np.ndarray: The test image.
    """
    # Dictionary of the coordinates and colors in those coordinates
    color_locations = {
        (0, 0): (255, 0, 0),
        (0, 2): (0, 0, 255),
        (2, 2): (0, 255, 0),
        (2, 0): (255, 255, 255)
    }
    
    # Create a test image
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    
    # Assign those colors to the image
    for location, color in color_locations.items():
        image[location] = color
    
    return image, color_locations


def test_kmeans_cluster():
    """ Tests the kmeans_cluster function. """
    # Create a test image
    image, _ = create_test_image()
    
    # Perform the K-means clustering
    cluster_dict, clustered_image = clustering.kmeans_cluster(image, 5)
    
    # Check that the cluster dictionary contains the correct number of clusters
    assert len(cluster_dict) == 5
    
    # Check that the cluster dictionary contains the correct colors
    # The order of the colors is not guaranteed
    assert (0, 0, 0) in cluster_dict.values()
    assert (0, 0, 255) in cluster_dict.values()
    assert (255, 0, 0) in cluster_dict.values()
    assert (0, 255, 0) in cluster_dict.values()
    assert (255, 255, 255) in cluster_dict.values()

    # Check that no other colors are in the cluster dictionary
    assert len(cluster_dict.values()) == 5
    
    # Check that clustered image matches the original image
    assert np.array_equal(image, np.array(clustered_image))
    
    
def test_create_mask():
    """ Tests the create_mask function. """
    # Create a test image
    image, color_locations = create_test_image()
    
    # Loop through the color locations and create a mask for each color
    for location, color in color_locations.items():
        mask = clustering.create_mask(image, color)
        
        # Check that the mask matches the expected image
        expected_mask = np.zeros((3, 3), dtype=np.uint8)
        
        # A value of '1' should be at the position of the color
        expected_mask[location] = 1
        
        # Assert that the masks are equal
        assert np.array_equal(expected_mask, np.array(mask))

def test_create_mask_with_many_color():
    """ Tests that create_mask function can handle many of the same color """
    # Create a test image
    image, _ = create_test_image()
    
    # Add another red point to the image
    image[1, 1] = (255, 0, 0)
    
    # Create the mask for the red color
    mask = clustering.create_mask(image, (255, 0, 0))
    
    # Check that the mask matches the expected image
    expected_mask = np.zeros((3, 3), dtype=np.uint8)
    expected_mask[0, 0] = 1
    expected_mask[1, 1] = 1
    assert np.array_equal(expected_mask, np.array(mask))

def test_clustering_integration():
    """
    Integration test for the clustering module.
    
    Tests:
        - kmeans_cluster
        - create_mask
    """
    # Create a test image
    image, _ = create_test_image()
    
    # Add another red point to the image
    image[1, 1] = (255, 0, 0)
    
    # Perform the K-means clustering
    _, clustered_image = clustering.kmeans_cluster(image, 5)
    
    # Create a mask for the red color
    mask = clustering.create_mask(clustered_image, (255, 0, 0))
    
    # Check that the mask matches the expected image
    expected_mask = np.zeros((3, 3), dtype=np.uint8)
    expected_mask[0, 0] = 1
    expected_mask[1, 1] = 1
    assert np.array_equal(expected_mask, np.array(mask))
    
    # Create a mask for the blue color
    mask = clustering.create_mask(clustered_image, (0, 0, 255))
    
    # Check that the mask matches the expected image
    expected_mask = np.zeros((3, 3), dtype=np.uint8)
    expected_mask[0, 2] = 1
    assert np.array_equal(expected_mask, np.array(mask))
