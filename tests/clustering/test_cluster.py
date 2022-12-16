"""
Tests the functions of the clustering module.
"""
import clustering
import numpy as np
from PIL import Image


def test_kmeans_cluster():
    """ Tests the kmeans_cluster function. """
    # Create a test image
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    
    # Set the top left pixel to red
    image[0, 0, :] = [255, 0, 0]
    
    # Set the next pixel to blue
    image[0, 1, :] = [0, 0, 255]
    
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