"""
Performs clustering on an image to find colours that are similar.
Any art-by-numbers type craft will simplify the image into a few colours because
an image could have thousands of colours and we (probably) don't have thousands of paints!

Clustering finds k colors that represent the image. All colors are then converted
into one of those k colors based on their similarity (Euclidean distance).

Once we convert the colors into k colors, we can create a mask for each color. The mask
is used in the next steps to create the art-by-numbers image.
"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans_cluster(image:np.ndarray, k:int) -> tuple[dict, np.ndarray]:
    """ Performs a K-means cluster. Returns a dict containing the cluster labels and colors.
    
    Args:
        image (np.ndarray): The image to cluster 
        k (int): The number of clusters.
        
    Returns:
        dict: A dictionary containing the cluster labels and colors. Colors are RGB values.
        Image: The clustered image.
        
    Examples:
        >>> kmeans_cluster(image, 3)
        {0: (0, 0, 0), 1: (255, 255, 255), 2: (255, 0, 0)}, Image
    """
    # Check types
    assert isinstance(image, np.ndarray)
    assert isinstance(k, int)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Get the shape of the image
    w, h, d = image_array.shape
    
    # Reshape the image to be a list of pixels
    assert d == 3
    image_array = np.reshape(image_array, (w * h, d))
    
    # Perform the K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Convert the cluster centers to RGB values
    cluster_colors = [tuple(map(int, center)) for center in cluster_centers]
    
    # For each label, create a dictionary entry
    cluster_dict = {}
    for label in labels:
        cluster_dict[label] = cluster_colors[label]
    
    # Reshape the labels to the original image shape
    labels = np.reshape(labels, (w, h))

    # Create a new image array
    new_image_array = np.zeros((w, h, d), dtype=np.uint8)
    
    # Assign the cluster colors to the new image array
    for i in range(w):
        for j in range(h):
            new_image_array[i, j] = cluster_dict[labels[i, j]]
    
    new_image = np.array(new_image_array, dtype=np.uint8)
    
    return cluster_dict, new_image


def create_mask(clustered_image:np.ndarray, rgb:tuple) -> np.ndarray:
    """ Creates a mask of the image that matches a color.
    
    Args:
        clustered_image (Image): The clustered image.
        rgb (tuple): The RGB color to mask.
        
    Returns:
        Image: The mask image which aligns with the rgb 
               values in the clustered image.
    """
    # Check types
    assert isinstance(clustered_image, np.ndarray)
    assert isinstance(rgb, tuple)
    
    # Convert the image to a numpy array
    image_array = np.array(clustered_image)
    
    # Get the shape of the image
    w, h, d = image_array.shape
    
    # Create a new image array
    mask_image_array = np.zeros((w, h), dtype=np.uint8)
    
    # Assign the cluster colors to the new image array
    for i in range(w):
        for j in range(h):
            if np.array_equal(image_array[i, j, :], rgb):
                mask_image_array[i, j] = 1
            
    # Convert the new image array to an image
    mask = np.array(mask_image_array, dtype=np.uint8)
    
    return mask
    