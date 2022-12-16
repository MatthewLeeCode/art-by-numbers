from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def kmeans_cluster(image:Image, k:int) -> dict:
    """ Performs a K-means cluster. Returns a dict containing the cluster labels and colors.
    
    Args:
        image (Image): The image to cluster.
        k (int): The number of clusters.
        
    Returns:
        dict: A dictionary containing the cluster labels and colors. Colors are RGB values.
        Image: The clustered image.
        
    Examples:
        >>> kmeans_cluster(image, 3)
        {0: (0, 0, 0), 1: (255, 255, 255), 2: (255, 0, 0)}, Image
    """
    # Check types
    assert isinstance(image, Image.Image) or isinstance(image, np.ndarray)
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
    
    # Create a dictionary of cluster labels and colors
    cluster_dict = dict(zip(labels, cluster_colors))
    
    # Reshape the labels to the original image shape
    labels = np.reshape(labels, (w, h))

    # Create a new image array
    new_image_array = np.zeros((w, h, d), dtype=np.uint8)
    
    # Assign the cluster colors to the new image array
    for i in range(w):
        for j in range(h):
            new_image_array[i, j, :] = cluster_dict[labels[i, j]]
            
    # Convert the new image array to an image
    new_image = Image.fromarray(new_image_array)
    
    return cluster_dict, new_image
    