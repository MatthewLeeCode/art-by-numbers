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
import cv2


def remove_noise(image:np.ndarray) -> np.ndarray:
    """
    Removes noise from an image. We don't want small clusters
    of pixels to be considered as a color. We want large areas.
    Removing noise means the clustering algorithm can find larger regions

    Args:
        image (np.ndarray): The image to remove noise from.
    
    Returns:
        np.ndarray: The image with noise removed.
    """
    # Blur the image
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    
    return blur


def kmeans_cluster(image:np.ndarray, k:int) -> tuple[dict, np.ndarray]:
    """ Performs a K-means cluster. Returns a dict containing the cluster labels and colors.
    
    Args:
        image (np.ndarray): The image to cluster 
        k (int): The number of clusters.
        
    Returns:
        dict: A dictionary containing the cluster labels and colors. Colors are RGB values.
        np.ndarray: The clustered image.
        
    Examples:
        >>> kmeans_cluster(image, 3)
        {0: (0, 0, 0), 1: (255, 255, 255), 2: (255, 0, 0)}, Image
    """
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Get the shape of the image
    w, h, d = image_array.shape
    
    # Reshape the image to be a list of pixels
    assert d == 3
    image_array = np.reshape(image_array, (w * h, d))
    
    # Perform the K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(image_array)
    
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
        clustered_image (np.ndarray): The clustered image.
        rgb (tuple): The RGB color to mask.
        
    Returns:
        np.ndarray: The mask image which aligns with the rgb 
               values in the clustered image.
    """
    # Convert the image to a numpy array
    image_array = np.array(clustered_image)
    
    # Find the pixels that match the rgb color
    mask = np.all(image_array == rgb, axis=-1).astype(int)
    
    return mask


def morphology(mask:np.ndarray) -> np.ndarray:
    """ Performs morphology on the mask image.
    
    Morphology is used to remove noise from the mask image.
    
    Args:
        mask (np.ndarray): The mask image.
        
    Returns:
        np.ndarry: The morphology image.
    """
    mask = mask.astype(np.uint8)
    
    # Create a kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform morphology
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return opening