"""
Contours are lines or curves that represent the boundaries of an object 
or surface in a two-dimensional image. This is useful for creating a 'paint-by-numbers'
image where we section each colour into their contours before placing a numbered label 
on them.

Here we use opencv to find the contours of an image. 
We need to consider the following:
- How big should the contours be? We don't want to find contours that are too small. Makes it hard
  to paint them.
- Are any contours inside other contours? This will make it challenging to find a good label spot.
"""
import cv2
import numpy as np


def get_mask_contours(mask: np.ndarray, min_area: int | None=None) -> tuple[tuple, np.ndarray]:
    """ Finds the contours of a mask (see clustering.cluster_image.create_mask).
    
    Args:
        mask (np.ndarray): The mask to use for finding contours. Must be a binary image.
        min_area (int): The minimum area of a contour to be considered.
        
    Returns:
        list: A list of contours.
        hierarchy: The hierarchy of the contours.
    """
    # Check types
    assert isinstance(mask, np.ndarray)
    assert isinstance(min_area, int) or min_area is None
    
    # Find the contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter the contours
    if min_area is not None:
        contours = [contour for contour in contours if min_area < cv2.contourArea(contour)]
    
    return contours, hierarchy