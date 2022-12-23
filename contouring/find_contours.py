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
    # Find the contours. We use 'RETR_CCOMP' to find the hierarchy of the contours.
    # 'RETR_CCOMP' provides hierarchy for parent and its child contours (The holes).
    # This is all we need for our purposes. More info here: 
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    # Filter the contours
    if min_area is not None:
        contours = [contour for contour in contours if min_area < cv2.contourArea(contour)]
    
    return contours, hierarchy


def find_shell_holes(contours: list, hierarchy: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """ Finds the shells and related holes of a list of contours.
    
    Args:
        contours (list): A list of contours.
        hierarchy (np.ndarray): The hierarchy of the contours.
        
    Returns:
        list: A list of tuples containing the shell and holes.
            tuple: A tuple containing the shell and holes.
                np.ndarray: The shell.
                list: A list of holes.
                    np.ndarray: A hole.
    """
    # Find the shells and holes
    shells = []
    for i, contour in enumerate(contours):
        # Find the parent contour
        parent_index = hierarchy[0, i, 3]
        # If there is no parent, then this is a shell
        if parent_index == -1:
            # Find the holes
            holes = []
            for j, hole in enumerate(contours):
                # Find the child contour
                child_index = hierarchy[0, j, 3]
                # If the child is the current contour, then it is a hole
                if child_index == i:
                    holes.append(hole)
            if len(holes) == 0:
                holes = None
            shells.append((contour, holes))
            
    return shells