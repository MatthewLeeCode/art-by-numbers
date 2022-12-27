import numpy as np
import cv2


def remove_small_contours(shells: list[np.ndarray], holes: list[list[np.ndarray]], min_area: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """ Removes small contours from a list of shells and holes.
    
    Args:
        shells (list[np.ndarray]): A list of shells.
        holes (list[list[np.ndarray]]): A list of lists of holes.
        min_area (int): The minimum area of a contour to be considered.
        
    Returns:
        list[np.ndarray]: A list of shells.
        list[list[np.ndarray]]: A list of lists of holes.
    """
    # Remove small contours
    new_shells = []
    new_holes = []
    for shell, shell_holes in zip(shells, holes):
        # Get the area of the shell
        shell_area = cv2.contourArea(shell)
        
        # If the shell is too small, skip it
        if shell_area <= min_area:
            continue
        
        # Remove small holes
        new_shell_holes = []
        for hole in shell_holes:
            # Get the area of the hole
            hole_area = cv2.contourArea(hole)
            
            # If the hole is too small, skip it
            if hole_area <= min_area:
                continue
            
            # Add the hole to the list
            new_shell_holes.append(hole)
        
        # Add the shell and holes to the list
        new_shells.append(shell)
        new_holes.append(new_shell_holes)
    
    return new_shells, new_holes