"""
Finding the best place to put a label inside a polygon.

For simple polygons, the best place is the centroid of the polygon.
But this is not always the case for more complex polygons such as:
- polygons with holes
- concave polygons

What we want is the representation point of the polygon, which is
the point that has the most space around it inside the polygon.
"""
import numpy as np
import visual_center


def find_visual_center(shell: np.ndarray, holes: list[np.ndarray]=None) -> tuple[np.ndarray, float]:
    """
    Find the best place to put a label inside a polygon.

    Args:
        shell (np.ndarray): The outer boundary of the polygon representing the positive space.
        holes (list[np.ndarray]): A list of holes in the polygon representing negative space.
    
    Returns:
        np.ndarray: A point (x, y) that is the best place to put a label inside the polygon.
        float: The distance to the nearest edge of the polygon.
        
    Example:
                      Centroid
                         │
        ┌────────────────┼───────┐
        │Shell           │       │
        │        ┌───────┼────┐  │
        │        │Hole   │    │  │
        │        │       │    │  │
        │   x    │  x ◄──┘    │  │
        │   ▲    │            │  │
        │   │    └────────────┘  │
        │   │                    │
        └───┼────────────────────┘
            │
        Visual Center

        >>> shell = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
        >>> holes = [np.array([[50, 20], [50, 80], [80, 80], [80, 20]])]
        >>> find_visual_center(shell, holes)
        array([25, 50]), 25.
    """
    # Formats the shell and holes to be in the correct format
    shell = shell.reshape(-1, 2)
    if holes is not None:
        temp_holes = []
        for hole in holes:
            if hole.ndim == 3:
                hole = np.reshape(hole, (hole.shape[0], hole.shape[2]))
            temp_holes.append(hole)
        holes = temp_holes
        
    return visual_center.find_pole(shell, holes)
