"""
Finding the best place to put a label inside a polygon.

For simple polygons, the best place is the centroid of the polygon.
But this is not always the case for more complex polygons such as:
- polygons with holes
- concave polygons

What we want is the representation point of the polygon, which is
the point that has the most space around it inside the polygon.
"""
import shapely
import numpy as np


def find_representative_point(shell: np.ndarray, holes: list[np.ndarray]=None) -> np.ndarray:
    """
    Find the best place to put a label inside a polygon.

    Args:
        shell (np.ndarray): The outer boundary of the polygon representing the positive space.
        holes (list[np.ndarray]): A list of holes in the polygon representing negative space.
    
    Returns:
        np.ndarray: A point (x, y) that is the best place to put a label inside the polygon.
        
    Raises:
        ValueError: If the polygon is not valid.
        
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
        Representative Point

        >>> shell = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> holes = [np.array([[0.5, 0.2], [0.5, 0.8], [0.8, 0.8], [0.8, 0.2]])]
        >>> find_representative_point(shell, holes)
        array([0.25, 0.5])
    """
    assert shell.ndim == 2
    assert shell.shape[1] == 2
    assert shell.shape[0] >= 3
    if holes is not None:
        for hole in holes:
            assert hole.ndim == 2
            assert hole.shape[1] == 2
            assert hole.shape[0] >= 3
    
    polygon = shapely.geometry.Polygon(shell, holes)
    if polygon.is_valid:
        # This is the representative point of the polygon
        point = polygon.representative_point()
        # Convert to numpy array (Returns a shapely Point object)
        point = np.array([point.x, point.y])
        return point
    else:
        raise ValueError("The polygon is not valid. It is probably self-intersecting.")