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


def find_visual_center(shell: np.ndarray, holes: list[np.ndarray]=None) -> np.ndarray:
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
        Visual Center Point

        >>> shell = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
        >>> holes = [np.array([[[0.5, 0.2]], [[0.5, 0.8]], [[0.8, 0.8]], [[0.8, 0.2]]])]
        >>> find_visual_center(shell, holes)
        array([0.25, 0.5])
    """
    # Formats the shell and holes to be in the correct format
    shell = shell.reshape(-1, 2)
    if holes is not None:
        holes = [hole.reshape(-1, 2) for hole in holes]

    # Create a shapely polygon
    try:
        polygon = shapely.geometry.Polygon(shell, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        # This is the visual center point of the polygon
        point = polygon.representative_point()
        area = polygon.area
        # Convert to numpy array (Returns a shapely Point object)
        point = np.array([int(point.x), int(point.y)])
    except shapely.errors.GEOSException as e:
        print(e)
        return None, None
    return point, area
