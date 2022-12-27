import labelling
import numpy as np


def test_find_visual_center() -> None:
    """ Test the find_visual_center function. 
    
                          Centroid
                         │
        ┌────────────────┼───────┐
        │Shell           │       │
        │        ┌───────┼────┐  │
        │   x    │Hole   │    │  │
        │   ▲    │       │    │  │
        │   |    │  x ◄──┘    │  │
        │   |    │            │  │
        │   │    └────────────┘  │
        │   │                    │
        └───┼────────────────────┘
            │
        Visual Center. Can be anywhere on this vertical
        
    Tests that the visual center point is placed as expected
    """
    shell = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
    holes = [np.array([[50, 20], [50, 80], [80, 80], [80, 20]])]
    
    # The expected point y value can be anywhere from 25 to 75
    expected_x = 25
    expected_y = range(25, 75)
    expected_distance = 25.
    
    point, distance = labelling.find_visual_center(shell, holes)
    assert point[0] == expected_x, f"Expected x value {expected_x}, got {point[0]}"
    assert point[1] in expected_y, f"Expected y value {expected_y}, got {point[1]}"
    assert distance == expected_distance, f"Expected distance {expected_distance}, got {distance}"