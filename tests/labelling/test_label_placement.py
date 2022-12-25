import labelling
import numpy as np


def test_find_visual_center() -> None:
    """ Test the find_visual_center function. 
    
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
        
    Tests that the visual center point is placed as expected
    """
    shell = np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]])
    holes = [np.array([[[50, 20]], [[50, 80]], [[80, 80]], [[80, 20]]])]
    
    
    expected_point = np.array([25, 50])
    point = labelling.find_visual_center(shell, holes)
    assert np.allclose(point, expected_point)