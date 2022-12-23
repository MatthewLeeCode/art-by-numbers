import labelling
import numpy as np


def test_find_representative_point() -> None:
    """ Test the find_representative_point function. 
    
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
        
    Tests that the representative point is placed as expected
    """
    shell = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    holes = [np.array([[0.5, 0.2], [0.5, 0.8], [0.8, 0.8], [0.8, 0.2]])]
    
    
    expected_point = np.array([0.25, 0.5])
    point = labelling.find_representative_point(shell, holes)
    assert np.allclose(point, expected_point)