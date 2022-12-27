import numpy as np
import contouring


def test_remove_small_contours():
    """ Tests the remove_small_contours function. """
    shells = [np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]])]
    holes = [np.array([
        [[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]],
        [[[30, 30]], [[30, 50]], [[50, 50]], [[50, 30]]]
    ])]
    
    min_area = 10000
    
    out_shells, out_holes = contouring.remove_small_contours(shells, holes, min_area)
    
    assert len(out_shells) == 0, "Shells should be empty"
    assert len(out_holes) == 0, "Holes should be empty"
    
    min_area = 100

    out_shells, out_holes = contouring.remove_small_contours(shells, holes, min_area)

    assert len(out_shells) == 1, "Shells should have one element"
    assert len(out_holes[0]) == 1, "Holes should have one element"
