"""
Integration test for processing example, simplified images 
"""
from paint_by_numbers import PaintByNumbers
import cv2
import pytest


@pytest.mark.skip(reason="Integration test")
def test_example_image_grid():
    """
    Tests tests/images/examples/grid.jpg with paint by numbers
    """
    grid_image = cv2.imread("tests/images/examples/grid.jpg")
    
    # Create a paint by numbers object
    pbn = PaintByNumbers(grid_image, 2)
    
    # Process the image
    final_image = pbn.process()
    
    # Save the image
    cv2.imwrite("tests/images/outputs/grid_pbn.png", final_image)


@pytest.mark.skip(reason="Integration test")
def test_example_image_target():
    """
    Tests tests/images/examples/grid.jpg with paint by numbers
    """
    target_image = cv2.imread("tests/images/examples/target.jpg")
    
    # Create a paint by numbers object
    pbn = PaintByNumbers(target_image, 2)
    
    # Process the image
    final_image = pbn.process()
    
    # Save the image
    cv2.imwrite("tests/images/outputs/target_pbn.png", final_image)

    # Compare the image to the expected output
    expected_image = cv2.imread("tests/images/expected/target_pbn.png")

    
#@pytest.mark.skip(reason="Integration test")
def test_example_image_frog():
    """
    Tests tests/images/examples/frog.jpg with paint by numbers
    """
    frog_image = cv2.imread("tests/images/examples/frog.png")
    
    # Create a paint by numbers object
    pbn = PaintByNumbers(frog_image, 10, min_area=50, width=1024, height=1024)
    pbn.label_scale = 0.4
    
    # Process the image
    final_image = pbn.process()
    
    # Save the image
    cv2.imwrite("tests/images/outputs/frog_pbn.png", final_image)

    # Compare the image to the expected output
    #expected_image = cv2.imread("tests/images/expected/frog_pbn.png")

    #assert final_image.all() == expected_image.all()