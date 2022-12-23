from unittest.mock import patch
import drawing
import numpy as np
import cv2


def test_get_frame() -> None:
    """ Tests drawing.get_frame() 
    
    Expects the following:
        - The frame is the correct size
        - The frame is the correct color
    """
    # Create a frame
    frame = drawing.get_frame(10, 20, (0, 0, 0))
    
    # Check that the frame is the correct size
    assert frame.shape == (10, 20, 3)
    
    # Check that the frame is the correct color
    assert (frame == np.zeros((10, 20, 3), np.uint8)).all()
    

def test_draw_contours() -> None:
    """ Tests drawing.draw_contours() 
    
    Expects the following:
        - The contours are drawn on the frame
    """
    # Creates a 20x20 frame with a white background
    # We don't use 'get_frame' here so that the test is independant.
    # I'm fully aware that these are simple functions, but testing
    # helps me sleep at night.
    frame = np.ones((20, 20, 3), np.uint8) * 255
    
    # A contour that is a square 5x5 in the center of the frame
    contour = np.array([[[5, 5]], [[5, 10]], [[10, 10]], [[10, 5]]])
    
    # Draw the contour on the frame
    drawing.draw_contours(frame, [contour], (0, 0, 0), 1)

    # Check that the contour is drawn on the frame
    assert (frame[5:10, 5] == np.zeros((5, 1, 3), np.uint8)).all()
    assert (frame[5:10, 10] == np.zeros((5, 1, 3), np.uint8)).all()
    assert (frame[5, 5:10] == np.zeros((1, 5, 3), np.uint8)).all()
    assert (frame[10, 5:10] == np.zeros((1, 5, 3), np.uint8)).all()

    # Tests draw contours with a thickness of 2 
    frame = np.ones((20, 20, 3), np.uint8) * 255
    
    # Draw the contour on the frame with thickness 2
    drawing.draw_contours(frame, [contour], (0, 0, 0), 2)
    
    # Check that the contour is drawn on the frame
    assert (frame[5:11, 5] == np.zeros((6, 1, 3), np.uint8)).all()
    assert (frame[5:11, 10] == np.zeros((6, 1, 3), np.uint8)).all()
    assert (frame[5, 5:11] == np.zeros((1, 6, 3), np.uint8)).all()
    assert (frame[10, 5:11] == np.zeros((1, 6, 3), np.uint8)).all()
    
    # Tests draw contours with a color of (255, 0, 0)
    frame = np.ones((20, 20, 3), np.uint8) * 255
    
    # Draw the contour on the frame 
    drawing.draw_contours(frame, [contour], (255, 0, 0), 1)
    
    # Check that the contour is drawn on the frame
    assert (frame[5, 5] == (255, 0, 0)).all()
    assert (frame[5, 10] == (255, 0, 0)).all()
    assert (frame[10, 5] == (255, 0, 0)).all()
    assert (frame[10, 10] == (255, 0, 0)).all()
    
# Patch opencv.putText using pytest
@patch('cv2.putText')
def test_draw_labels(mock_putText) -> None:
    """ Tests that draw_labels works correctly 
    
    We're just testing that the function calls the opencv
    func 'putText' with the correct params
    
    Args:
        frame (np.ndarray): The frame to draw on.
        labels (list): The labels to draw.
        positions (list): The positions of the labels.
        color (tuple, optional): The color of the labels. Defaults to (0, 255, 0).
        font (int, optional): The font of the labels. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        scale (float, optional): The scale of the labels. Defaults to 1.
        thickness (int, optional): The thickness of the labels. Defaults to 1.
    """
    # Create a frame
    frame = drawing.get_frame(10, 20, (0, 0, 0))
    
    # Create some labels
    labels = ['label1', 'label2']
    
    # Create some positions
    positions = [(1, 1), (2, 2)]
    
    # Draw the labels on the frame
    drawing.draw_labels(frame, labels, positions)
    
    # Check that cv2.putText was called with the correct params
    assert mock_putText.call_count == 2
    mock_putText.assert_any_call(frame, 'label1', (1, 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    mock_putText.assert_any_call(frame, 'label2', (2, 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

