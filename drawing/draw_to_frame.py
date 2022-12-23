""" 
These are helper functions for drawing to a frame (a 2D array of pixels).
"""
import cv2
import numpy as np


def get_frame(height: int, width: int, color: tuple = (255, 255, 255)) -> np.ndarray:
    """ Creates a numpy frame of the given height and width with the given color.
    
    Args:
        height (int): The height of the frame.
        width (int): The width of the frame.
        color (tuple, optional): The color of the frame. Defaults to (255, 255, 255).
    
    Returns:
        np.ndarray: The frame.
    """
    return np.full((height, width, 3), color, np.uint8)


def draw_contours(frame: np.ndarray, contours: list, color: tuple = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """ Draws the given contours on the given frame.
    
    Args:
        frame (np.ndarray): The frame to draw on.
        contours (list): The contours to draw.
        color (tuple, optional): The color of the contours. Defaults to (0, 255, 0).
        thickness (int, optional): The thickness of the contours. Defaults to 1.
        
    Returns:
        np.ndarray: The frame with the contours drawn on it.
    """
    return cv2.drawContours(frame, contours, -1, color, thickness)


def draw_label(frame: np.ndarray, label: str, position: tuple, color: tuple = (0, 0, 0), font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 1, thickness: int = 1) -> np.ndarray:
    """ Draws the given label on the given frame.
    
    Args:
        frame (np.ndarray): The frame to draw on.
        label (str): The label to draw.
        position (tuple): The position of the label.
        color (tuple, optional): The color of the label. Defaults to (0, 255, 0).
        font (int, optional): The font of the label. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        scale (float, optional): The scale of the label. Defaults to 1.
        thickness (int, optional): The thickness of the label. Defaults to 1.
        
    Returns:
        np.ndarray: The frame with the label drawn on it.
    """
    # Check types
    return cv2.putText(frame, label, position, font, scale, color, thickness, cv2.LINE_AA)


def draw_labels(frame: np.ndarray, labels: list, positions: list, color: tuple = (0, 0, 0), font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 1, thickness: int = 1) -> np.ndarray:
    """ Draws the given labels on the given frame.
    
    Args:
        frame (np.ndarray): The frame to draw on.
        labels (list): The labels to draw.
        positions (list): The positions of the labels.
        color (tuple, optional): The color of the labels. Defaults to (0, 255, 0).
        font (int, optional): The font of the labels. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        scale (float, optional): The scale of the labels. Defaults to 1.
        thickness (int, optional): The thickness of the labels. Defaults to 1.
        
    Returns:
        np.ndarray: The frame with the labels drawn on it.
    """
    for label, position in zip(labels, positions):
        frame = draw_label(frame, label, position, color, font, scale, thickness)
    return frame