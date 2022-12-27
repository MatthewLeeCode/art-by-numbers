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


def draw_labels(frame: np.ndarray, labels: list, positions: list, color: tuple = (0, 0, 0), font: int = cv2.FONT_HERSHEY_SIMPLEX, scales: float = [], thickness: int = 1) -> np.ndarray:
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
    if scales == []:
        scales = [1] * len(labels)
    for label, position, scale in zip(labels, positions, scales):
        position = position.astype(int)
        label = str(label)
        cv2.putText(frame, str(label), position, font, scale, color, thickness, cv2.LINE_AA)
    return frame