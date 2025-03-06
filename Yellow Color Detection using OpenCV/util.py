import cv2
import numpy as np

def get_limits(color):
    """
    Computes the lower and upper HSV limits for color detection.
    
    Args:
        color (tuple): BGR color value (B, G, R)
    
    Returns:
        lower_limit (np.array): Lower bound for HSV thresholding.
        upper_limit (np.array): Upper bound for HSV thresholding.
    """
    bgr_color = np.uint8([[color]])  # Convert tuple to NumPy array (1,3)
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    hue = hsv_color[0][0][0]  # Extract Hue

    lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit