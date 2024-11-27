import numpy as np
import cv2
import matplotlib.pyplot as plt



# Compute Root mean square error for motion vectors
def compute_rmse(original, smoothed):
    diff = original - smoothed
    # diff = smoothed - original
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # RMSE for dx, dy, da separately
    return rmse

# Compute jerkiness (std deviation of frame-to-frame differences)
def compute_jerkiness(transforms):
    frame_differences = np.diff(transforms, axis=0)
    jerkiness = np.std(frame_differences, axis=0)
    return jerkiness

