import cv2
import numpy as np

# 1548 x 1080
original = cv2.imread('../scenes/minecraft3.png')
original_resized = cv2.resize(original, (800, 600))


grey = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)

# Blur first to suppress texture noise before edge detection
blurred = cv2.GaussianBlur(grey, ksize=(5, 5), sigmaX=20)

# Compute Sobel on each BGR channel separately, then take per-pixel max magnitude.
# This catches edges that are invisible in grayscale due to similar luminance
# (e.g. yellow object against a pinkish background).
channel_magnitudes = []
channel_sobelx = []
channel_sobely = []
for ch in cv2.split(cv2.GaussianBlur(original_resized, (5, 5), 1.5)):
    sx = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=5)
    sy = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=5)
    channel_magnitudes.append(np.sqrt(sx**2 + sy**2))
    channel_sobelx.append(sx)
    channel_sobely.append(sy)

# Index of the channel with the strongest gradient at each pixel
best_channel = np.argmax(channel_magnitudes, axis=0)
magnitude = np.max(channel_magnitudes, axis=0)

# Use gradient direction from the winning channel
sobelx = np.choose(best_channel, channel_sobelx)
sobely = np.choose(best_channel, channel_sobely)
angle = np.arctan2(sobely, sobelx)  # radians, range [-pi, pi]

EDGE_THRESHOLD = 740
edges = magnitude > EDGE_THRESHOLD  # boolean mask, True = edge pixel

# Normalize magnitude to 0-255 for visualization
magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
edges_vis = (edges * 255).astype(np.uint8)

cv2.imwrite("magnitude.png", magnitude_vis)
cv2.imwrite("edges.png", edges_vis)

cv2.imshow("Sobel magnitude", magnitude_vis)
cv2.imshow("Edges (thresholded)", edges_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
