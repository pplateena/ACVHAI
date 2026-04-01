import cv2
import numpy as np

# 1548 x 1080
original = cv2.imread('../scenes/minecraft2.png')
original_resized = cv2.resize(original, (800, 600))
hsv = cv2.cvtColor(original_resized, cv2.COLOR_BGR2HSV)



lower = np.array([0, 38, 0])  # H S V min
upper = np.array([12, 86, 255])  # H S V max
erode_iter = 2
open_iter  = 0

kernel = np.ones((3, 3), np.uint8)

mask_bg = cv2.inRange(hsv, lower, upper)
mask_fg = cv2.bitwise_not(mask_bg)

mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=open_iter)
mask_fg = cv2.erode(mask_fg, kernel, iterations=erode_iter)

result = cv2.bitwise_and(original_resized, original_resized, mask=mask_fg)

cv2.imwrite('clean_blocks.png', result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
