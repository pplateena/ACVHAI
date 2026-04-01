import time
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

# --- Load & resize ---
original = cv2.imread('../scenes/minecraft3.png')
original_resized = cv2.resize(original, (800, 600))
height, width = original_resized.shape[:2]

# --- Step 2: Ground plane mask (HSV color range) ---
hsv = cv2.cvtColor(original_resized, cv2.COLOR_BGR2HSV)
lower = np.array([11, 5, 103])  # H S V min
upper = np.array([23, 53, 255])  # H S V max
erode_iter = 1
open_iter  = 1
kernel = np.ones((3, 3), np.uint8)
mask_bg = cv2.inRange(hsv, lower, upper)
mask_fg = cv2.bitwise_not(mask_bg)
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=open_iter)
mask_fg = cv2.erode(mask_fg, kernel, iterations=erode_iter)

ground_mask = ~mask_fg.astype(bool)   # True = ground plane pixel (inverse of cleaned foreground)

# --- Step 3: Sobel gradients (per BGR channel, take max magnitude) ---
channel_magnitudes = []
channel_sobelx = []
channel_sobely = []
for ch in cv2.split(cv2.GaussianBlur(original_resized, (5, 5), 1.5)):
    sx = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=5)
    sy = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=5)
    channel_magnitudes.append(np.sqrt(sx**2 + sy**2))
    channel_sobelx.append(sx)
    channel_sobely.append(sy)

best_channel = np.argmax(channel_magnitudes, axis=0)
magnitude = np.max(channel_magnitudes, axis=0)
sobelx = np.choose(best_channel, channel_sobelx)
sobely = np.choose(best_channel, channel_sobely)
angle = np.arctan2(sobely, sobelx)   # radians, range [-pi, pi]

EDGE_THRESHOLD = 740
edges = (magnitude > EDGE_THRESHOLD)  # True = edge pixel

# --- Step 4: Classify pixels ---
VERT_THRESHOLD = np.sin(np.radians(18))

# ~ == is not

ground      = ground_mask
edge        = edges & ~ground
vert_edge   = edge & (np.abs(np.sin(angle)) < VERT_THRESHOLD)
horiz_edge  = edge & ~vert_edge
face        = ~ground & ~edge

# Visualize: encode classes as colors (BGR)
#   ground     = blue
#   vert_edge  = red
#   horiz_edge = green
#   face       = white
classification_vis = np.zeros((height, width, 3), dtype=np.uint8)
classification_vis[ground]      = (255, 0,   0)
classification_vis[vert_edge]   = (0,   0,   255)
classification_vis[horiz_edge]  = (0,   255, 0)
classification_vis[face]        = (200, 200, 200)

cv2.imwrite("classification.png", classification_vis)
# cv2.imshow("Classification", classification_vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- Step 5: Build sparse linear system A @ Y_vec = b ---
# PHI only affects the RHS of vertical edge constraints (1/cos(PHI)),
# not the matrix A itself — so we build A once and solve for multiple angles.

def flat(x, y):
    return int(x) + int(y) * width

rows, cols, data = [], [], []
b = []
vert_edge_rows = []  # track which rows are vertical edge constraints
row_idx = 0

def add(r, x, y, value):
    rows.append(r)
    cols.append(flat(x, y))
    data.append(value)

print("Building constraint matrix...")
start = time.time()

for y in range(1, height - 1):
    for x in range(1, width - 1):

        if ground[y, x]:
            add(row_idx, x, y, 1)
            b.append(0)
            row_idx += 1

        elif vert_edge[y, x]:
            add(row_idx, x, y + 1,  1)
            add(row_idx, x, y - 1, -1)
            b.append(0)  # placeholder — filled per angle below
            vert_edge_rows.append(row_idx)
            row_idx += 1

        elif horiz_edge[y, x]:
            # Contact edge: horizontal edge with ground below and no ground above
            # → this is where a block meets the floor, so Y = 0
            CONTACT_SEARCH = 10
            below = min(y + CONTACT_SEARCH, height - 1)
            above = max(y - CONTACT_SEARCH, 0)
            is_contact = ground[below, x] and not ground[above, x]

            if is_contact:
                add(row_idx, x, y, 1)
                b.append(0)
                row_idx += 1
            else:
                mag = magnitude[y, x]
                nx = sobelx[y, x] / mag
                ny = sobely[y, x] / mag
                add(row_idx, x + 1, y,  -ny)
                add(row_idx, x - 1, y,   ny)
                add(row_idx, x, y + 1,   nx)
                add(row_idx, x, y - 1,  -nx)
                b.append(0)
                row_idx += 1

        else:  # face
            if x + 2 < width and x - 2 >= 0:
                add(row_idx, x,     y,  -2)
                add(row_idx, x + 2, y,   1)
                add(row_idx, x - 2, y,   1)
                b.append(0)
                row_idx += 1

            if y + 2 < height and y - 2 >= 0:
                add(row_idx, x, y,      -2)
                add(row_idx, x, y + 2,   1)
                add(row_idx, x, y - 2,   1)
                b.append(0)
                row_idx += 1

            if x + 1 < width and x - 1 >= 0 and y + 1 < height and y - 1 >= 0:
                add(row_idx, x + 1, y + 1,  1)
                add(row_idx, x - 1, y + 1, -1)
                add(row_idx, x + 1, y - 1, -1)
                add(row_idx, x - 1, y - 1,  1)
                b.append(0)
                row_idx += 1

print(f"Constraints built in {time.time() - start:.1f}s — {row_idx} equations, {len(vert_edge_rows)} vertical edge rows")

A = coo_matrix((data, (rows, cols)), shape=(row_idx, width * height)).tocsr()
b_base = np.array(b, dtype=np.float64)
vert_edge_rows = np.array(vert_edge_rows)

# Weight vertical edge constraints so they aren't drowned out by the
# ~400x more numerous face/ground constraints.
# Ratio: 681398 total / 1657 vert edges ≈ 411 — use sqrt as a balanced weight.
VERT_WEIGHT = np.sqrt(row_idx / max(len(vert_edge_rows), 1))
print(f"Vertical edge weight: {VERT_WEIGHT:.1f}")

A = A.tolil()
A[vert_edge_rows] *= VERT_WEIGHT
A = A.tocsr()

# --- Step 6: Solve ---
# PHI has negligible effect on Y (only 0.24% of constraints depend on it).
# Tune PHI in vis.py instead — it strongly affects Z recovery there.
PHI = 25
print(f"Solving for PHI={PHI}°...")
b_vec = b_base.copy()
b_vec[vert_edge_rows] = -(1.0 / np.cos(np.radians(PHI))) * VERT_WEIGHT

Y_flat, *_ = lsqr(A, b_vec)

Y = Y_flat.reshape(height, width)
np.save("Y_solved.npy", Y)
print("Saved Y_solved.npy")

