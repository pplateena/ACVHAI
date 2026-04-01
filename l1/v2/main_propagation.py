import os
import time
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
import plotly.graph_objects as go
from collections import deque
from scipy.ndimage import label as _label

SAVE_DIR = "processed_propagation"
os.makedirs(SAVE_DIR, exist_ok=True)

# Config
IMAGE = 'minecraft7.png'
PHI = 20  # camera tilt from horizontal (degrees) — tune this
EDGE_THRESHOLD = 400  # sobel magnitude threshold
VERT_ANGLE_THRESH = 22  # gradient within N° of horizontal → vertical edge
CONTACT_SEARCH = 15  # pixels to look above/below for contact edge detection
VERT_WEIGHT = 20  # weight boost for vertical edge constraints
VIS_STEP = 1  # visualization downsampling (higher = faster plot)
TOP_SEARCH = 80

# Ground plane HSV mask — tune with tuner.py
HSV_LOWER = np.array([0, 0, 150])
HSV_UPPER = np.array([95, 106, 255])
ERODE_ITER = 2
OPEN_ITER = 1


# Load
original = cv2.imread(IMAGE)
original = cv2.resize(original, (800, 600))
height, width = original.shape[:2]
print(f"Image: {width}x{height}")

# Ground plane mask
# Detect background by color in HSV, then invert + clean to get foreground mask.
# ground = True means the pixel is background/ground (Y = 0).
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
kernel = np.ones((3, 3), np.uint8)
mask_bg = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
mask_fg = cv2.bitwise_not(mask_bg)
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
mask_fg = cv2.erode(mask_fg, kernel, iterations=ERODE_ITER)
ground = ~mask_fg.astype(bool)  # True = ground plane

# Sobel gradients
# Per-channel BGR Sobel, take the channel with the strongest gradient at each pixel.
# This catches edges invisible in grayscale (similar luminance, different hue).
ch_mag, ch_sx, ch_sy = [], [], []
for ch in cv2.split(cv2.GaussianBlur(original, (5, 5), 1.5)):
    sx = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=5)
    sy = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=5)
    ch_mag.append(np.sqrt(sx**2 + sy**2))
    ch_sx.append(sx)
    ch_sy.append(sy)

best = np.argmax(ch_mag, axis=0)
mag = np.max(ch_mag, axis=0)
sobelx = np.choose(best, ch_sx)
sobely = np.choose(best, ch_sy)
angle = np.arctan2(sobely, sobelx)  # gradient direction (perpendicular to edge)

# Classify pixels
# Every pixel belongs to exactly one class:
#   ground      → Y = 0 constraint
#   vert_edge   → ∂Y/∂y constraint (depth changes along vertical edge)
#   horiz_edge  → perpendicular-direction constraint (or Y=0 if contact edge)
#   face        → zero second-derivative constraints (flat surface)
#
# Vertical edge: normal points horizontally → gradient angle near 0° or 180°
# → |sin(angle)| is small
VERT_SIN_THRESH = np.sin(np.radians(VERT_ANGLE_THRESH))

edges = (mag > EDGE_THRESHOLD) & ~ground
vert_edge = edges & (np.abs(np.sin(angle)) < VERT_SIN_THRESH)
horiz_edge = edges & ~vert_edge
face = ~ground & ~edges

def flat(x, y):
    return int(x) + int(y) * width

# Detect seed top-edge pixels: horizontal edges with open space above,
# linked to a vertical edge pixel found by horizontal scan.
seed_top = np.zeros((height, width), dtype=bool)
seed_top_link = np.full((height, width), -1, dtype=int)

for py in range(1, height - 1):
    for px in range(1, width - 1):
        if not horiz_edge[py, px]:
            continue
        y_above = max(py - CONTACT_SEARCH, 0)
        if face[y_above, px] or vert_edge[y_above, px]:
            continue
        y_below = min(py + CONTACT_SEARCH, height - 1)
        if ground[y_below, px] and not ground[y_above, px]:
            continue  # skip contact edges
        for dx in range(1, TOP_SEARCH + 1):
            found = False
            for nx in [px - dx, px + dx]:
                if 0 <= nx < width and vert_edge[py, nx]:
                    seed_top[py, px] = True
                    seed_top_link[py, px] = flat(nx, py)
                    found = True
                    break
            if found:
                break

# Expand top edge label along connected horizontal edges
top_face = seed_top.copy()
top_face_link = seed_top_link.copy()

queue = deque(zip(*np.where(seed_top)))
while queue:
    py, px = queue.popleft()
    link = top_face_link[py, px]
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = py + dy, px + dx
        if 0 <= ny < height and 0 <= nx < width:
            if horiz_edge[ny, nx] and top_face_link[ny, nx] == -1:
                top_face[ny, nx] = True
                top_face_link[ny, nx] = link
                queue.append((ny, nx))

top_edge = top_face

print(f"ground={ground.sum():,}  vert={vert_edge.sum():,}  "
      f"horiz={horiz_edge.sum():,}  top={top_edge.sum():,}  face={face.sum():,}")

# Debug images
debug_ground = original.copy()
debug_ground[ground] = (200, 0, 0)
cv2.imwrite(f"{SAVE_DIR}/debug_ground.png", debug_ground)

mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(f"{SAVE_DIR}/debug_magnitude.png", mag_vis)

cv2.imwrite(f"{SAVE_DIR}/debug_edges.png", (edges * 255).astype(np.uint8))

cls_vis = np.zeros((height, width, 3), dtype=np.uint8)
cls_vis[ground] = (200, 0, 0)
cls_vis[vert_edge] = (0, 0, 255)
cls_vis[horiz_edge] = (0, 200, 0)
cls_vis[top_edge] = (0, 165, 255)  # orange
cls_vis[face] = (160, 160, 160)
cv2.imwrite(f"{SAVE_DIR}/debug_classification.png", cls_vis)
print("Debug images saved to processed/")

# Build sparse constraint matrix
# Y_vec[x + y*width] = Y(x, y), solve A @ Y_vec ≈ b.
#
# Sign convention for vertical edge constraint:
#   ∂Y/∂y_math = 1/cos(φ), y_math increases upward, y_pixel downward
#   → ∂Y/∂y_pixel = -1/cos(φ)
#   Central difference: Y(x, y+1) - Y(x, y-1) ≈ -2/cos(φ) → RHS = -1/cos(φ)

rs, cs, vs = [], [], []
b_list = []
row = 0
vert_rows = []

def add(r, x, y, val):
    rs.append(r); cs.append(flat(x, y)); vs.append(val)

print("Building constraints...")
t0 = time.time()

for py in range(1, height - 1):
    for px in range(1, width - 1):

        if ground[py, px]:
            add(row, px, py, 1)
            b_list.append(0.0)
            row += 1

        elif vert_edge[py, px]:
            add(row, px, py + 1, 1)
            add(row, px, py - 1, -1)
            b_list.append(0.0)  # placeholder, filled per phi below
            vert_rows.append(row)
            row += 1

        elif top_edge[py, px]:
            linked_flat = top_face_link[py, px]
            rs.append(row); cs.append(flat(px, py)); vs.append(1)
            rs.append(row); cs.append(linked_flat); vs.append(-1)
            b_list.append(0.0)
            row += 1

        elif horiz_edge[py, px]:
            y_below = min(py + CONTACT_SEARCH, height - 1)
            y_above = max(py - CONTACT_SEARCH, 0)
            is_contact = ground[y_below, px] and not ground[y_above, px]

            if is_contact:
                add(row, px, py, 1)
                b_list.append(0.0)
                row += 1
            else:
                # depth constant along edge direction:
                # -ny*(Y(x+1,y)-Y(x-1,y)) + nx*(Y(x,y+1)-Y(x,y-1)) = 0
                m = mag[py, px]
                nx = sobelx[py, px] / m
                ny = sobely[py, px] / m
                add(row, px + 1, py, -ny)
                add(row, px - 1, py, ny)
                add(row, px, py + 1, nx)
                add(row, px, py - 1, -nx)
                b_list.append(0.0)
                row += 1

        else:  # face: zero second derivatives
            if px + 2 < width and px - 2 >= 0:
                add(row, px, py, -2); add(row, px+2, py, 1); add(row, px-2, py, 1)
                b_list.append(0.0); row += 1

            if py + 2 < height and py - 2 >= 0:
                add(row, px, py, -2); add(row, px, py+2, 1); add(row, px, py-2, 1)
                b_list.append(0.0); row += 1

            if px+1 < width and px-1 >= 0 and py+1 < height and py-1 >= 0:
                add(row, px+1, py+1, 1); add(row, px-1, py+1, -1)
                add(row, px+1, py-1, -1); add(row, px-1, py-1, 1)
                b_list.append(0.0); row += 1

print(f"  {row:,} constraints ({len(vert_rows):,} vertical edge rows) in {time.time()-t0:.1f}s")

A = coo_matrix((vs, (rs, cs)), shape=(row, width * height)).tocsr()
b_base = np.array(b_list, dtype=np.float64)
vert_rows = np.array(vert_rows)

# Boost vertical edge rows so they aren't swamped by face constraints
A = A.tolil()
A[vert_rows] *= VERT_WEIGHT
A = A.tocsr()

# Solve
b_vec = b_base.copy()
b_vec[vert_rows] = -(1.0 / np.cos(np.radians(PHI))) * VERT_WEIGHT

print("Solving...")
t0 = time.time()
Y_flat, *_ = lsqr(A, b_vec)
print(f"  solved in {time.time()-t0:.1f}s  |  Y range: [{Y_flat.min():.2f}, {Y_flat.max():.2f}]")

Y = Y_flat.reshape(height, width)
Y = Y - Y[ground].mean()
Y = np.clip(Y, 0, None)
print(f"  after normalization: Y range [{Y.min():.2f}, {Y.max():.2f}]")

np.save(f"{SAVE_DIR}/Y_solved.npy", Y)

# Fill face regions with correct Y gradient
# A side face at constant depth Z requires dY/dy_pixel = -1/cos(φ).
# For each connected face region: find mean Y at the top boundary,
# then fill downward: Y(py) = Y_top - (py - py_top) / cos(φ)
face_labels, n_regions = _label(face)
print(f"  {n_regions} face regions found")

phi_rad = np.radians(PHI)
Y_post = Y.copy()

for rid in range(1, n_regions + 1):
    region_mask = face_labels == rid
    rows = np.where(region_mask.any(axis=1))[0]
    if len(rows) == 0:
        continue
    py_top = rows[0]

    top_row_mask = region_mask[py_top]
    top_cols = np.where(top_row_mask)[0]
    anchor_ys = []
    for search_row in range(max(0, py_top - 2), min(height, py_top + 3)):
        for cx in top_cols:
            for dcx in range(-2, 3):
                nx = cx + dcx
                if 0 <= nx < width and (vert_edge[search_row, nx] or top_edge[search_row, nx]):
                    anchor_ys.append(Y[search_row, nx])

    if not anchor_ys:
        continue  # no anchor found — keep solver values

    y_top_val = np.mean(anchor_ys)
    region_rows, region_cols = np.where(region_mask)
    Y_post[region_rows, region_cols] = y_top_val - (region_rows - py_top) / np.cos(phi_rad)

Y_post = np.clip(Y_post, 0, None)
Y = Y_post
print(f"  after region fill: Y range [{Y.min():.2f}, {Y.max():.2f}]")

# Recover Z
# y_math = Y·cos(φ) - Z·sin(φ),  y_math = -(y_pixel - height/2)
# Z = (Y·cos(φ) + y_pixel - height/2) / sin(φ)
phi = np.radians(PHI)
y_pix = np.tile(np.arange(height).reshape(height, 1), (1, width)).astype(np.float64)
x_pix = np.tile(np.arange(width).reshape(1, width), (height, 1)).astype(np.float64)

Z_raw = (Y * np.cos(phi) + y_pix - height / 2) / np.sin(phi)
Z = Z_raw.max() - Z_raw  # flip: near=0, far=positive
X = x_pix - width / 2

# Visualize
ys = np.arange(0, height, VIS_STEP)
xs = np.arange(0, width, VIS_STEP)
yy, xx = np.meshgrid(ys, xs, indexing='ij')

img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
colors = img_rgb[yy, xx].reshape(-1, 3)
color_strs = [f'rgb({r},{g},{b})' for r, g, b in colors]

fig = go.Figure(go.Scatter3d(
    x=X[yy, xx].ravel(),
    y=Z[yy, xx].ravel(),
    z=Y[yy, xx].ravel(),
    mode='markers',
    marker=dict(size=1.5, color=color_strs),
))
fig.update_layout(
    title=f"Propagation approach — PHI={PHI}°",
    scene=dict(
        xaxis_title="X (left / right)",
        yaxis_title="depth (near → far)",
        zaxis_title="height (above ground)",
        camera=dict(eye=dict(x=-1.2, y=-1.2, z=0.8)),
        zaxis=dict(range=[0, Y.max() * 1.2]),
    )
)
fig.write_html(f"{SAVE_DIR}/reconstruction.html")
fig.show()