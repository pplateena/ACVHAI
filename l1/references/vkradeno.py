import cv2
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as spl

# --- Initialization and Image Loading ---
image_data = cv2.imread("./scene_3_downsampled.png", cv2.IMREAD_GRAYSCALE)
color_image_data = cv2.imread("./scene_3_downsampled.png")

print(f"Shape of incoming image: {color_image_data.shape}")

# #1. Detect image coordinates of the ground plane.
hsv = cv2.cvtColor(color_image_data, cv2.COLOR_BGR2HSV)

GREY_SATURATION_THRESHOLD = 80
# Define the lower and upper boundaries of the color in HSV
lower_bound = np.array([0, 0, 0])  # Pure white
upper_bound = np.array([179, GREY_SATURATION_THRESHOLD, 225])  # Gray range

# Create a mask for the specified color range
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Perform a bitwise AND operation to extract the colored region
ground_plane = cv2.bitwise_and(color_image_data, color_image_data, mask=mask)

height, width = image_data.shape[:2]

# Display the result (or save)
ground_plane = np.where(ground_plane > 110, ground_plane, 0)
# ground_plot = px.imshow(ground_plane)
# ground_plot.update_layout(title="Segmented Ground Plane")
# ground_plot.show()

# --- Gradient Calculation ---
sobelx = cv2.Sobel(image_data, cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(image_data, cv2.CV_64F, dx=0, dy=1, ksize=5)

gradient_matrix = np.stack([sobelx, sobely], axis=-1)

# Calculate magnitude
gradient_mag = np.sqrt(sobely ** 2 + sobelx ** 2)

magnitude, angles = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
is_ground_plane = np.where(np.all(ground_plane != 0), 1, 0)

output = image_data.copy()

# --- Thresholds and Parameters ---
EDGE_THRESHOLD = 400
STEP = 10
PHI = 22
ONE_OVER_COS_PHI = 1 / np.cos(np.deg2rad(PHI))
ANGLE_THRESHOLD = 15
SIN_ANGLE_THRESHOLD = np.sin(np.deg2rad(ANGLE_THRESHOLD))
max_magnitude = np.max(magnitude)

# --- Edge Detection and Masking ---
edges = np.where((magnitude > EDGE_THRESHOLD) & ~is_ground_plane.astype(bool), 1, 0)

vertical_edge_mask = np.where(np.abs(np.sin(np.deg2rad(angles))) < SIN_ANGLE_THRESHOLD, 1, 0)
vertical_edges = edges & vertical_edge_mask
horizontal_edges = edges & ~vertical_edge_mask

# Find contact edges
# TODO: Improve with masks?
contact_edges = np.zeros_like(horizontal_edges)
for y in range(height):
    for x in range(width):
        if horizontal_edges[y, x] != 0:
            # Only at horizontal edges, check 10 pixels below and above
            if np.all(ground_plane[min(y + 10, height - 1), x] != 0) and np.all(ground_plane[max(y - 10, 0), x] == 0):
                contact_edges[y, x] = horizontal_edges[y, x]

combined_edges = np.where(vertical_edges, 1, np.where(contact_edges, 2, np.where(horizontal_edges, 3, 0)))
# px.imshow(combined_edges).update_layout(title="Edges (vertical, horizontal and contact)").show()

# --- Constraint Setup for Linear System ---
full_length = width * height
start = time.time()

constraint_data = []
constraint_row_indices = []
constraint_column_indices = []
b = []


def add_constraint_coefficient(condition_index: int, x: int, y: int, value: int):
    flat_index = x + y * width
    constraint_row_indices.append(condition_index)
    constraint_column_indices.append(flat_index)
    constraint_data.append(value)


current_condition_index = 0

for y in range(0, height):
    for x in range(0, width):
        # print(f"Processing pixel {x + y * width}/{full_length}")

        # Check if the object is ground plane
        if np.all(ground_plane[y, x] != 0):
            # Y(x,y) at pixel location Y must be zero
            add_constraint_coefficient(current_condition_index, x, y, 1)
            b.append(0)
            current_condition_index += 1
            continue

        if vertical_edges[y, x]:
            # Y(x, y+1) - Y(x, y-1) = 1/cos(phi)
            # TODO: Rework to Sobel gradients?
            add_constraint_coefficient(current_condition_index, x, y + 1, 1)
            add_constraint_coefficient(current_condition_index, x, y - 1, -1)
            b.append(ONE_OVER_COS_PHI)
            current_condition_index += 1
            continue

        elif horizontal_edges[y, x]:
            if contact_edges[y, x]:
                add_constraint_coefficient(current_condition_index, x, y, 1)
                b.append(0)
                current_condition_index += 1
                continue

            mag = gradient_mag[y, x]
            n_x, n_y = gradient_matrix[y, x] / mag

            # -n_y dY/dx + n_x dY/dy = 0
            # -n_y (Y(x+1, y) - Y(x-1, y)) + n_x (Y(x, y+1) - Y(x, y-1)) = 0
            add_constraint_coefficient(current_condition_index, x + 1, y, -n_y)
            add_constraint_coefficient(current_condition_index, x - 1, y, n_y)
            add_constraint_coefficient(current_condition_index, x, y + 1, n_x)
            add_constraint_coefficient(current_condition_index, x, y - 1, -n_x)
            b.append(0)
            current_condition_index += 1
        else:
            # print(f"Processing face condition h={y} w={x}")
            # Dealing with a face here

            # Constraint over dx2: d2Y/dx2 = Y(x+2, y) - 2 * Y(x, y) + Y(x-2, y) = 0
            add_constraint_coefficient(current_condition_index, x, y, -2)
            add_constraint_coefficient(current_condition_index, x + 2, y, 1)
            add_constraint_coefficient(current_condition_index, x - 2, y, 1)
            b.append(0)
            current_condition_index += 1

            # Constraint over dy2: d2Y/dy2 = Y(x, y+2) - 2 * Y(x, y) + Y(x, y-2) = 0
            add_constraint_coefficient(current_condition_index, x, y, -2)
            add_constraint_coefficient(current_condition_index, x, y + 2, 1)
            add_constraint_coefficient(current_condition_index, x, y - 2, 1)
            b.append(0)
            current_condition_index += 1

            # Constraint over dxdy: d2Y/dxdy = Y(x+1, y+1) - Y(x-1, y+1) - Y(x+1, y-1) + Y(x-1, y-1) = 0
            add_constraint_coefficient(current_condition_index, x + 1, y + 1, 1)
            add_constraint_coefficient(current_condition_index, x - 1, y + 1, -1)
            add_constraint_coefficient(current_condition_index, x + 1, y - 1, -1)
            add_constraint_coefficient(current_condition_index, x - 1, y - 1, 1)
            b.append(0)
            current_condition_index += 1

# --- Matrix Assembly and Solving ---

print(f"Number of constraints: {current_condition_index}")
print(f"B size: {len(b)}")
print(f"Number of row indices: {len(constraint_row_indices)}")
print(f"Number of column indices: {len(constraint_column_indices)}")

constraint_matrix = coo_matrix((constraint_data, (constraint_row_indices, constraint_column_indices)),
                               shape=(current_condition_index, full_length))
constraint_coefficients = np.array(b)

print(f"Constraint matrix shape: {constraint_matrix.shape}")
print(f"Constraint coefficient shape: {constraint_coefficients.shape}")
print(f"Took: {time.time() - start}s")

# Solving the linear system using Least Squares (LSQR)
Y_flat, *_ = spl.lsqr(constraint_matrix.tocsr(), constraint_coefficients)

# Reshape the flat result back into the image dimensions
Y_solved = Y_flat.reshape(height, width)

# Save the resulting height map
np.save("../v1/Y_solved.npy", Y_solved)
# --- Visualization Logic (Arrows) ---
# This part corresponds to the Plotly annotation logic in photo_6 and photo_7
fig = px.imshow(color_image_data)
fig.update_layout(
    xaxis_range=[0, width],
    yaxis_range=[0, height],
    yaxis_autorange="reversed",
    title="Edge orientations"
)

arrows = []
for y in range(0, height, STEP):
    for x in range(0, width, STEP):
        if not edges[y, x]:
            continue
        rad = np.deg2rad(angles[y, x])
        normalized_length = 40
        x2 = int(x + normalized_length * np.cos(rad))
        y2 = int(y + normalized_length * np.sin(rad))

        arrows.append(go.layout.Annotation(
            x=x2,
            y=y2,
            ax=x,
            ay=y,
            text="",
            showarrow=True,
            arrowwidth=1,
            arrowcolor="red",
            arrowhead=3,
            axref="x",
            ayref="y"
        ))

fig.update_layout(annotations=arrows)
fig.show()

# --- Summary of Pipeline ---
# 1. Detect image coordinates of the ground plane.
# 2. Detect edges based on the magnitude of the Sobel gradient.
# 3. Detect faces as Image - Edges - Ground plane.
# 4. Solve Y = Y(x, y) based on every edge condition + face conditions.