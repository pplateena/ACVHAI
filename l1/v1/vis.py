import cv2
import numpy as np
import plotly.graph_objects as go

original = cv2.imread('../scenes/minecraft2.png')
original_resized = cv2.resize(original, (800, 600))
height, width = original_resized.shape[:2]

PHI = 25  # tune this to adjust Z (depth) recovery
Y = np.load("Y_solved.npy")

# --- Step 7: Recover Z from projection equation ---
# From lecture: y_pixel = Y*cos(θ) - Z*sin(θ)
# Solving for Z: Z = (Y*cos(θ) - y_pixel) / sin(θ)
theta = np.radians(PHI)

# y_pixel_coords[y, x] = y (the row index at each pixel)
# Image y increases downward, but the projection equation uses math convention (y up).
# Convert: y_math = -(y_pixel - height/2)
y_pixel_coords = np.tile((np.arange(height) - height / 2).reshape(height, 1), (1, width)).astype(np.float64)
y_math = -y_pixel_coords  # flip to math convention: positive = up
x_pixel_coords = np.tile(np.arange(width).reshape(1, width), (height, 1)).astype(np.float64)

# y_math = Y*cos(θ) - Z*sin(θ)  →  Z = (Y*cos(θ) - y_math) / sin(θ)
Z = (Y * np.cos(theta) - y_math) / np.sin(theta)

# X coordinate = pixel x (from lecture: x = X)
X_coords = x_pixel_coords

# --- Step 8: Visualize in 3D ---

# Sample every N pixels to keep the plot responsive
STEP = 4
ys = np.arange(0, height, STEP)
xs = np.arange(0, width, STEP)
yy, xx = np.meshgrid(ys, xs, indexing='ij')

# Get colors from original image (convert BGR→RGB)
img_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)

def to_rgb_str(pixels):
    return [f'rgb({r},{g},{b})' for r, g, b in pixels]

sampled_colors = img_rgb[yy, xx].reshape(-1, 3)
color_strings = to_rgb_str(sampled_colors)

fig = go.Figure(data=go.Scatter3d(
    x=X_coords[yy, xx].ravel(),
    y=Z[yy, xx].ravel(),   # Z=scene depth as the forward axis
    z=Y[yy, xx].ravel(),   # Y=height as the vertical axis
    mode='markers',
    marker=dict(size=1.5, color=color_strings),
))

fig.update_layout(
    title=f"old (PHI={PHI}°)",
    scene=dict(
        xaxis_title="X (left-right)",
        yaxis_title="Z (depth into scene)",
        zaxis_title="Y (height)",
        camera=dict(eye=dict(x=0, y=-2, z=0.5))  # view from camera position
    )
)
fig.write_html(f"old_reconstruction.html")
fig.show()