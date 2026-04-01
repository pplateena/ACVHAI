import cv2
import numpy as np
import plotly.graph_objects as go

IMAGE    = 'minecraft7.png'
PHI      = 20       # must match what was used when solving
VIS_STEP = 2        # increase to reduce point count and speed up plot

original = cv2.imread(IMAGE)
original = cv2.resize(original, (800, 600))
height, width = original.shape[:2]

Y_flat = np.load("processed/Y_solved.npy")

# Normalize: shift so minimum = 0, clip negatives
Y_flat = Y_flat - Y_flat.min()
Y = Y_flat.reshape(height, width)
print(f"Y range after normalization: [{Y.min():.2f}, {Y.max():.2f}]")

# Recover Z from projection equation
phi   = np.radians(PHI)
y_pix = np.tile(np.arange(height).reshape(height, 1), (1, width)).astype(np.float64)
x_pix = np.tile(np.arange(width ).reshape(1, width),  (height, 1)).astype(np.float64)

Z_raw = (Y * np.cos(phi) + y_pix - height / 2) / np.sin(phi)
Z = Z_raw.max() - Z_raw   # flip: near=0, far=positive
X = x_pix - width / 2

# Subsample
ys = np.arange(0, height, VIS_STEP)
xs = np.arange(0, width,  VIS_STEP)
yy, xx = np.meshgrid(ys, xs, indexing='ij')

img_rgb    = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
colors     = img_rgb[yy, xx].reshape(-1, 3)
color_strs = [f'rgb({r},{g},{b})' for r, g, b in colors]

fig = go.Figure(go.Scatter3d(
    x=X[yy, xx].ravel(),
    y=Z[yy, xx].ravel(),
    z=Y[yy, xx].ravel(),
    mode='markers',
    marker=dict(size=1.5, color=color_strs),
))
fig.update_layout(
    title=f"3D Reconstruction — PHI={PHI}°",
    scene=dict(
        xaxis_title="X (left / right)",
        yaxis_title="depth (near → far)",
        zaxis_title="height (above ground)",
        camera=dict(eye=dict(x=-1.2, y=-1.2, z=0.8)),
        zaxis=dict(range=[0, Y.max() * 1.2]),
    )
)
fig.write_html("processed/reconstruction.html")
fig.show()