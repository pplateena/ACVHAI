import cv2
import numpy as np

# ── Load & preprocess (done once) ────────────────────────────────────────────
IMAGE = 'minecraft7.png'
original = cv2.imread(IMAGE)
original = cv2.resize(original, (800, 600))
height, width = original.shape[:2]

# Per-channel Sobel (computed once — doesn't change with config)
ch_mag, ch_sx, ch_sy = [], [], []
for ch in cv2.split(cv2.GaussianBlur(original, (5, 5), 1.5)):
    sx = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=5)
    sy = cv2.Sobel(ch.astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=5)
    ch_mag.append(np.sqrt(sx**2 + sy**2))
    ch_sx.append(sx)
    ch_sy.append(sy)

mag   = np.max(ch_mag, axis=0)
angle = np.arctan2(
    np.choose(np.argmax(ch_mag, axis=0), ch_sy),
    np.choose(np.argmax(ch_mag, axis=0), ch_sx),
)
mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── Window + trackbars ───────────────────────────────────────────────────────
WIN = "Classification Tuner  |  q = quit  |  s = save config"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1200, 700)

# HSV ground mask params
cv2.createTrackbar("HSV H-min",  WIN,   0, 179, lambda v: None)
cv2.createTrackbar("HSV H-max",  WIN,  30, 179, lambda v: None)
cv2.createTrackbar("HSV S-min",  WIN,   0, 255, lambda v: None)
cv2.createTrackbar("HSV S-max",  WIN,  60, 255, lambda v: None)
cv2.createTrackbar("HSV V-min",  WIN, 100, 255, lambda v: None)
cv2.createTrackbar("HSV V-max",  WIN, 255, 255, lambda v: None)
cv2.createTrackbar("Erode iter", WIN,   2,  10, lambda v: None)

# Edge + classification params
cv2.createTrackbar("Edge thresh x10", WIN,  40, 200, lambda v: None)  # divide by 10
cv2.createTrackbar("Vert angle deg",  WIN,  30,  89, lambda v: None)

def classify(h_min, h_max, s_min, s_max, v_min, v_max, erode,
             edge_thresh, vert_deg):
    # Ground mask
    hsv     = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    kernel  = np.ones((3, 3), np.uint8)
    mask_bg = cv2.inRange(hsv,
                          np.array([h_min, s_min, v_min]),
                          np.array([h_max, s_max, v_max]))
    mask_fg = cv2.bitwise_not(mask_bg)
    mask_fg = cv2.erode(mask_fg, kernel, iterations=erode)
    ground  = ~mask_fg.astype(bool)

    # Edge + classification
    edges      = (mag > edge_thresh) & ~ground
    vert_sin   = np.sin(np.radians(vert_deg))
    vert_edge  = edges & (np.abs(np.sin(angle)) < vert_sin)
    horiz_edge = edges & ~vert_edge
    face       = ~ground & ~edges

    # Color map: ground=blue, vert=red, horiz=green, face=gray
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    vis[ground]     = (200,   0,   0)
    vis[vert_edge]  = (  0,   0, 255)
    vis[horiz_edge] = (  0, 200,   0)
    vis[face]       = (160, 160, 160)

    stats = (f"ground={ground.sum():,}  vert={vert_edge.sum():,}  "
             f"horiz={horiz_edge.sum():,}  face={face.sum():,}")
    return vis, stats, dict(
        HSV_lower=(h_min, s_min, v_min),
        HSV_upper=(h_max, s_max, v_max),
        erode_iter=erode,
        edge_threshold=edge_thresh,
        vert_angle_thresh=vert_deg,
    )

mouse_pos = [0, 0]

def on_mouse(event, x, y, flags, param):
    mouse_pos[0] = x
    mouse_pos[1] = y

cv2.setMouseCallback(WIN, on_mouse)

print("Adjust sliders to tune classification.")
print("Press  s  to print current config,  q  to quit.")

while True:
    h_min  = cv2.getTrackbarPos("HSV H-min",  WIN)
    h_max  = cv2.getTrackbarPos("HSV H-max",  WIN)
    s_min  = cv2.getTrackbarPos("HSV S-min",  WIN)
    s_max  = cv2.getTrackbarPos("HSV S-max",  WIN)
    v_min  = cv2.getTrackbarPos("HSV V-min",  WIN)
    v_max  = cv2.getTrackbarPos("HSV V-max",  WIN)
    erode  = cv2.getTrackbarPos("Erode iter", WIN)
    thresh = cv2.getTrackbarPos("Edge thresh x10", WIN) * 10
    vdeg   = cv2.getTrackbarPos("Vert angle deg",  WIN)

    vis, stats, cfg = classify(h_min, h_max, s_min, s_max,
                                v_min, v_max, erode, thresh, vdeg)

    # Overlay stats text
    cv2.putText(vis, stats, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay HSV value under mouse
    mx, my = mouse_pos
    if 0 <= my < height and 0 <= mx < width:
        hsv_img = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[my, mx]
        hsv_text = f"HSV: ({h}, {s}, {v})  px: ({mx}, {my})"
        cv2.putText(vis, hsv_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WIN, vis)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    if key == ord('s'):
        print("\n── Current config ──────────────────")
        for k, v in cfg.items():
            print(f"  {k} = {v}")
        print("────────────────────────────────────\n")

cv2.destroyAllWindows()
