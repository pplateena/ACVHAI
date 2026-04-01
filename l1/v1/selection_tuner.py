import cv2
import numpy as np

# 1548 x 1080
original = cv2.imread('../scenes/minecraft4.png')
original_resized = cv2.resize(original, (800, 600))
hsv = cv2.cvtColor(original_resized, cv2.COLOR_BGR2HSV)

mask_bg = np.zeros((600, 800), dtype=np.uint8)

def nothing(x):
    pass

cv2.namedWindow("mask tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mask tuner", 1200, 700)

cv2.createTrackbar("H min  0-180", "mask tuner",   0, 180, nothing)
cv2.createTrackbar("H max  0-180", "mask tuner",  20, 180, nothing)
cv2.createTrackbar("S min  0-255", "mask tuner",  40, 255, nothing)
cv2.createTrackbar("S max  0-255", "mask tuner",  80, 255, nothing)
cv2.createTrackbar("V min  0-255", "mask tuner", 180, 255, nothing)
cv2.createTrackbar("V max  0-255", "mask tuner", 230, 255, nothing)
cv2.createTrackbar("erode  iters", "mask tuner",   1,  10, nothing)
cv2.createTrackbar("open   iters", "mask tuner",   1,  10, nothing)

def mouse_callback(event, x, y, flags, param):
    # clicks are on the combined image (800+800 wide), remap x if on mask side
    src_x = x if x < 800 else x - 800
    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= src_x < 800 and 0 <= y < 600:
            bgr_px = original_resized[y, src_x]
            hsv_px = hsv[y, src_x]
            print(f"[{src_x}, {y}] BGR={bgr_px} | HSV={hsv_px} | masked={mask_bg[y, src_x]}")

cv2.setMouseCallback("mask tuner", mouse_callback)

UPDATE_MS = 60  # ~16 fps

while True:
    key = cv2.waitKey(UPDATE_MS) & 0xFF

    h_min = cv2.getTrackbarPos("H min  0-180", "mask tuner")
    h_max = cv2.getTrackbarPos("H max  0-180", "mask tuner")
    s_min = cv2.getTrackbarPos("S min  0-255", "mask tuner")
    s_max = cv2.getTrackbarPos("S max  0-255", "mask tuner")
    v_min = cv2.getTrackbarPos("V min  0-255", "mask tuner")
    v_max = cv2.getTrackbarPos("V max  0-255", "mask tuner")
    erode_iter = cv2.getTrackbarPos("erode  iters", "mask tuner")
    open_iter  = cv2.getTrackbarPos("open   iters", "mask tuner")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask_bg = cv2.inRange(hsv, lower, upper)
    mask_fg = cv2.bitwise_not(mask_bg)

    kernel = np.ones((3, 3), np.uint8)
    if open_iter > 0:
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if erode_iter > 0:
        mask_fg = cv2.erode(mask_fg, kernel, iterations=erode_iter)

    result   = cv2.bitwise_and(original_resized, original_resized, mask=mask_fg)
    mask_bgr = cv2.cvtColor(mask_fg, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([result, mask_bgr])

    # overlay current values on image
    labels = [
        f"H: {h_min}-{h_max}",
        f"S: {s_min}-{s_max}",
        f"V: {v_min}-{v_max}",
        f"erode={erode_iter}  open={open_iter}",
        f"[S] save img  [E] export cfg  [ESC] quit",
    ]
    for i, text in enumerate(labels):
        cv2.putText(combined, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("mask tuner", combined)

    if key == ord('s'):
        cv2.imwrite("result.png", result)
        print(f"[saved] result.png")

    elif key == ord('e'):
        cfg = (
            f"\n# --- exported mask config ---\n"
            f"lower = np.array([{h_min}, {s_min}, {v_min}])  # H S V min\n"
            f"upper = np.array([{h_max}, {s_max}, {v_max}])  # H S V max\n"
            f"erode_iter = {erode_iter}\n"
            f"open_iter  = {open_iter}\n"
            f"# ----------------------------\n"
        )
        print(cfg)
        with open("mask_config.txt", "w") as f:
            f.write(cfg)
        print("[saved] mask_config.txt")

    elif key == 27:
        break


# [206, 418] BGR=[157 168 204] | HSV=[  7  59 204] | masked=255

cv2.destroyAllWindows()