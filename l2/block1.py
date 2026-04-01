"""
Block 1 - Basic DFT Transformations
------------------------------------
Tasks:
  1. Load 10 images, resize to 256x256, visualize DFT amplitude & phase
  2. Phase swap: own amplitude + phase from another image -> inverse DFT
  3. Own amplitude + random phases -> inverse DFT
  4a. Original phases + constant amplitude 1/DC
  4b. Original phases + zeroed random quadrant of amplitude
"""

import os
import random
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIZE = 256          # target image size
OUT_DIR = "output"  # directory for saved results
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
# Place your own images in l2/images/ named 0.png .. 9.png (or jpg), OR
# let the script generate synthetic test images automatically.

def load_images(size: int) -> list[np.ndarray]:
    """Return list of 10 grayscale images at (size x size)."""
    img_dir = "images"
    images = []

    if os.path.isdir(img_dir):
        candidates = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        for fname in candidates[:10]:
            img = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (size, size))
                images.append(img.astype(np.float32))

    # Pad with synthetic images if fewer than 10 real ones were found
    rng = np.random.default_rng(42)
    patterns = [
        lambda r=rng: _checkerboard(size, 16),
        lambda r=rng: _checkerboard(size, 8),
        lambda r=rng: _sinusoidal(size, 4, 0),
        lambda r=rng: _sinusoidal(size, 8, np.pi / 4),
        lambda r=rng: _sinusoidal(size, 16, np.pi / 2),
        lambda r=rng: _radial_gradient(size),
        lambda r=rng: _random_blobs(size, rng),
        lambda r=rng: _concentric_circles(size),
        lambda r=rng: _diagonal_stripes(size, 12),
        lambda r=rng: _noise(size, rng),
    ]
    while len(images) < 10:
        images.append(patterns[len(images)]())

    return images[:10]


def _checkerboard(size, sq):
    x = np.arange(size)
    xx, yy = np.meshgrid(x, x)
    return (((xx // sq) + (yy // sq)) % 2 * 255).astype(np.float32)

def _sinusoidal(size, freq, phase):
    x = np.linspace(0, 2 * np.pi * freq, size)
    col = (np.sin(x + phase) * 127 + 128).astype(np.float32)
    return np.tile(col, (size, 1))

def _radial_gradient(size):
    cx = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.hypot(x - cx, y - cx)
    return (np.clip(r / r.max() * 255, 0, 255)).astype(np.float32)

def _random_blobs(size, rng):
    img = np.zeros((size, size), np.float32)
    for _ in range(8):
        cx, cy = rng.integers(30, size - 30, size=2)
        r = rng.integers(10, 40)
        cv2.circle(img, (int(cx), int(cy)), int(r), 255, -1)
    return img

def _concentric_circles(size):
    cx = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.hypot(x - cx, y - cx)
    return ((np.sin(r / 8) * 127 + 128)).astype(np.float32)

def _diagonal_stripes(size, width):
    x = np.arange(size)
    xx, yy = np.meshgrid(x, x)
    return (((xx + yy) // width) % 2 * 255).astype(np.float32)

def _noise(size, rng):
    return rng.uniform(0, 255, (size, size)).astype(np.float32)


# ---------------------------------------------------------------------------
# DFT helpers
# ---------------------------------------------------------------------------

def dft2(img: np.ndarray) -> np.ndarray:
    """Compute shifted 2D DFT of a float32 image. Returns complex array."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return F


def idft2(F: np.ndarray) -> np.ndarray:
    """Inverse shifted DFT. Returns real float32 image."""
    return np.fft.ifft2(np.fft.ifftshift(F)).real.astype(np.float32)


def amplitude(F: np.ndarray) -> np.ndarray:
    return np.abs(F)


def phase(F: np.ndarray) -> np.ndarray:
    return np.angle(F)


def log_amplitude(F: np.ndarray) -> np.ndarray:
    """Log-scaled amplitude for visualization."""
    return np.log1p(np.abs(F)).astype(np.float32)


def normalize_vis(arr: np.ndarray) -> np.ndarray:
    """Normalize float array to uint8 [0, 255]."""
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def clip_reconstruct(img: np.ndarray) -> np.ndarray:
    """Clip reconstructed image to [0, 255] and convert to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 1: Visualize amplitude & phase for all 10 images
# ---------------------------------------------------------------------------

def task1_visualize(images: list[np.ndarray]):
    print("Task 1: visualizing DFT amplitude & phase...")
    for i, img in enumerate(images):
        F = dft2(img)
        amp_vis = normalize_vis(log_amplitude(F))
        ph_vis  = normalize_vis(phase(F))          # phase in [-pi, pi] -> 0..255
        img_vis = normalize_vis(img)

        row = np.hstack([img_vis, amp_vis, ph_vis])
        cv2.imwrite(f"{OUT_DIR}/task1_img{i:02d}_orig_amp_phase.png", row)

    print(f"  Saved {len(images)} images to {OUT_DIR}/task1_img*.png")


# ---------------------------------------------------------------------------
# Task 2: Phase swap
#   Reconstruct each image using its own amplitude but the phase of image (i+1)%10
# ---------------------------------------------------------------------------

def task2_phase_swap(images: list[np.ndarray]):
    print("Task 2: phase swap...")
    Fs = [dft2(img) for img in images]
    for i, img in enumerate(images):
        j = (i + 1) % len(images)
        A = amplitude(Fs[i])
        P = phase(Fs[j])
        F_new = A * np.exp(1j * P)
        recon = idft2(F_new)

        img_vis   = normalize_vis(img)
        donor_vis = normalize_vis(images[j])
        recon_vis = clip_reconstruct(recon)

        row = np.hstack([img_vis, donor_vis, recon_vis])
        cv2.imwrite(f"{OUT_DIR}/task2_swap_img{i:02d}_with_phase_of_{j:02d}.png", row)

    print(f"  Saved to {OUT_DIR}/task2_swap_*.png")


# ---------------------------------------------------------------------------
# Task 3: Random phases
#   Reconstruct using own amplitude + random phases
# ---------------------------------------------------------------------------

def task3_random_phases(images: list[np.ndarray]):
    print("Task 3: random phases...")
    rng = np.random.default_rng(0)
    for i, img in enumerate(images):
        F = dft2(img)
        A = amplitude(F)
        random_P = rng.uniform(-np.pi, np.pi, A.shape)
        F_new = A * np.exp(1j * random_P)
        recon = idft2(F_new)

        img_vis   = normalize_vis(img)
        recon_vis = clip_reconstruct(recon)

        row = np.hstack([img_vis, recon_vis])
        cv2.imwrite(f"{OUT_DIR}/task3_random_phase_img{i:02d}.png", row)

    print(f"  Saved to {OUT_DIR}/task3_random_phase_*.png")


# ---------------------------------------------------------------------------
# Task 4a: Constant amplitude 1/DC
#   DC value = F[0,0] (center after fftshift) = mean(image) * N^2
#   Constant amplitude = 1 / abs(F_dc)
# ---------------------------------------------------------------------------

def task4a_constant_amplitude(images: list[np.ndarray]):
    print("Task 4a: constant amplitude (1/DC)...")
    for i, img in enumerate(images):
        F = dft2(img)
        h, w = F.shape
        dc_value = np.abs(F[h // 2, w // 2])
        if dc_value < 1e-6:
            dc_value = 1.0
        const_amp = 1.0 / dc_value

        P = phase(F)
        F_new = const_amp * np.exp(1j * P)
        recon = idft2(F_new)

        img_vis   = normalize_vis(img)
        recon_vis = clip_reconstruct(normalize_vis(recon))  # normalize first: tiny values

        row = np.hstack([img_vis, recon_vis])
        cv2.imwrite(f"{OUT_DIR}/task4a_const_amp_img{i:02d}.png", row)

    print(f"  Saved to {OUT_DIR}/task4a_const_amp_*.png")


# ---------------------------------------------------------------------------
# Task 4b: Zero out a random quadrant of the amplitude
# ---------------------------------------------------------------------------

QUADRANT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]

def _zero_quadrant(A: np.ndarray, q: int) -> np.ndarray:
    """Return a copy of amplitude A with quadrant q zeroed out."""
    A = A.copy()
    h, w = A.shape
    mh, mw = h // 2, w // 2
    if q == 0:   A[:mh, :mw] = 0      # top-left
    elif q == 1: A[:mh, mw:] = 0      # top-right
    elif q == 2: A[mh:, :mw] = 0      # bottom-left
    else:        A[mh:, mw:] = 0      # bottom-right
    return A


def task4b_zero_quadrant(images: list[np.ndarray]):
    print("Task 4b: zeroing a random quadrant of amplitude...")
    rng = random.Random(7)
    for i, img in enumerate(images):
        F = dft2(img)
        A = amplitude(F)
        P = phase(F)
        q = rng.randint(0, 3)
        A_mod = _zero_quadrant(A, q)
        F_new = A_mod * np.exp(1j * P)
        recon = idft2(F_new)

        # visualize: original | modified amplitude (log) | reconstruction
        img_vis   = normalize_vis(img)
        amp_orig  = normalize_vis(np.log1p(A))
        amp_mod   = normalize_vis(np.log1p(A_mod))
        recon_vis = clip_reconstruct(recon)

        row = np.hstack([img_vis, amp_orig, amp_mod, recon_vis])
        cv2.imwrite(
            f"{OUT_DIR}/task4b_zero_q{q}_{QUADRANT_NAMES[q].replace('-','_')}_img{i:02d}.png",
            row,
        )

    print(f"  Saved to {OUT_DIR}/task4b_zero_quadrant_*.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading images (size={SIZE}x{SIZE})...")
    images = load_images(SIZE)
    print(f"  Loaded {len(images)} images.")

    task1_visualize(images)
    task2_phase_swap(images)
    task3_random_phases(images)
    task4a_constant_amplitude(images)
    task4b_zero_quadrant(images)

    print(f"\nAll done. Results saved to ./{OUT_DIR}/")
    print("\nTip: place your own images in ./images/ (any count up to 10)")
    print("     to replace the synthetic test patterns.")