import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# ------------------------------------------------------------------
# bring the handler into scope  ─–––– replace with your own import
from rlhfblender.projections.inverse_projection_handler import InverseProjectionHandler

# ------------------------------------------------------------------


def true_fn(x, y):
    """Ground‑truth surface: change this to probe other functions."""
    return np.sin(x) * np.cos(y)


def make_dataset(n_grid: int = 25, n_val: int = 250, rng=np.random.default_rng(0)):
    """Return (grid_coords, grid_values, val_coords, val_values)."""
    xs = np.linspace(-3, 3, n_grid)
    ys = np.linspace(-3, 3, n_grid)
    gx, gy = np.meshgrid(xs, ys)
    grid_coords = np.c_[gx.ravel(), gy.ravel()]
    grid_values = true_fn(grid_coords[:, 0], grid_coords[:, 1])

    val_coords = rng.uniform(-3, 3, size=(n_val, 2))
    val_values = true_fn(val_coords[:, 0], val_coords[:, 1])
    return grid_coords, grid_values, val_coords, val_values


def decode_and_show(b64_png: str, title: str):
    """Helper: show the base‑64 PNG that precompute_interpolated_surface returns."""
    buf = io.BytesIO(base64.b64decode(b64_png))
    img = plt.imread(buf, format="png")
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def run_one(method: str, resolution: int = 300):
    grid_c, grid_v, val_c, val_v = make_dataset()
    meta = InverseProjectionHandler.precompute_interpolated_surface(
        grid_coords=grid_c,
        grid_values=grid_v,
        additional_coords=val_c,  # purely for the plot: white dots if you un‑comment in source
        additional_values=val_v,
        resolution=resolution,
        method=method,
    )

    # numeric error at the *validation* points
    pred_val = griddata(grid_c, grid_v, val_c, method=method)  # SciPy’s own interp.
    mae = np.mean(np.abs(pred_val - val_v))
    print(f"[{method:<7}] validation MAE = {mae:.4f}")

    decode_and_show(meta["image"], f"{method} interpolation")


if __name__ == "__main__":
    # Try as many methods as you like
    methods = ["nearest", "linear", "cubic", "rbf"]
    for m in methods:
        run_one(m)
