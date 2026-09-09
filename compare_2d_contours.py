# compare_2d_contours.py

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

SGD_PATH = (
    "results/2d_sgd_no_momentum/"
    "epoch_300.npz"
)

NESTEROV_PATH = (
    "results/2d_sgd_nesterov/"
    "epoch_300.npz"
)

OUTPUT_DIR = "plots/2d_optimizer_comparison"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD
# ==========================================================

def load_surface(path):

    data = np.load(path)

    xs = data["xs"]
    ys = data["ys"]
    Z = data["train_Z"]

    return xs, ys, Z


sgd_xs, sgd_ys, sgd_Z = load_surface(
    SGD_PATH
)

nest_xs, nest_ys, nest_Z = load_surface(
    NESTEROV_PATH
)


if not np.allclose(
    sgd_xs,
    nest_xs
):
    raise ValueError(
        "alpha grids are different"
    )

if not np.allclose(
    sgd_ys,
    nest_ys
):
    raise ValueError(
        "beta grids are different"
    )


xs = sgd_xs
ys = sgd_ys

X, Y = np.meshgrid(
    xs,
    ys
)


# ==========================================================
# SAME CONTOUR LEVELS FOR BOTH OPTIMIZERS
# ==========================================================

# Dense levels like the Loss Landscape paper.
#
# You can adjust this depending on the loss range.
#
# These are intentionally much denser than before.

levels = np.array([
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.80,
    2.00,
    2.20,
    2.40,
    2.60,
    2.80,
    3.00,
    3.20,
    3.40,
    3.60,
    3.80,
    4.00,
    4.20,
    4.40,
    4.60,
])


# ==========================================================
# PAPER-STYLE CONTOUR FUNCTION
# ==========================================================

def draw_paper_contour(
    ax,
    Z,
    title,
):

    # NO contourf()
    #
    # This is the key change.
    # We only draw equal-loss lines.

    contours = ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        cmap="viridis",
        linewidths=1.0
    )

    # Add numeric loss labels like the paper
    ax.clabel(
        contours,
        inline=True,
        fontsize=6,
        fmt="%.2f"
    )

    # Actual trained checkpoint
    ax.scatter(
        0,
        0,
        marker="x",
        s=80,
        linewidths=2,
        color="red",
        zorder=10
    )

    ax.set_title(
        title,
        fontsize=13
    )

    ax.set_xlabel(
        "alpha"
    )

    ax.set_ylabel(
        "beta"
    )

    ax.set_xlim(
        -1,
        1
    )

    ax.set_ylim(
        -1,
        1
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    return contours


# ==========================================================
# SIDE-BY-SIDE FIGURE
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 6)
)


draw_paper_contour(
    axes[0],
    sgd_Z,
    "Plain SGD — Epoch 300"
)

draw_paper_contour(
    axes[1],
    nest_Z,
    "SGD + Nesterov — Epoch 300"
)


fig.suptitle(
    "2D Train Loss Landscapes\n"
    "ResNet-56 No-Skip on CIFAR-10",
    fontsize=16
)


plt.tight_layout()


save_path = os.path.join(
    OUTPUT_DIR,
    "sgd_vs_nesterov_epoch300_paper_style.png"
)


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {save_path}"
)
