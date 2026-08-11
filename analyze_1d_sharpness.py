import os
import glob
import numpy as np
import matplotlib.pyplot as plt


RESULT_DIR = "results/1d_sgd_nesterov"
OUTPUT_DIR = "plots/1d_sgd_nesterov"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def local_curvature(xs, losses):
    """
    Estimate second derivative at alpha=0:

        L''(0) ≈ [L(-h) - 2L(0) + L(+h)] / h^2

    Larger value = sharper along this direction.
    """

    center = np.argmin(np.abs(xs))

    if center == 0 or center == len(xs) - 1:
        raise ValueError("alpha=0 must have neighbors.")

    h_left = xs[center] - xs[center - 1]
    h_right = xs[center + 1] - xs[center]

    if not np.isclose(h_left, h_right):
        raise ValueError("Grid must be evenly spaced.")

    h = h_right

    curvature = (
        losses[center - 1]
        - 2 * losses[center]
        + losses[center + 1]
    ) / (h ** 2)

    return curvature


def basin_rise(xs, losses):
    """
    Additional, less-local sharpness measure:

    average loss increase at alpha=-1 and +1
    relative to alpha=0.
    """

    center = np.argmin(np.abs(xs))

    L0 = losses[center]

    left_rise = losses[0] - L0
    right_rise = losses[-1] - L0

    return (left_rise + right_rise) / 2


def main():

    files = sorted(
        glob.glob(
            os.path.join(
                RESULT_DIR,
                "epoch_*.npz"
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No NPZ files found in {RESULT_DIR}"
        )

    epochs = []

    train_curvatures = []
    test_curvatures = []

    train_basin_rise = []
    test_basin_rise = []

    train_center_losses = []
    test_center_losses = []

    print(
        "\n"
        "Epoch | Train L(0) | Test L(0) | "
        "Train Curv | Test Curv | "
        "Train Rise | Test Rise"
    )

    print("-" * 100)

    for path in files:

        data = np.load(path)

        epoch = int(data["epoch"])

        xs = data["xs"]
        train_losses = data["train_losses"]
        test_losses = data["test_losses"]

        center = np.argmin(
            np.abs(xs)
        )

        train_L0 = train_losses[center]
        test_L0 = test_losses[center]

        train_curv = local_curvature(
            xs,
            train_losses
        )

        test_curv = local_curvature(
            xs,
            test_losses
        )

        train_rise = basin_rise(
            xs,
            train_losses
        )

        test_rise = basin_rise(
            xs,
            test_losses
        )

        epochs.append(epoch)

        train_center_losses.append(
            train_L0
        )

        test_center_losses.append(
            test_L0
        )

        train_curvatures.append(
            train_curv
        )

        test_curvatures.append(
            test_curv
        )

        train_basin_rise.append(
            train_rise
        )

        test_basin_rise.append(
            test_rise
        )

        print(
            f"{epoch:5d} | "
            f"{train_L0:10.4f} | "
            f"{test_L0:9.4f} | "
            f"{train_curv:10.4f} | "
            f"{test_curv:9.4f} | "
            f"{train_rise:10.4f} | "
            f"{test_rise:9.4f}"
        )

    # Convert to arrays
    epochs = np.array(epochs)

    train_curvatures = np.array(
        train_curvatures
    )

    test_curvatures = np.array(
        test_curvatures
    )

    train_basin_rise = np.array(
        train_basin_rise
    )

    test_basin_rise = np.array(
        test_basin_rise
    )

    # ---------------------------------------
    # Sort numerically by epoch
    # ---------------------------------------

    order = np.argsort(epochs)

    epochs = epochs[order]

    train_curvatures = train_curvatures[order]
    test_curvatures = test_curvatures[order]

    train_basin_rise = train_basin_rise[order]
    test_basin_rise = test_basin_rise[order]

    # ---------------------------------------
    # Plot local curvature
    # ---------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_curvatures,
        marker="o",
        label="Train"
    )

    plt.plot(
        epochs,
        test_curvatures,
        marker="o",
        label="Test"
    )

    # LR drops
    for lr_epoch in [150, 225, 275]:
        plt.axvline(
            lr_epoch,
            linestyle="--",
            alpha=0.5
        )

    plt.xlabel("Epoch")
    plt.ylabel("Estimated local curvature")
    plt.title(
        "1D Local Sharpness During Training\n"
        "SGD + Nesterov"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "sharpness_curvature_vs_epoch.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved: {path}")

    # ---------------------------------------
    # Plot basin rise
    # ---------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_basin_rise,
        marker="o",
        label="Train"
    )

    plt.plot(
        epochs,
        test_basin_rise,
        marker="o",
        label="Test"
    )

    for lr_epoch in [150, 225, 275]:
        plt.axvline(
            lr_epoch,
            linestyle="--",
            alpha=0.5
        )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Average loss increase from α=0 to |α|=1"
    )

    plt.title(
        "1D Basin Sharpness During Training\n"
        "SGD + Nesterov"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "sharpness_basin_rise_vs_epoch.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

    # ---------------------------------------
    # Save metrics
    # ---------------------------------------

    metrics_path = os.path.join(
        RESULT_DIR,
        "sharpness_metrics.npz"
    )

    np.savez(
        metrics_path,
        epochs=epochs,
        train_curvature=train_curvatures,
        test_curvature=test_curvatures,
        train_basin_rise=train_basin_rise,
        test_basin_rise=test_basin_rise,
    )

    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
