import numpy as np


def local_2d_hessian(xs, ys, Z):
    """
    Finite-difference Hessian at alpha=0, beta=0.

    H = [[d2L/da2, d2L/dadb],
         [d2L/dbda, d2L/db2]]
    """

    i0 = np.argmin(np.abs(ys))
    j0 = np.argmin(np.abs(xs))

    if i0 == 0 or i0 == len(ys) - 1:
        raise ValueError("beta=0 must have neighbors")

    if j0 == 0 or j0 == len(xs) - 1:
        raise ValueError("alpha=0 must have neighbors")

    da = xs[j0 + 1] - xs[j0]
    db = ys[i0 + 1] - ys[i0]

    # second derivative wrt alpha
    d2_da2 = (
        Z[i0, j0 + 1]
        - 2 * Z[i0, j0]
        + Z[i0, j0 - 1]
    ) / (da ** 2)

    # second derivative wrt beta
    d2_db2 = (
        Z[i0 + 1, j0]
        - 2 * Z[i0, j0]
        + Z[i0 - 1, j0]
    ) / (db ** 2)

    # mixed derivative
    d2_dadb = (
        Z[i0 + 1, j0 + 1]
        - Z[i0 + 1, j0 - 1]
        - Z[i0 - 1, j0 + 1]
        + Z[i0 - 1, j0 - 1]
    ) / (4 * da * db)

    H = np.array([
        [d2_da2, d2_dadb],
        [d2_dadb, d2_db2]
    ])

    return H


def analyze(path, split="train_Z"):
    data = np.load(path)

    xs = data["xs"]
    ys = data["ys"]
    Z = data[split]

    H = local_2d_hessian(
        xs,
        ys,
        Z
    )

    eigvals, eigvecs = np.linalg.eigh(H)

    eigvals = np.sort(eigvals)

    lambda_min = eigvals[0]
    lambda_max = eigvals[-1]

    anisotropy = (
        lambda_max / lambda_min
        if lambda_min > 1e-12
        else np.inf
    )

    center_i = np.argmin(np.abs(ys))
    center_j = np.argmin(np.abs(xs))

    center_loss = Z[
        center_i,
        center_j
    ]

    print("\nFile:", path)
    print("Split:", split)

    print("\nCenter loss:")
    print(center_loss)

    print("\nLocal Hessian:")
    print(H)

    print("\nEigenvalues:")
    print(eigvals)

    print("\nLargest curvature:")
    print(lambda_max)

    print("\nSmallest curvature:")
    print(lambda_min)

    print("\nAnisotropy ratio:")
    print(anisotropy)

    return {
        "H": H,
        "eigenvalues": eigvals,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "anisotropy": anisotropy,
        "center_loss": center_loss,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("PLAIN SGD")
    print("=" * 60)

    sgd = analyze(
        "results/2d_sgd_no_momentum/epoch_300.npz",
        split="train_Z"
    )

    print("\n" + "=" * 60)
    print("SGD + NESTEROV")
    print("=" * 60)

    nesterov = analyze(
        "results/2d_sgd_nesterov/epoch_300.npz",
        split="train_Z"
    )

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"SGD lambda_max:       "
        f"{sgd['lambda_max']:.4f}"
    )

    print(
        f"Nesterov lambda_max:  "
        f"{nesterov['lambda_max']:.4f}"
    )

    print(
        f"SGD anisotropy:       "
        f"{sgd['anisotropy']:.4f}"
    )

    print(
        f"Nesterov anisotropy:  "
        f"{nesterov['anisotropy']:.4f}"
    )

    if (
        sgd["lambda_max"]
        >
        nesterov["lambda_max"]
    ):
        print(
            "\nPlain SGD is sharper "
            "along the sharpest direction "
            "in this 2D plane."
        )
    else:
        print(
            "\nSGD + Nesterov is sharper "
            "along the sharpest direction "
            "in this 2D plane."
        )
