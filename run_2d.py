# run_2d_multi.py

import os
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resnet56_ll import (
    setup,
    get_params_vector,
    set_params_vector,
    random_direction_filterwise,
    bn_reestimate,
    eval_loss,
)

import copy


def compute_2d_loss_landscape(
    model,
    bn_loader,
    eval_loader,
    criterion,
    device,
    d1,
    d2,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    steps=21,
    bn_batches=5,
    eval_batches=10,
):
    """
    Compute a 2D loss landscape:

        L(w_center + alpha*d1 + beta*d2)

    Returns
    -------
    xs : np.ndarray
    ys : np.ndarray
    Z  : np.ndarray
        Shape [steps, steps]
    """

    model.eval()

    # Save full checkpoint state including BatchNorm buffers
    original_state = copy.deepcopy(model.state_dict())

    # Center of this surface = current checkpoint
    w_center = get_params_vector(model).clone()

    xs = np.linspace(
        x_range[0],
        x_range[1],
        steps
    )

    ys = np.linspace(
        y_range[0],
        y_range[1],
        steps
    )

    Z = np.zeros(
        (steps, steps),
        dtype=np.float32
    )

    total_points = steps * steps
    point_counter = 0

    for i, beta in enumerate(ys):

        for j, alpha in enumerate(xs):

            point_counter += 1

            # Move away from checkpoint in 2 directions
            w = (
                w_center
                + alpha * d1
                + beta * d2
            )

            set_params_vector(
                model,
                w
            )

            # Re-estimate BN statistics from training data
            bn_reestimate(
                model,
                bn_loader,
                device,
                num_batches=bn_batches
            )

            # Evaluate loss on desired split
            Z[i, j] = eval_loss(
                model,
                eval_loader,
                criterion,
                device,
                max_batches=eval_batches
            )

            print(
                f"[{point_counter:04d}/{total_points}] "
                f"alpha={alpha:+.2f} "
                f"beta={beta:+.2f} "
                f"loss={Z[i, j]:.4f}"
            )

        print(
            f"Completed row {i + 1}/{steps}"
        )

    # Restore exact checkpoint
    model.load_state_dict(
        original_state,
        strict=True
    )

    model.eval()

    print("Original model state restored.")

    return xs, ys, Z


def save_2d_plot(
    xs,
    ys,
    Z,
    save_path,
    title,
    use_log=False,
):
    """
    Save 3D surface plot without opening GUI window.
    """

    X, Y = np.meshgrid(
        xs,
        ys
    )

    if use_log:
        Z_plot = np.log(
            Z + 1e-8
        )
        zlabel = "log(loss)"
    else:
        Z_plot = Z
        zlabel = "Cross-Entropy Loss"

    fig = plt.figure(
        figsize=(8, 6)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    surf = ax.plot_surface(
        X,
        Y,
        Z_plot,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # Mark checkpoint center at alpha=0, beta=0
    center_i = np.argmin(
        np.abs(ys)
    )

    center_j = np.argmin(
        np.abs(xs)
    )

    center_z = Z_plot[
        center_i,
        center_j
    ]

    ax.scatter(
        [0],
        [0],
        [center_z],
        s=70,
        marker="o"
    )

    ax.set_xlabel(
        "alpha (direction 1)"
    )

    ax.set_ylabel(
        "beta (direction 2)"
    )

    ax.set_zlabel(
        zlabel
    )

    ax.set_title(
        title
    )

    ax.view_init(
        elev=25,
        azim=-60
    )

    fig.colorbar(
        surf,
        shrink=0.6,
        aspect=12
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved plot: {save_path}"
    )


def main():

    # ==================================================
    # 1. Checkpoints to compare
    # ==================================================

    epochs = [
        150,
        300,
    ]

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # 2. Output folders
    # ==================================================

    result_dir = (
        "results/2d_sgd_nesterov"
    )

    plot_dir = (
        "plots/2d_sgd_nesterov"
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    # ==================================================
    # 3. Load epoch 300 reference model
    # ==================================================

    reference_ckpt = (
        "training/checkpoints/sgd_nesterov/run2/"
        "resnet56_epoch_300.pth"
    )

    print("\n" + "=" * 60)
    print(
        "LOADING REFERENCE CHECKPOINT "
        "FOR 2D DIRECTIONS"
    )
    print("=" * 60)

    (
        reference_model,
        _,
        _,
        _,
        _,
        reference_device,
    ) = setup(
        reference_ckpt
    )

    # ==================================================
    # 4. Create shared directions
    # ==================================================

    d1 = random_direction_filterwise(
        reference_model,
        reference_device
    )

    d2 = random_direction_filterwise(
        reference_model,
        reference_device
    )

    # ----------------------------------------------
    # Orthogonalize d2 against d1
    # ----------------------------------------------

    projection = (
        torch.dot(d2, d1)
        /
        (
            torch.dot(d1, d1)
            + 1e-12
        )
    )

    d2 = (
        d2
        - projection * d1
    )

    # Optional: normalize d2 scale relative to d1
    # so orthogonalization doesn't shrink it too much

    d2 = (
        d2
        * (
            d1.norm()
            /
            (d2.norm() + 1e-12)
        )
    )

    print(
        "Created shared directions d1 and d2."
    )

    print(
        "dot(d1, d2) =",
        float(
            torch.dot(d1, d2)
            .detach()
            .cpu()
        )
    )

    # ==================================================
    # 5. Save directions
    # ==================================================

    np.save(
        os.path.join(
            result_dir,
            "shared_direction_d1_epoch300.npy"
        ),
        d1.detach().cpu().numpy()
    )

    np.save(
        os.path.join(
            result_dir,
            "shared_direction_d2_epoch300.npy"
        ),
        d2.detach().cpu().numpy()
    )

    print(
        "Saved shared 2D directions."
    )

    # ==================================================
    # 6. Loop through checkpoints
    # ==================================================

    for epoch in epochs:

        print(
            "\n" + "=" * 60
        )

        print(
            f"PROCESSING EPOCH {epoch}"
        )

        print(
            "=" * 60
        )

        ckpt_path = (
            "training/checkpoints/sgd_nesterov/run2/"
            f"resnet56_epoch_{epoch}.pth"
        )

        if not os.path.exists(
            ckpt_path
        ):
            print(
                f"Checkpoint missing: "
                f"{ckpt_path}"
            )
            continue

        (
            model,
            trainset,
            testset,
            trainloader,
            testloader,
            device,
        ) = setup(
            ckpt_path
        )

        # Move directions to correct device
        d1_device = d1.to(device)
        d2_device = d2.to(device)

        # ==================================================
        # 7. TRAIN 2D landscape
        # ==================================================

        print(
            "\nComputing TRAIN 2D landscape..."
        )

        train_xs, train_ys, train_Z = (
            compute_2d_loss_landscape(
                model=model,

                bn_loader=trainloader,
                eval_loader=trainloader,

                criterion=criterion,
                device=device,

                d1=d1_device,
                d2=d2_device,

                x_range=(-1.0, 1.0),
                y_range=(-1.0, 1.0),

                steps=21,

                bn_batches=5,
                eval_batches=10,
            )
        )

        train_plot_path = (
            os.path.join(
                plot_dir,
                f"epoch_{epoch:03d}_train_2d.png"
            )
        )

        save_2d_plot(
            xs=train_xs,
            ys=train_ys,
            Z=train_Z,
            save_path=train_plot_path,
            title=(
                f"2D Train Loss Landscape — Epoch {epoch}\n"
                f"ResNet-56 No-Skip — SGD + Nesterov"
            ),
            use_log=False,
        )

        # ==================================================
        # 8. TEST 2D landscape
        # ==================================================

        print(
            "\nComputing TEST 2D landscape..."
        )

        test_xs, test_ys, test_Z = (
            compute_2d_loss_landscape(
                model=model,

                # BN still estimated from training data
                bn_loader=trainloader,

                # loss measured on test data
                eval_loader=testloader,

                criterion=criterion,
                device=device,

                d1=d1_device,
                d2=d2_device,

                x_range=(-1.0, 1.0),
                y_range=(-1.0, 1.0),

                steps=21,

                bn_batches=5,
                eval_batches=10,
            )
        )

        test_plot_path = (
            os.path.join(
                plot_dir,
                f"epoch_{epoch:03d}_test_2d.png"
            )
        )

        save_2d_plot(
            xs=test_xs,
            ys=test_ys,
            Z=test_Z,
            save_path=test_plot_path,
            title=(
                f"2D Test Loss Landscape — Epoch {epoch}\n"
                f"ResNet-56 No-Skip — SGD + Nesterov"
            ),
            use_log=False,
        )

        # ==================================================
        # 9. Save raw data
        # ==================================================

        result_path = (
            os.path.join(
                result_dir,
                f"epoch_{epoch:03d}.npz"
            )
        )

        np.savez(
            result_path,

            xs=train_xs,
            ys=train_ys,

            train_Z=train_Z,
            test_Z=test_Z,

            epoch=epoch,
        )

        print(
            f"Saved numerical data: "
            f"{result_path}"
        )

        # ==================================================
        # 10. Print checkpoint summary
        # ==================================================

        train_center = (
            train_Z[
                np.argmin(
                    np.abs(train_ys)
                ),
                np.argmin(
                    np.abs(train_xs)
                )
            ]
        )

        test_center = (
            test_Z[
                np.argmin(
                    np.abs(test_ys)
                ),
                np.argmin(
                    np.abs(test_xs)
                )
            ]
        )

        print(
            "\nCheckpoint summary:"
        )

        print(
            f"Train center loss: "
            f"{train_center:.4f}"
        )

        print(
            f"Test center loss:  "
            f"{test_center:.4f}"
        )

        print(
            f"Train surface min/max: "
            f"{train_Z.min():.4f} / "
            f"{train_Z.max():.4f}"
        )

        print(
            f"Test surface min/max: "
            f"{test_Z.min():.4f} / "
            f"{test_Z.max():.4f}"
        )

    # ==================================================
    # 11. Done
    # ==================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL 2D LANDSCAPES COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Results: {result_dir}"
    )

    print(
        f"Plots:   {plot_dir}"
    )


if __name__ == "__main__":
    main()
