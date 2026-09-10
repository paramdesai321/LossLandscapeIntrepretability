#!/usr/bin/env python3
"""Generate standalone Grad-CAM visualizations at selected checkpoints.

This follows the same epoch/image loop as ``run_ig_epochs.py``.  Each plot
contains the input image, the Grad-CAM heatmap, and the heatmap overlay.
"""

import os

import numpy as np

from resnet56_ll import setup
from grad_cam import visualize_grad_cam


def main():

    # ==================================================
    # Configuration
    # ==================================================

    epochs = [
        150,
        300,
    ]

    checkpoint_dir = (
        "training/checkpoints/sgd_no_momentum/"
    )

    output_dir = (
        "plots/grad_cam/"
        "sgd_no_momentum"
    )

    result_dir = (
        "results/grad_cam/"
        "sgd_no_momentum"
    )

    # None automatically selects the final Conv2d layer.  For an explicit
    # layer, use its dotted module name, for example "stage3.8.conv2".
    target_layer = None

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # SAME images for every epoch
    # --------------------------------------------------

    test_indices = [
        0,
        1,
        2,
        3,
        4,
    ]

    # ==================================================
    # Loop over checkpoints
    # ==================================================

    for epoch in epochs:

        print("\n" + "=" * 60)
        print(f"GRAD-CAM — EPOCH {epoch}")
        print("=" * 60)

        ckpt_path = (
            f"{checkpoint_dir}/"
            f"resnet56_epoch_{epoch}.pth"
        )

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

        epoch_output_dir = os.path.join(
            output_dir,
            f"epoch_{epoch:03d}"
        )

        os.makedirs(
            epoch_output_dir,
            exist_ok=True
        )

        # ==============================================
        # Same test images at every checkpoint
        # ==============================================

        for idx in test_indices:

            image, label = testset[idx]
            label_int = int(label)

            print(
                f"\nEpoch {epoch} | "
                f"test index {idx} | "
                f"target={label_int}"
            )

            save_path = os.path.join(
                epoch_output_dir,
                f"image_{idx:05d}_grad_cam.png"
            )

            result = visualize_grad_cam(
                model=model,
                image=image,
                label=label,
                device=device,
                target_layer=target_layer,

                # Accepted for API compatibility; Grad-CAM does not use it.
                n_steps=50,

                save_path=save_path,

                title=(
                    f"Grad-CAM — Epoch {epoch}\n"
                    f"Test image {idx}"
                ),
            )

            # ------------------------------------------
            # Save raw Grad-CAM data too
            # ------------------------------------------

            result_path = os.path.join(
                result_dir,
                (
                    f"epoch_{epoch:03d}_"
                    f"image_{idx:05d}.npz"
                )
            )

            np.savez(
                result_path,

                heatmap=result["heatmap"],

                attributions=(
                    result["attributions"]
                    .numpy()
                ),

                target=result["target"],

                prediction=result["prediction"],

                confidence=result["confidence"],

                epoch=epoch,

                image_index=idx,
            )

            print(
                f"Saved plot: {save_path}"
            )

            print(
                f"Saved data: {result_path}"
            )

    print("\n" + "=" * 60)
    print("GRAD-CAM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
