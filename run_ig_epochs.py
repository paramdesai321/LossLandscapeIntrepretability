import os
import numpy as np
import torch

from resnet56_ll import setup
from integrated_gradients import visualize_ig


def main():

    # ==================================================
    # Configuration
    # ==================================================

    epochs = [
        150,
        300,
    ]

    checkpoint_dir = (
        "training/checkpoints/sgd_nesterov/run2/"
    )

    output_dir = (
        "plots/integrated_gradients/"
        "sgd_nesterov"
    )

    result_dir = (
        "results/integrated_gradients/"
        "sgd_nesterov"
    )

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
        print(f"INTEGRATED GRADIENTS — EPOCH {epoch}")
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
                f"image_{idx:05d}_ig.png"
            )

            result = visualize_ig(
                model=model,
                image=image,
                label=label,
                device=device,

                # Can increase later
                n_steps=50,

                use_abs=True,

                save_path=save_path,

                title=(
                    f"Integrated Gradients — Epoch {epoch}\n"
                    f"Test image {idx}"
                ),
            )

            # ------------------------------------------
            # Save raw attribution data too
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

                delta=(
                    result["delta"]
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
    print("INTEGRATED GRADIENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
