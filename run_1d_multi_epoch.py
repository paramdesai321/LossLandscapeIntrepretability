import os
import numpy as np
import torch.nn as nn

from resnet56_ll import (
    setup,
    plot_1d_loss_landscape,
    random_direction_filterwise,
)


def main():

    # ==================================================
    # 1. Checkpoints to inspect
    # ==================================================

    epochs = [
        5, 25, 50, 100,
        145, 150, 155,
        220, 225, 230,
        270, 275, 280,
        300,
    ]

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # 2. Output directories
    # ==================================================

    output_dir = "results/1d_sgd_nesterov"
    plot_dir = "plots/1d_sgd_nesterov"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    # ==================================================
    # 3. Build ONE shared direction from epoch 300
    # ==================================================

    reference_ckpt = (
        "training/checkpoints/sgd_nesterov/run2/"
        "resnet56_epoch_300.pth"
    )

    print("\n" + "=" * 60)
    print("LOADING REFERENCE CHECKPOINT")
    print("=" * 60)
    print(f"Reference: {reference_ckpt}")

    (
        reference_model,
        _,
        _,
        _,
        _,
        reference_device,
    ) = setup(reference_ckpt)

    shared_direction = random_direction_filterwise(
        reference_model,
        reference_device
    )

    print("Shared direction created from epoch 300.")

    # Save the exact direction used in the experiment
    shared_direction_path = os.path.join(
        output_dir,
        "shared_direction_epoch300.npy"
    )

    np.save(
        shared_direction_path,
        shared_direction.detach().cpu().numpy()
    )

    print(
        f"Saved shared direction: "
        f"{shared_direction_path}"
    )

    # ==================================================
    # 4. Loop through checkpoints
    # ==================================================

    for epoch in epochs:

        print("\n" + "=" * 60)
        print(f"PROCESSING EPOCH {epoch}")
        print("=" * 60)

        ckpt_path = (
            f"training/checkpoints/sgd_nesterov/run2/"
            f"resnet56_epoch_{epoch}.pth"
        )

        # Skip missing checkpoints safely
        if not os.path.exists(ckpt_path):
            print(
                f"WARNING: checkpoint not found: "
                f"{ckpt_path}"
            )
            print("Skipping this epoch.")
            continue

        # ----------------------------------------------
        # Load this epoch
        # ----------------------------------------------

        (
            model,
            trainset,
            testset,
            trainloader,
            testloader,
            device,
        ) = setup(ckpt_path)

        # Move the shared direction to this device
        direction = shared_direction.to(device)

        # ==================================================
        # 5. TRAIN loss landscape
        # ==================================================

        print("\n" + "-" * 50)
        print(f"EPOCH {epoch}: TRAIN LANDSCAPE")
        print("-" * 50)

        train_plot_path = os.path.join(
            plot_dir,
            f"epoch_{epoch:03d}_train.png"
        )

        (
            xs,
            train_losses,
            _
        ) = plot_1d_loss_landscape(
            model=model,

            # BatchNorm always estimated using train data
            bn_loader=trainloader,

            # Evaluate training loss
            eval_loader=trainloader,

            criterion=criterion,
            device=device,

            # Same direction across every checkpoint
            direction=direction,

            xmin=-1.0,
            xmax=1.0,
            steps=21,

            bn_batches=5,
            eval_batches=10,

            # Save PNG instead of displaying
            save_path=train_plot_path,

            title=(
                f"1D Train Loss Landscape — Epoch {epoch}\n"
                f"ResNet-56 No-Skip — SGD + Nesterov"
            ),
        )

        print(
            f"Train plot saved: "
            f"{train_plot_path}"
        )

        # ==================================================
        # 6. TEST loss landscape
        # ==================================================

        print("\n" + "-" * 50)
        print(f"EPOCH {epoch}: TEST LANDSCAPE")
        print("-" * 50)

        test_plot_path = os.path.join(
            plot_dir,
            f"epoch_{epoch:03d}_test.png"
        )

        (
            test_xs,
            test_losses,
            _
        ) = plot_1d_loss_landscape(
            model=model,

            # IMPORTANT:
            # BatchNorm statistics still from training data
            bn_loader=trainloader,

            # Evaluate test loss
            eval_loader=testloader,

            criterion=criterion,
            device=device,

            # Exact same direction as training landscape
            direction=direction,

            xmin=-1.0,
            xmax=1.0,
            steps=21,

            bn_batches=5,
            eval_batches=10,

            # Save PNG instead of displaying
            save_path=test_plot_path,

            title=(
                f"1D Test Loss Landscape — Epoch {epoch}\n"
                f"ResNet-56 No-Skip — SGD + Nesterov"
            ),
        )

        print(
            f"Test plot saved: "
            f"{test_plot_path}"
        )

        # ==================================================
        # 7. Save numerical results
        # ==================================================

        result_path = os.path.join(
            output_dir,
            f"epoch_{epoch:03d}.npz"
        )

        np.savez(
            result_path,
            xs=xs,
            test_xs=test_xs,
            train_losses=train_losses,
            test_losses=test_losses,
            epoch=epoch,
        )

        print(
            f"Numerical data saved: "
            f"{result_path}"
        )

        # ==================================================
        # 8. Print summary for this checkpoint
        # ==================================================

        train_min_index = train_losses.argmin()
        test_min_index = test_losses.argmin()

        print("\nEpoch summary:")

        print(
            f"Train minimum loss = "
            f"{train_losses[train_min_index]:.4f} "
            f"at alpha={xs[train_min_index]:+.3f}"
        )

        print(
            f"Test minimum loss  = "
            f"{test_losses[test_min_index]:.4f} "
            f"at alpha={test_xs[test_min_index]:+.3f}"
        )

        print(
            f"Train loss at checkpoint alpha=0: "
            f"{train_losses[np.argmin(np.abs(xs))]:.4f}"
        )

        print(
            f"Test loss at checkpoint alpha=0:  "
            f"{test_losses[np.argmin(np.abs(test_xs))]:.4f}"
        )

    # ==================================================
    # 9. Complete
    # ==================================================

    print("\n" + "=" * 60)
    print("ALL 1D CHECKPOINT LANDSCAPES COMPLETE")
    print("=" * 60)

    print(f"\nNumerical results: {output_dir}")
    print(f"PNG plots:         {plot_dir}")


if __name__ == "__main__":
    main()
