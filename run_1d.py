import torch.nn as nn

from resnet56_ll import (
    setup,
    plot_1d_loss_landscape,
)


def main():

    # --------------------------------
    # Load checkpoint + CIFAR-10
    # --------------------------------

    (
        model,
        trainset,
        testset,
        trainloader,
        testloader,
        device,
    ) = setup()

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # 1. TRAIN LOSS LANDSCAPE
    # ==================================================

    print("\n==============================")
    print("Computing TRAIN loss landscape")
    print("==============================\n")

#    train_xs, train_losses, direction = (
#        plot_1d_loss_landscape(
#            model=model,
#
#            # Always use training data for BN
#            bn_loader=trainloader,
#
#            # Measure TRAIN loss
#            eval_loader=trainloader,
#
#            criterion=criterion,
#            device=device,
#
#            direction=None,  # generate direction here
#
#            xmin=-1.0,
#            xmax=1.0,
#            steps=51,
#
#            bn_batches=20,
#            eval_batches=None,
#        )
#    )
#
#    print("\nTRAIN landscape complete.")
#    print("Minimum train loss:", train_losses.min())
#    print(
#        "Minimum train-loss alpha:",
#        train_xs[train_losses.argmin()],
#    )

    # ==================================================
    # 2. TEST LOSS LANDSCAPE
    # ==================================================

    print("\n=============================")
    print("Computing TEST loss landscape")
    print("=============================\n")

    test_xs, test_losses, _ = (
        plot_1d_loss_landscape(
            model=model,

            # IMPORTANT:
            # BN statistics still come from TRAINING data
            bn_loader=trainloader,

            # But loss is measured on TEST data
            eval_loader=testloader,

            criterion=criterion,
            device=device,

            # VERY IMPORTANT:
            # reuse the SAME direction
            direction=direction,

            xmin=-1.0,
            xmax=1.0,
            steps=51,

            bn_batches=20,
            eval_batches=None,
        )
    )

    print("\nTEST landscape complete.")
    print("Minimum test loss:", test_losses.min())
    print(
        "Minimum test-loss alpha:",
        test_xs[test_losses.argmin()],
    )

    # ==================================================
    # 3. Summary
    # ==================================================

    print("\n=============================")
    print("SUMMARY")
    print("=============================")

    print(
        f"Train minimum: "
        f"{train_losses.min():.4f} "
        f"at alpha="
        f"{train_xs[train_losses.argmin()]:+.3f}"
    )

    print(
        f"Test minimum:  "
        f"{test_losses.min():.4f} "
        f"at alpha="
        f"{test_xs[test_losses.argmin()]:+.3f}"
    )


if __name__ == "__main__":
    main()
