# gradcam.py

from captum.attr import LayerGradCam, LayerAttribution
import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_gradcam(
    model,
    image,
    target,
    device,
    target_layer
):
    """
    Compute Grad-CAM for one CIFAR-10 image.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network.

    image : torch.Tensor
        Normalized image of shape [3, 32, 32].

    target : int or torch.Tensor
        Class whose prediction should be explained.

    device : torch.device
        Device containing the model (mps/cuda/cpu).

    target_layer : torch.nn.Module
        Convolutional layer whose activations will be
        used to compute Grad-CAM.

    Returns
    -------
    heatmap : torch.Tensor
        Grad-CAM heatmap of shape [32, 32].

    target : int
        Class that was explained.
    """

    model.eval()

    # ---------------------------------
    # 1. Prepare input
    # ---------------------------------

    input_batch = image.unsqueeze(0).to(device)
    # [1, 3, 32, 32]

    if isinstance(target, torch.Tensor):
        target = int(target.item())
    else:
        target = int(target)

    # ---------------------------------
    # 2. Create Grad-CAM object
    # ---------------------------------

    gradcam = LayerGradCam(
        model,
        target_layer
    )

    # ---------------------------------
    # 3. Compute attribution
    # ---------------------------------

    attributions = gradcam.attribute(
        input_batch,
        target=target
    )

    # The convolutional feature map is
    # smaller than the original image.
    #
    # Upsample Grad-CAM to 32 x 32.

    attributions = LayerAttribution.interpolate(
        attributions,
        input_batch.shape[2:]
    )

    # ---------------------------------
    # 4. Convert to 2D heatmap
    # ---------------------------------

    heatmap = (
        attributions
        .squeeze()
        .detach()
        .cpu()
    )

    # Captum usually already returns a
    # single-channel Grad-CAM map.
    #
    # This is just a safety check.

    if heatmap.ndim == 3:
        heatmap = heatmap.mean(dim=0)

    # ---------------------------------
    # 5. Standard Grad-CAM ReLU
    # ---------------------------------

    # Keep positive evidence for target class.

    heatmap = torch.relu(heatmap)

    # Normalize to [0, 1] for visualization.

    heatmap = heatmap / (
        heatmap.max() + 1e-8
    )

    return heatmap, target


def visualize_gradcam(
    model,
    image,
    label,
    device,
    target_layer,
    explain_prediction=True
):
    """
    Compute and visualize Grad-CAM.

    By default, explains the MODEL'S PREDICTED class.

    If explain_prediction=False, explains the
    ground-truth class instead.
    """

    model.eval()

    # ---------------------------------
    # 1. Determine prediction
    # ---------------------------------

    input_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(input_batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = int(
            logits.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[0, prediction].item()
        )

    # Ground-truth class

    if isinstance(label, torch.Tensor):
        ground_truth = int(label.item())
    else:
        ground_truth = int(label)

    # ---------------------------------
    # 2. Decide what class to explain
    # ---------------------------------

    if explain_prediction:
        target = prediction
    else:
        target = ground_truth

    # ---------------------------------
    # 3. Compute Grad-CAM
    # ---------------------------------

    heatmap, target = compute_gradcam(
        model=model,
        image=image,
        target=target,
        device=device,
        target_layer=target_layer
    )

    # ---------------------------------
    # 4. Unnormalize CIFAR-10 image
    # ---------------------------------

    # These MUST match the normalization
    # used during training.

    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616]
    ).view(3, 1, 1)

    display_img = (
        image.detach().cpu() * std
        + mean
    )

    display_img = display_img.clamp(
        0,
        1
    )

    # CHW -> HWC

    display_img = (
        display_img
        .permute(1, 2, 0)
        .numpy()
    )

    heatmap_np = heatmap.numpy()

    # ---------------------------------
    # 5. Visualization
    # ---------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # Original image

    axes[0].imshow(display_img)

    axes[0].set_title(
        f"Input\n"
        f"GT={ground_truth}, "
        f"Pred={prediction}"
    )

    axes[0].axis("off")

    # Grad-CAM alone

    im = axes[1].imshow(
        heatmap_np,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"Grad-CAM\n"
        f"Target={target}"
    )

    axes[1].axis("off")

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046
    )

    # Overlay

    axes[2].imshow(
        display_img
    )

    axes[2].imshow(
        heatmap_np,
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        f"Overlay\n"
        f"confidence={confidence:.3f}"
    )

    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # ---------------------------------
    # 6. Return data for analysis
    # ---------------------------------

    return {
        "heatmap": heatmap,
        "target": target,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "confidence": confidence
    }
