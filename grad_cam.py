"""Grad-CAM utilities with an API parallel to ``integrated_gradients.py``.

The public functions accept the same core arguments as the integrated
gradients module: a model, one image in ``[C, H, W]`` format, a target label,
and a device.  The last ``nn.Conv2d`` layer is used by default, or a layer can
be supplied explicitly.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


LayerSpec = Optional[Union[str, nn.Module]]


def _label_to_int(label) -> int:
    """Convert a scalar label/tensor to a Python integer."""

    if isinstance(label, torch.Tensor):
        return int(label.detach().item())
    return int(label)


def _get_logits(output):
    """Extract logits from the common model-output containers."""

    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "output", "predictions"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    return output


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last ``Conv2d`` module in ``model``."""

    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError(
            "Grad-CAM requires at least one torch.nn.Conv2d layer. "
            "Pass an appropriate target_layer if your model uses a custom layer."
        )
    return conv_layers[-1]


def _resolve_target_layer(model: nn.Module, target_layer: LayerSpec) -> nn.Module:
    """Resolve a module object or dotted module name, defaulting to last Conv2d."""

    if target_layer is None:
        return get_last_conv_layer(model)

    if isinstance(target_layer, nn.Module):
        return target_layer

    if isinstance(target_layer, str):
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(
                f"Could not find target layer {target_layer!r}. "
                "Use a dotted module name from model.named_modules()."
            )
        return modules[target_layer]

    raise TypeError("target_layer must be None, a module, or a module name.")


def _prepare_image(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert one ``[C,H,W]`` image to a model-ready ``[1,C,H,W]`` tensor."""

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [C,H,W] or a single-image shape [1,C,H,W].")

    return image.to(device)


def compute_grad_cam(
    model,
    image,
    label,
    device,
    target_layer=None,
    n_steps=50,
):
    """Compute a Grad-CAM map for one image.

    ``n_steps`` is accepted for call-site compatibility with the integrated
    gradients implementation, but is not used by Grad-CAM.

    Returns:
        A tuple ``(heatmap, prediction)`` where ``heatmap`` is a normalized
        tensor of shape ``[H, W]`` on CPU and ``prediction`` is the model's
        predicted class index.
    """

    del n_steps  # API compatibility with compute_integrated_gradients.

    model.eval()
    image_batch = _prepare_image(image, device)
    target = _label_to_int(label)
    layer = _resolve_target_layer(model, target_layer)

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        output = output[0] if isinstance(output, (tuple, list)) else output
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradient = grad_outputs[0]
        gradients.append(gradient)

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = _get_logits(model(image_batch))

        if logits.ndim != 2:
            raise ValueError(
                "Expected model output shaped [batch, classes] for Grad-CAM; "
                f"got {tuple(logits.shape)}."
            )
        if target < 0 or target >= logits.shape[1]:
            raise ValueError(
                f"Target label {target} is outside the model's class range "
                f"[0, {logits.shape[1] - 1}]."
            )

        logits[0, target].backward()

        if not activations or not gradients:
            raise RuntimeError("The target layer did not produce activations and gradients.")

        activation = activations[-1]
        gradient = gradients[-1]
        if activation.ndim != 4 or gradient.ndim != 4:
            raise ValueError(
                "The target layer must produce spatial feature maps shaped "
                "[batch, channels, height, width]."
            )

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(
            cam,
            size=image_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)
        cam = cam / (cam.max() + 1e-8)

        prediction = int(logits.detach().argmax(dim=1).item())
        return cam.detach().cpu(), prediction
    finally:
        forward_handle.remove()
        backward_handle.remove()


def visualize_grad_cam(
    model,
    image,
    label,
    device,
    n_steps=50,
    use_abs=True,
    save_path=None,
    title=None,
    target_layer=None,
):
    """Compute and plot Grad-CAM using the visualization structure of IG.

    ``n_steps`` and ``use_abs`` are retained for drop-in call compatibility.
    Grad-CAM is already nonnegative after its ReLU, so ``use_abs`` does not
    change the heatmap.
    """

    del use_abs  # Grad-CAM applies ReLU by definition.

    target = _label_to_int(label)
    heatmap_tensor, prediction = compute_grad_cam(
        model=model,
        image=image,
        label=label,
        device=device,
        target_layer=target_layer,
        n_steps=n_steps,
    )
    heatmap = heatmap_tensor.numpy()

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image.squeeze(0)

    # Unnormalize the CIFAR-10 image, matching integrated_gradients.py.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    display_img = (image * std + mean).clamp(0, 1)
    display_img = display_img.permute(1, 2, 0).numpy()

    model.eval()
    with torch.no_grad():
        input_batch = _prepare_image(image, device)
        logits = _get_logits(model(input_batch))
        confidence = float(torch.softmax(logits, dim=1)[0, prediction].item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(display_img)
    axes[0].set_title(f"Input\nGT={target}, Pred={prediction}")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(display_img)
    axes[2].imshow(heatmap, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(f"Grad-CAM Overlay\nconfidence={confidence:.3f}")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        # Keep the same keys used by visualize_ig where possible.
        "attributions": heatmap_tensor.unsqueeze(0).detach().cpu(),
        "heatmap": heatmap,
        "delta": None,
        "target": target,
        "prediction": prediction,
        "confidence": confidence,
    }


# Compatibility aliases make this module usable from callers that follow the
# naming convention of the integrated_gradients.py file.
compute_gradcam = compute_grad_cam
visualize_gradcam = visualize_grad_cam


def compute_integrated_gradients(
    model,
    image,
    label,
    device,
    n_steps=50,
    target_layer=None,
):
    """Compatibility wrapper for callers importing the old compute name.

    The positional ``n_steps`` argument is intentionally kept in the same
    position as in ``integrated_gradients.py``; Grad-CAM simply ignores it.
    """

    return compute_grad_cam(
        model=model,
        image=image,
        label=label,
        device=device,
        target_layer=target_layer,
        n_steps=n_steps,
    )


visualize_ig = visualize_grad_cam
