from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_integrated_gradients(
    model,
    image,
    label,
    device,
    n_steps=50
):
    model.eval()

    image = image.unsqueeze(0).to(device)

    if isinstance(label, torch.Tensor):
        target = int(label.item())
    else:
        target = int(label)

    baseline = torch.zeros_like(image)

    ig = IntegratedGradients(model)

    attributions, delta = ig.attribute(
        image,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    return attributions, delta

def visualize_ig(
    model,
    image,
    label,
    device,
    n_steps=50,
    use_abs=True
):
    attributions, delta = compute_integrated_gradients(
        model=model,
        image=image,
        label=label,
        device=device,
        n_steps=n_steps
    )

    if isinstance(label, torch.Tensor):
        target = int(label.item())
    else:
        target = int(label)

    # Attribution: [1, 3, 32, 32]
    attr = attributions.squeeze(0).detach().cpu()

    if use_abs:
        heatmap = attr.abs().sum(dim=0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    else:
        heatmap = attr.sum(dim=0)
        max_abs = heatmap.abs().max() + 1e-8
        heatmap = heatmap / max_abs

    heatmap = heatmap.numpy()

    # Unnormalize CIFAR-10 image
    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616]
    ).view(3, 1, 1)

    display_img = (
        image.detach().cpu() * std + mean
    ).clamp(0, 1)

    display_img = display_img.permute(
        1, 2, 0
    ).numpy()

    # Prediction
    model.eval()

    with torch.no_grad():
        input_batch = image.unsqueeze(0).to(device)

        logits = model(input_batch)

        prediction = int(
            logits.argmax(dim=1).item()
        )

        confidence = float(
            torch.softmax(logits, dim=1)[0, prediction].item()
        )

    # Plot
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(display_img)
    axes[0].set_title(
        f"Input\nGT={target}, Pred={prediction}"
    )
    axes[0].axis("off")

    if use_abs:
        im = axes[1].imshow(
            heatmap,
            cmap="hot"
        )
    else:
        im = axes[1].imshow(
            heatmap,
            cmap="seismic",
            vmin=-1,
            vmax=1
        )

    axes[1].set_title(
        "Integrated Gradients"
    )
    axes[1].axis("off")

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046
    )

    axes[2].imshow(display_img)
    axes[2].imshow(
        heatmap,
        cmap="hot" if use_abs else "seismic",
        alpha=0.5
    )

    axes[2].set_title(
        f"IG Overlay\nconfidence={confidence:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(
        "IG convergence delta:",
        float(delta.detach().cpu().item())
    )

    return {
        "attributions": attributions.detach().cpu(),
        "heatmap": heatmap,
        "delta": delta.detach().cpu(),
        "target": target,
        "prediction": prediction,
        "confidence": confidence,
    }
