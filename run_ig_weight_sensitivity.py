#!/usr/bin/env python3
"""
run_ig_weight_sensitivity.py

Controlled experiment linking local 2D loss-landscape sharpness to
Integrated Gradients (IG) sensitivity under matched weight perturbations.

For each checkpoint:
  1. Load the exact d1/d2 directions used for the saved 2D landscape.
  2. Read local test-loss curvature (lambda_max) from the saved 2D NPZ.
  3. For the same fixed CIFAR-10 test images, compute IG at:
       theta
       theta + eps_alpha * d1
       theta - eps_alpha * d1
       theta + eps_beta  * d2
       theta - eps_beta  * d2
  4. Measure how much each perturbed attribution differs from the center.
  5. Save raw attributions, per-perturbation metrics, per-image summaries,
     epoch summaries, and comparison figures.

The default BN mode, "match_landscape", re-estimates BatchNorm statistics
at every weight-space point because the current 2D landscape code does so.
Set BN_MODE = "checkpoint" to keep checkpoint BN buffers fixed instead.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from integrated_gradients import compute_integrated_gradients
from resnet56_ll import (
    bn_reestimate,
    get_params_vector,
    set_params_vector,
    setup,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "sgd_nesterov"

EPOCHS = [150, 300]

CHECKPOINT_DIR = Path("training/checkpoints/sgd_nesterov/run2")

# This directory must contain:
#   shared_direction_d1_epoch300.npy
#   shared_direction_d2_epoch300.npy
#   epoch_150.npz
#   epoch_300.npz
#
# Change this if your 51x51 files were saved elsewhere.
LANDSCAPE_DIR = Path("results/2d_sgd_nesterov")

RESULT_DIR = Path(
    "results/ig_weight_sensitivity/sgd_nesterov"
)

PLOT_DIR = Path(
    "plots/ig_weight_sensitivity/sgd_nesterov"
)

# Use a balanced pilot: 2 images from each CIFAR-10 class = 20 images.
# Set EXPLICIT_TEST_INDICES to a list such as [0, 1, 2, 3, 4] to override.
EXPLICIT_TEST_INDICES: List[int] | None = None
IMAGES_PER_CLASS = 2

# IG numerical integration steps.
N_STEPS = 50

# Ground-truth targets make comparisons consistent even when predictions differ.
TARGET_MODE = "ground_truth"

# "match_landscape": re-estimate BN at center and each perturbation.
# "checkpoint": keep the checkpoint BN buffers fixed.
BN_MODE = "match_landscape"
BN_BATCHES = 5

# One landscape-grid step is used by default:
#   51x51 over [-1, 1] -> epsilon = 0.04.
# Set this to a float to override, e.g. EPSILON_OVERRIDE = 0.05.
EPSILON_OVERRIDE: float | None = None

# Save a six-panel visualization for every image/epoch.
SAVE_PERTURBATION_FIGURES = True

# Top fraction of pixels used for overlap.
TOP_K_FRACTION = 0.10

# Use test-loss geometry because IG is evaluated on test images.
LANDSCAPE_SPLIT = "test_Z"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}\n"
            "Update the configuration paths at the top of the script."
        )


def select_balanced_indices(
    dataset,
    images_per_class: int,
    num_classes: int = 10,
) -> List[int]:
    """Select the first N examples from each class, deterministically."""

    selected: List[int] = []
    counts = {class_id: 0 for class_id in range(num_classes)}

    for index in range(len(dataset)):
        _, label = dataset[index]
        class_id = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

        if counts[class_id] < images_per_class:
            selected.append(index)
            counts[class_id] += 1

        if all(count == images_per_class for count in counts.values()):
            break

    missing = {
        class_id: images_per_class - count
        for class_id, count in counts.items()
        if count < images_per_class
    }

    if missing:
        raise RuntimeError(
            f"Could not select a balanced image set. Missing counts: {missing}"
        )

    return selected


def safe_cosine_similarity(
    array_a: np.ndarray,
    array_b: np.ndarray,
) -> float:
    a = np.asarray(array_a, dtype=np.float64).reshape(-1)
    b = np.asarray(array_b, dtype=np.float64).reshape(-1)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-15:
        return float("nan")

    return float(np.dot(a, b) / denominator)


def safe_pearson_correlation(
    array_a: np.ndarray,
    array_b: np.ndarray,
) -> float:
    a = np.asarray(array_a, dtype=np.float64).reshape(-1)
    b = np.asarray(array_b, dtype=np.float64).reshape(-1)

    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def top_k_overlap(
    array_a: np.ndarray,
    array_b: np.ndarray,
    fraction: float,
) -> float:
    """Jaccard overlap between the top-k attribution pixels."""

    a = np.asarray(array_a).reshape(-1)
    b = np.asarray(array_b).reshape(-1)

    if a.size != b.size:
        raise ValueError("Attribution arrays must have the same number of pixels.")

    k = max(1, int(round(fraction * a.size)))

    top_a = set(np.argpartition(a, -k)[-k:].tolist())
    top_b = set(np.argpartition(b, -k)[-k:].tolist())

    union = top_a | top_b

    if not union:
        return float("nan")

    return float(len(top_a & top_b) / len(union))


def attribution_arrays(
    attribution: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      signed_raw: [3, H, W]
      magnitude_map: [H, W], sum of absolute RGB attributions
    """

    signed_raw = (
        attribution
        .squeeze(0)
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    magnitude_map = np.abs(signed_raw).sum(axis=0)

    return signed_raw, magnitude_map


def compare_attributions(
    center_attr: torch.Tensor,
    perturbed_attr: torch.Tensor,
) -> Dict[str, float]:
    center_signed, center_magnitude = attribution_arrays(center_attr)
    pert_signed, pert_magnitude = attribution_arrays(perturbed_attr)

    signed_cosine = safe_cosine_similarity(
        center_signed,
        pert_signed,
    )

    magnitude_cosine = safe_cosine_similarity(
        center_magnitude,
        pert_magnitude,
    )

    magnitude_pearson = safe_pearson_correlation(
        center_magnitude,
        pert_magnitude,
    )

    relative_l1 = float(
        np.abs(center_magnitude - pert_magnitude).sum()
        / (np.abs(center_magnitude).sum() + 1e-12)
    )

    return {
        "signed_cosine": signed_cosine,
        "signed_cosine_distance": 1.0 - signed_cosine,
        "magnitude_cosine": magnitude_cosine,
        "magnitude_cosine_distance": 1.0 - magnitude_cosine,
        "magnitude_pearson": magnitude_pearson,
        "relative_l1": relative_l1,
        "top_k_overlap": top_k_overlap(
            center_magnitude,
            pert_magnitude,
            TOP_K_FRACTION,
        ),
    }


def prediction_info(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: int,
    device: torch.device,
) -> Dict[str, float | int]:
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)

        prediction = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, prediction].item())
        target_probability = float(probabilities[0, target].item())

    return {
        "prediction": prediction,
        "confidence": confidence,
        "target_probability": target_probability,
    }


def convergence_delta_value(delta: torch.Tensor) -> float:
    return float(
        delta
        .detach()
        .cpu()
        .float()
        .reshape(-1)
        .mean()
        .item()
    )


def load_local_landscape_geometry(
    npz_path: Path,
    split: str,
) -> Dict[str, float | np.ndarray]:
    """
    Estimate the 2x2 local Hessian at alpha=beta=0 using the saved grid.
    """

    require_file(npz_path)

    data = np.load(npz_path)

    xs = np.asarray(data["xs"], dtype=np.float64)
    ys = np.asarray(data["ys"], dtype=np.float64)

    if split not in data:
        raise KeyError(
            f"{split!r} is missing from {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    Z = np.asarray(data[split], dtype=np.float64)

    i0 = int(np.argmin(np.abs(ys)))
    j0 = int(np.argmin(np.abs(xs)))

    if i0 == 0 or i0 == len(ys) - 1:
        raise ValueError("beta=0 needs neighbors on both sides.")

    if j0 == 0 or j0 == len(xs) - 1:
        raise ValueError("alpha=0 needs neighbors on both sides.")

    h_alpha = float(xs[j0 + 1] - xs[j0])
    h_beta = float(ys[i0 + 1] - ys[i0])

    d2_dalpha2 = (
        Z[i0, j0 + 1]
        - 2.0 * Z[i0, j0]
        + Z[i0, j0 - 1]
    ) / (h_alpha ** 2)

    d2_dbeta2 = (
        Z[i0 + 1, j0]
        - 2.0 * Z[i0, j0]
        + Z[i0 - 1, j0]
    ) / (h_beta ** 2)

    d2_dalphadbeta = (
        Z[i0 + 1, j0 + 1]
        - Z[i0 + 1, j0 - 1]
        - Z[i0 - 1, j0 + 1]
        + Z[i0 - 1, j0 - 1]
    ) / (4.0 * h_alpha * h_beta)

    hessian = np.array(
        [
            [d2_dalpha2, d2_dalphadbeta],
            [d2_dalphadbeta, d2_dbeta2],
        ],
        dtype=np.float64,
    )

    eigenvalues = np.linalg.eigvalsh(hessian)
    eigenvalues.sort()

    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])

    anisotropy = (
        float(lambda_max / lambda_min)
        if lambda_min > 1e-12
        else float("inf")
    )

    epsilon_alpha = (
        float(EPSILON_OVERRIDE)
        if EPSILON_OVERRIDE is not None
        else abs(h_alpha)
    )

    epsilon_beta = (
        float(EPSILON_OVERRIDE)
        if EPSILON_OVERRIDE is not None
        else abs(h_beta)
    )

    return {
        "xs": xs,
        "ys": ys,
        "Z": Z,
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "anisotropy": anisotropy,
        "center_loss": float(Z[i0, j0]),
        "epsilon_alpha": epsilon_alpha,
        "epsilon_beta": epsilon_beta,
    }


def prepare_model_state(
    model: torch.nn.Module,
    original_state: Dict[str, torch.Tensor],
    parameter_vector: torch.Tensor,
    train_loader,
    device: torch.device,
) -> None:
    """
    Restore the checkpoint, set a requested parameter vector, optionally
    re-estimate BN statistics, and switch to eval mode for IG.
    """

    model.load_state_dict(
        original_state,
        strict=True,
    )

    set_params_vector(
        model,
        parameter_vector,
    )

    if BN_MODE == "match_landscape":
        bn_reestimate(
            model,
            train_loader,
            device,
            num_batches=BN_BATCHES,
        )
    elif BN_MODE != "checkpoint":
        raise ValueError(
            "BN_MODE must be either 'match_landscape' or 'checkpoint'."
        )

    model.eval()


def unnormalize_cifar_image(
    image: torch.Tensor,
) -> np.ndarray:
    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465],
        dtype=image.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616],
        dtype=image.dtype,
    ).view(3, 1, 1)

    display_image = (
        image.detach().cpu() * std + mean
    ).clamp(0, 1)

    return (
        display_image
        .permute(1, 2, 0)
        .numpy()
    )


def save_perturbation_figure(
    image: torch.Tensor,
    magnitude_maps: Dict[str, np.ndarray],
    metric_rows: Dict[str, Dict[str, float]],
    epoch: int,
    image_index: int,
    target: int,
    center_prediction: int,
    lambda_max: float,
    save_path: Path,
) -> None:
    display_image = unnormalize_cifar_image(image)

    ordered_names = [
        "center",
        "plus_d1",
        "minus_d1",
        "plus_d2",
        "minus_d2",
    ]

    maximum = max(
        float(np.max(magnitude_maps[name]))
        for name in ordered_names
    )

    maximum = max(maximum, 1e-12)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
    )

    axes = axes.reshape(-1)

    axes[0].imshow(display_image)
    axes[0].set_title(
        f"Input\nGT={target}, Pred={center_prediction}"
    )
    axes[0].axis("off")

    panel_titles = {
        "center": "Center checkpoint",
        "plus_d1": "+epsilon d1",
        "minus_d1": "-epsilon d1",
        "plus_d2": "+epsilon d2",
        "minus_d2": "-epsilon d2",
    }

    last_image = None

    for panel_index, name in enumerate(ordered_names, start=1):
        last_image = axes[panel_index].imshow(
            magnitude_maps[name],
            cmap="hot",
            vmin=0.0,
            vmax=maximum,
        )

        title = panel_titles[name]

        if name != "center":
            distance = metric_rows[name]["magnitude_cosine_distance"]
            title += f"\n1-cos={distance:.3f}"

        axes[panel_index].set_title(title)
        axes[panel_index].axis("off")

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes[1:].tolist(),
            shrink=0.8,
            label="Absolute IG magnitude",
        )

    fig.suptitle(
        f"IG sensitivity to local weight perturbations\n"
        f"{EXPERIMENT_NAME}, epoch {epoch}, image {image_index}, "
        f"test lambda_max={lambda_max:.3f}"
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.92,
        bottom=0.04,
        top=0.88,
        wspace=0.18,
        hspace=0.25,
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    d1_path = (
        LANDSCAPE_DIR
        / "shared_direction_d1_epoch300.npy"
    )

    d2_path = (
        LANDSCAPE_DIR
        / "shared_direction_d2_epoch300.npy"
    )

    require_file(d1_path)
    require_file(d2_path)

    d1_numpy = np.load(d1_path)
    d2_numpy = np.load(d2_path)

    per_perturbation_rows: List[Dict[str, object]] = []
    per_image_rows: List[Dict[str, object]] = []
    epoch_summary_rows: List[Dict[str, object]] = []

    center_attributions: Dict[
        Tuple[int, int],
        np.ndarray,
    ] = {}

    selected_indices: List[int] | None = None

    print("\n" + "=" * 72)
    print("IG WEIGHT-SPACE SENSITIVITY EXPERIMENT")
    print("=" * 72)
    print(f"Experiment:       {EXPERIMENT_NAME}")
    print(f"Epochs:           {EPOCHS}")
    print(f"Landscape split:  {LANDSCAPE_SPLIT}")
    print(f"BN mode:          {BN_MODE}")
    print(f"IG steps:         {N_STEPS}")

    for epoch in EPOCHS:
        checkpoint_path = (
            CHECKPOINT_DIR
            / f"resnet56_epoch_{epoch}.pth"
        )

        landscape_path = (
            LANDSCAPE_DIR
            / f"epoch_{epoch:03d}.npz"
        )

        require_file(checkpoint_path)
        require_file(landscape_path)

        geometry = load_local_landscape_geometry(
            landscape_path,
            LANDSCAPE_SPLIT,
        )

        print("\n" + "=" * 72)
        print(f"EPOCH {epoch}")
        print("=" * 72)
        print(f"lambda_min:       {geometry['lambda_min']:.6f}")
        print(f"lambda_max:       {geometry['lambda_max']:.6f}")
        print(f"anisotropy:       {geometry['anisotropy']:.6f}")
        print(f"epsilon alpha:    {geometry['epsilon_alpha']:.6f}")
        print(f"epsilon beta:     {geometry['epsilon_beta']:.6f}")

        (
            model,
            train_set,
            test_set,
            train_loader,
            _,
            device,
        ) = setup(checkpoint_path)

        if selected_indices is None:
            if EXPLICIT_TEST_INDICES is not None:
                selected_indices = list(EXPLICIT_TEST_INDICES)
            else:
                selected_indices = select_balanced_indices(
                    test_set,
                    IMAGES_PER_CLASS,
                )

            with (
                RESULT_DIR
                / "selected_test_indices.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    selected_indices,
                    handle,
                    indent=2,
                )

            print(
                f"Selected {len(selected_indices)} fixed test images: "
                f"{selected_indices}"
            )

        original_state = copy.deepcopy(
            model.state_dict()
        )

        center_weights = get_params_vector(
            model
        ).clone()

        d1 = torch.from_numpy(
            d1_numpy
        ).to(
            device=device,
            dtype=center_weights.dtype,
        )

        d2 = torch.from_numpy(
            d2_numpy
        ).to(
            device=device,
            dtype=center_weights.dtype,
        )

        if d1.numel() != center_weights.numel():
            raise ValueError(
                "d1 has the wrong length for this model: "
                f"{d1.numel()} vs {center_weights.numel()}"
            )

        if d2.numel() != center_weights.numel():
            raise ValueError(
                "d2 has the wrong length for this model: "
                f"{d2.numel()} vs {center_weights.numel()}"
            )

        perturbations = {
            "plus_d1": (
                center_weights
                + float(geometry["epsilon_alpha"]) * d1
            ),
            "minus_d1": (
                center_weights
                - float(geometry["epsilon_alpha"]) * d1
            ),
            "plus_d2": (
                center_weights
                + float(geometry["epsilon_beta"]) * d2
            ),
            "minus_d2": (
                center_weights
                - float(geometry["epsilon_beta"]) * d2
            ),
        }

        epoch_image_sensitivities: List[float] = []
        epoch_signed_sensitivities: List[float] = []
        epoch_prediction_stability: List[float] = []
        epoch_center_correct: List[float] = []

        for image_number, image_index in enumerate(
            selected_indices,
            start=1,
        ):
            image, label = test_set[image_index]
            ground_truth = int(label.item())

            if TARGET_MODE != "ground_truth":
                raise ValueError(
                    "Only TARGET_MODE='ground_truth' is implemented."
                )

            target = ground_truth

            print(
                f"\n[{image_number:02d}/{len(selected_indices)}] "
                f"epoch={epoch}, image={image_index}, target={target}"
            )

            # -------------------------------------------------------------
            # Center checkpoint
            # -------------------------------------------------------------

            prepare_model_state(
                model,
                original_state,
                center_weights,
                train_loader,
                device,
            )

            center_attr, center_delta = compute_integrated_gradients(
                model=model,
                image=image,
                label=target,
                device=device,
                n_steps=N_STEPS,
            )

            center_prediction_info = prediction_info(
                model,
                image,
                target,
                device,
            )

            center_signed, center_magnitude = attribution_arrays(
                center_attr
            )

            center_attributions[
                (epoch, image_index)
            ] = center_signed.copy()

            magnitude_maps: Dict[str, np.ndarray] = {
                "center": center_magnitude,
            }

            metric_rows_for_figure: Dict[
                str,
                Dict[str, float],
            ] = {
                "center": {},
            }

            raw_attributions: Dict[str, np.ndarray] = {
                "center": center_signed,
            }

            image_metric_rows: List[Dict[str, object]] = []

            # -------------------------------------------------------------
            # Four matched local perturbations
            # -------------------------------------------------------------

            for perturbation_name, perturbed_weights in perturbations.items():
                prepare_model_state(
                    model,
                    original_state,
                    perturbed_weights,
                    train_loader,
                    device,
                )

                perturbed_attr, perturbed_delta = (
                    compute_integrated_gradients(
                        model=model,
                        image=image,
                        label=target,
                        device=device,
                        n_steps=N_STEPS,
                    )
                )

                perturbed_prediction_info = prediction_info(
                    model,
                    image,
                    target,
                    device,
                )

                metrics = compare_attributions(
                    center_attr,
                    perturbed_attr,
                )

                pert_signed, pert_magnitude = attribution_arrays(
                    perturbed_attr
                )

                magnitude_maps[
                    perturbation_name
                ] = pert_magnitude

                metric_rows_for_figure[
                    perturbation_name
                ] = metrics

                raw_attributions[
                    perturbation_name
                ] = pert_signed

                row: Dict[str, object] = {
                    "experiment": EXPERIMENT_NAME,
                    "epoch": epoch,
                    "image_index": image_index,
                    "ground_truth": ground_truth,
                    "target": target,
                    "landscape_split": LANDSCAPE_SPLIT,
                    "bn_mode": BN_MODE,
                    "n_steps": N_STEPS,
                    "perturbation": perturbation_name,
                    "epsilon_alpha": geometry["epsilon_alpha"],
                    "epsilon_beta": geometry["epsilon_beta"],
                    "lambda_min": geometry["lambda_min"],
                    "lambda_max": geometry["lambda_max"],
                    "anisotropy": geometry["anisotropy"],
                    "center_prediction": center_prediction_info["prediction"],
                    "center_confidence": center_prediction_info["confidence"],
                    "center_target_probability": (
                        center_prediction_info["target_probability"]
                    ),
                    "perturbed_prediction": (
                        perturbed_prediction_info["prediction"]
                    ),
                    "perturbed_confidence": (
                        perturbed_prediction_info["confidence"]
                    ),
                    "perturbed_target_probability": (
                        perturbed_prediction_info["target_probability"]
                    ),
                    "prediction_unchanged": int(
                        center_prediction_info["prediction"]
                        == perturbed_prediction_info["prediction"]
                    ),
                    "center_correct": int(
                        center_prediction_info["prediction"]
                        == ground_truth
                    ),
                    "perturbed_correct": int(
                        perturbed_prediction_info["prediction"]
                        == ground_truth
                    ),
                    "center_ig_delta": convergence_delta_value(
                        center_delta
                    ),
                    "perturbed_ig_delta": convergence_delta_value(
                        perturbed_delta
                    ),
                    **metrics,
                }

                per_perturbation_rows.append(
                    row
                )

                image_metric_rows.append(
                    row
                )

                print(
                    f"  {perturbation_name:9s} | "
                    f"1-cos(mag)={metrics['magnitude_cosine_distance']:.4f} | "
                    f"top-k={metrics['top_k_overlap']:.3f} | "
                    f"pred same={row['prediction_unchanged']}"
                )

            # Restore the exact checkpoint before moving on.
            model.load_state_dict(
                original_state,
                strict=True,
            )

            model.eval()

            magnitude_distances = np.array(
                [
                    float(row["magnitude_cosine_distance"])
                    for row in image_metric_rows
                ],
                dtype=np.float64,
            )

            signed_distances = np.array(
                [
                    float(row["signed_cosine_distance"])
                    for row in image_metric_rows
                ],
                dtype=np.float64,
            )

            prediction_unchanged_values = np.array(
                [
                    float(row["prediction_unchanged"])
                    for row in image_metric_rows
                ],
                dtype=np.float64,
            )

            image_summary = {
                "experiment": EXPERIMENT_NAME,
                "epoch": epoch,
                "image_index": image_index,
                "ground_truth": ground_truth,
                "center_prediction": center_prediction_info["prediction"],
                "center_confidence": center_prediction_info["confidence"],
                "center_correct": int(
                    center_prediction_info["prediction"]
                    == ground_truth
                ),
                "lambda_min": geometry["lambda_min"],
                "lambda_max": geometry["lambda_max"],
                "anisotropy": geometry["anisotropy"],
                "mean_magnitude_cosine_distance": float(
                    np.nanmean(magnitude_distances)
                ),
                "max_magnitude_cosine_distance": float(
                    np.nanmax(magnitude_distances)
                ),
                "mean_signed_cosine_distance": float(
                    np.nanmean(signed_distances)
                ),
                "max_signed_cosine_distance": float(
                    np.nanmax(signed_distances)
                ),
                "prediction_stability": float(
                    np.nanmean(prediction_unchanged_values)
                ),
            }

            per_image_rows.append(
                image_summary
            )

            epoch_image_sensitivities.append(
                image_summary[
                    "mean_magnitude_cosine_distance"
                ]
            )

            epoch_signed_sensitivities.append(
                image_summary[
                    "mean_signed_cosine_distance"
                ]
            )

            epoch_prediction_stability.append(
                image_summary[
                    "prediction_stability"
                ]
            )

            epoch_center_correct.append(
                image_summary[
                    "center_correct"
                ]
            )

            raw_path = (
                RESULT_DIR
                / "raw_attributions"
                / f"epoch_{epoch:03d}"
                / f"image_{image_index:05d}.npz"
            )

            raw_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                raw_path,
                center=raw_attributions["center"],
                plus_d1=raw_attributions["plus_d1"],
                minus_d1=raw_attributions["minus_d1"],
                plus_d2=raw_attributions["plus_d2"],
                minus_d2=raw_attributions["minus_d2"],
                ground_truth=ground_truth,
                center_prediction=center_prediction_info["prediction"],
                epoch=epoch,
                image_index=image_index,
            )

            if SAVE_PERTURBATION_FIGURES:
                figure_path = (
                    PLOT_DIR
                    / f"epoch_{epoch:03d}"
                    / f"image_{image_index:05d}_ig_sensitivity.png"
                )

                save_perturbation_figure(
                    image=image,
                    magnitude_maps=magnitude_maps,
                    metric_rows=metric_rows_for_figure,
                    epoch=epoch,
                    image_index=image_index,
                    target=target,
                    center_prediction=int(
                        center_prediction_info["prediction"]
                    ),
                    lambda_max=float(
                        geometry["lambda_max"]
                    ),
                    save_path=figure_path,
                )

        epoch_summary = {
            "experiment": EXPERIMENT_NAME,
            "epoch": epoch,
            "num_images": len(selected_indices),
            "landscape_split": LANDSCAPE_SPLIT,
            "bn_mode": BN_MODE,
            "n_steps": N_STEPS,
            "epsilon_alpha": geometry["epsilon_alpha"],
            "epsilon_beta": geometry["epsilon_beta"],
            "center_loss": geometry["center_loss"],
            "lambda_min": geometry["lambda_min"],
            "lambda_max": geometry["lambda_max"],
            "anisotropy": geometry["anisotropy"],
            "mean_ig_sensitivity": float(
                np.nanmean(epoch_image_sensitivities)
            ),
            "std_ig_sensitivity": float(
                np.nanstd(
                    epoch_image_sensitivities,
                    ddof=1,
                )
                if len(epoch_image_sensitivities) > 1
                else 0.0
            ),
            "mean_signed_ig_sensitivity": float(
                np.nanmean(epoch_signed_sensitivities)
            ),
            "prediction_stability": float(
                np.nanmean(epoch_prediction_stability)
            ),
            "selected_image_accuracy": float(
                np.nanmean(epoch_center_correct)
            ),
        }

        epoch_summary_rows.append(
            epoch_summary
        )

        print("\nEpoch summary")
        print(
            f"  lambda_max:             "
            f"{epoch_summary['lambda_max']:.6f}"
        )
        print(
            f"  mean IG sensitivity:    "
            f"{epoch_summary['mean_ig_sensitivity']:.6f}"
        )
        print(
            f"  prediction stability:   "
            f"{epoch_summary['prediction_stability']:.3f}"
        )
        print(
            f"  selected-image accuracy:"
            f" {epoch_summary['selected_image_accuracy']:.3f}"
        )

    # =========================================================================
    # Cross-epoch center-attribution drift
    # =========================================================================

    cross_epoch_rows: List[Dict[str, object]] = []

    if len(EPOCHS) >= 2 and selected_indices is not None:
        first_epoch = EPOCHS[0]
        last_epoch = EPOCHS[-1]

        for image_index in selected_indices:
            first_attr = center_attributions[
                (first_epoch, image_index)
            ]

            last_attr = center_attributions[
                (last_epoch, image_index)
            ]

            first_tensor = torch.from_numpy(
                first_attr
            ).unsqueeze(0)

            last_tensor = torch.from_numpy(
                last_attr
            ).unsqueeze(0)

            drift_metrics = compare_attributions(
                first_tensor,
                last_tensor,
            )

            cross_epoch_rows.append(
                {
                    "experiment": EXPERIMENT_NAME,
                    "first_epoch": first_epoch,
                    "last_epoch": last_epoch,
                    "image_index": image_index,
                    **drift_metrics,
                }
            )

    # =========================================================================
    # Save tabular results
    # =========================================================================

    write_csv(
        RESULT_DIR / "per_perturbation_metrics.csv",
        per_perturbation_rows,
    )

    write_csv(
        RESULT_DIR / "per_image_summary.csv",
        per_image_rows,
    )

    write_csv(
        RESULT_DIR / "epoch_summary.csv",
        epoch_summary_rows,
    )

    write_csv(
        RESULT_DIR / "cross_epoch_ig_drift.csv",
        cross_epoch_rows,
    )

    # =========================================================================
    # Summary figure: sharpness versus local IG sensitivity
    # =========================================================================

    epochs = np.array(
        [
            int(row["epoch"])
            for row in epoch_summary_rows
        ],
        dtype=np.int64,
    )

    lambda_max_values = np.array(
        [
            float(row["lambda_max"])
            for row in epoch_summary_rows
        ],
        dtype=np.float64,
    )

    ig_sensitivity_values = np.array(
        [
            float(row["mean_ig_sensitivity"])
            for row in epoch_summary_rows
        ],
        dtype=np.float64,
    )

    ig_sensitivity_std = np.array(
        [
            float(row["std_ig_sensitivity"])
            for row in epoch_summary_rows
        ],
        dtype=np.float64,
    )

    order = np.argsort(epochs)

    epochs = epochs[order]
    lambda_max_values = lambda_max_values[order]
    ig_sensitivity_values = ig_sensitivity_values[order]
    ig_sensitivity_std = ig_sensitivity_std[order]

    fig, axis_left = plt.subplots(
        figsize=(8, 5),
    )

    axis_left.plot(
        epochs,
        lambda_max_values,
        marker="o",
        label="Test landscape lambda_max",
    )

    axis_left.set_xlabel(
        "Epoch"
    )

    axis_left.set_ylabel(
        "Local test-loss sharpness (lambda_max)"
    )

    axis_right = axis_left.twinx()

    axis_right.errorbar(
        epochs,
        ig_sensitivity_values,
        yerr=ig_sensitivity_std,
        marker="s",
        capsize=4,
        label="Mean local IG sensitivity",
    )

    axis_right.set_ylabel(
        "Mean IG sensitivity: 1 - cosine"
    )

    axis_left.grid(
        alpha=0.3
    )

    handles_left, labels_left = (
        axis_left.get_legend_handles_labels()
    )

    handles_right, labels_right = (
        axis_right.get_legend_handles_labels()
    )

    axis_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
    )

    fig.suptitle(
        f"Local loss sharpness vs IG weight sensitivity\n"
        f"{EXPERIMENT_NAME}"
    )

    fig.tight_layout()

    summary_plot_path = (
        PLOT_DIR
        / "sharpness_vs_ig_sensitivity.png"
    )

    fig.savefig(
        summary_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"Results: {RESULT_DIR}")
    print(f"Plots:   {PLOT_DIR}")
    print(
        "\nWith only two epochs, interpret the result descriptively; "
        "do not report a correlation coefficient as evidence of a trend."
    )


if __name__ == "__main__":
    main()
