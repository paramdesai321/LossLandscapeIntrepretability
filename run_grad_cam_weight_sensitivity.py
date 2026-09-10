#!/usr/bin/env python3
"""Run Grad-CAM weight-sensitivity experiments at matched checkpoints.

This is the Grad-CAM counterpart to ``run_ig_weight_sensitivity.py``.  It
uses the same checkpoints, saved loss-landscape directions, local Hessian
geometry, fixed balanced test images, BatchNorm policy, and four matched
weight perturbations.  Results are written under ``results/grad_cam_weight_sensitivity``.

Grad-CAM produces one nonnegative spatial heatmap rather than signed
per-channel IG values.  For compatibility with the IG runner's metric and
storage helpers, each heatmap is represented internally as a one-channel
tensor of shape ``[1, 1, H, W]``.  The reported magnitude map is therefore
the Grad-CAM heatmap itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import run_ig_weight_sensitivity as base
from grad_cam import compute_grad_cam


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "sgd_nesterov"
EPOCHS = [150, 300]
CHECKPOINT_DIR = Path("training/checkpoints/sgd_nesterov/run2")
LANDSCAPE_DIR = Path("results/2d_sgd_nesterov")
RESULT_DIR = Path("results/grad_cam_weight_sensitivity/sgd_nesterov")
PLOT_DIR = Path("plots/grad_cam_weight_sensitivity/sgd_nesterov")

EXPLICIT_TEST_INDICES: List[int] | None = None
IMAGES_PER_CLASS = 2
TARGET_MODE = "ground_truth"
BN_MODE = "match_landscape"
BN_BATCHES = 5
EPSILON_OVERRIDE: float | None = None
SAVE_PERTURBATION_FIGURES = True
TOP_K_FRACTION = 0.10
LANDSCAPE_SPLIT = "test_Z"

# None selects the last Conv2d layer.  A dotted name such as
# "stage3.8.conv2" can be supplied for an explicit target layer.
TARGET_LAYER = None


def grad_cam_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Return Grad-CAM as a compatibility-shaped attribution tensor."""

    heatmap, prediction = compute_grad_cam(
        model=model,
        image=image,
        label=target,
        device=device,
        target_layer=TARGET_LAYER,
    )

    # [H, W] -> [1, 1, H, W], matching the shape expected by the IG helpers.
    return heatmap.unsqueeze(0).unsqueeze(0), prediction


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
    display_image = base.unnormalize_cifar_image(image)
    ordered_names = ["center", "plus_d1", "minus_d1", "plus_d2", "minus_d2"]
    maximum = max(float(np.max(magnitude_maps[name])) for name in ordered_names)
    maximum = max(maximum, 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.reshape(-1)
    axes[0].imshow(display_image)
    axes[0].set_title(f"Input\nGT={target}, Pred={center_prediction}")
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
            magnitude_maps[name], cmap="hot", vmin=0.0, vmax=maximum
        )
        panel_title = panel_titles[name]
        if name != "center":
            panel_title += (
                f"\n1-cos={metric_rows[name]['magnitude_cosine_distance']:.3f}"
            )
        axes[panel_index].set_title(panel_title)
        axes[panel_index].axis("off")

    if last_image is not None:
        fig.colorbar(last_image, ax=axes[1:].tolist(), shrink=0.8, label="Grad-CAM")

    fig.suptitle(
        f"Grad-CAM sensitivity to local weight perturbations\n"
        f"{EXPERIMENT_NAME}, epoch {epoch}, image {image_index}, "
        f"test lambda_max={lambda_max:.3f}"
    )
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.04, top=0.88,
                        wspace=0.18, hspace=0.25)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # The shared helper functions read these values from their original
    # module namespace, so update that namespace before using them.
    base.BN_MODE = BN_MODE
    base.BN_BATCHES = BN_BATCHES
    base.EPSILON_OVERRIDE = EPSILON_OVERRIDE
    base.TOP_K_FRACTION = TOP_K_FRACTION

    d1_path = LANDSCAPE_DIR / "shared_direction_d1_epoch300.npy"
    d2_path = LANDSCAPE_DIR / "shared_direction_d2_epoch300.npy"
    base.require_file(d1_path)
    base.require_file(d2_path)
    d1_numpy = np.load(d1_path)
    d2_numpy = np.load(d2_path)

    per_perturbation_rows: List[Dict[str, object]] = []
    per_image_rows: List[Dict[str, object]] = []
    epoch_summary_rows: List[Dict[str, object]] = []
    center_attributions: Dict[Tuple[int, int], np.ndarray] = {}
    selected_indices: List[int] | None = None

    print("\n" + "=" * 72)
    print("GRAD-CAM WEIGHT-SPACE SENSITIVITY EXPERIMENT")
    print("=" * 72)
    print(f"Experiment:       {EXPERIMENT_NAME}")
    print(f"Epochs:           {EPOCHS}")
    print(f"Landscape split:  {LANDSCAPE_SPLIT}")
    print(f"BN mode:          {BN_MODE}")
    print(f"Target layer:     {TARGET_LAYER or 'last Conv2d'}")

    for epoch in EPOCHS:
        checkpoint_path = CHECKPOINT_DIR / f"resnet56_epoch_{epoch}.pth"
        landscape_path = LANDSCAPE_DIR / f"epoch_{epoch:03d}.npz"
        base.require_file(checkpoint_path)
        base.require_file(landscape_path)

        geometry = base.load_local_landscape_geometry(landscape_path, LANDSCAPE_SPLIT)
        print("\n" + "=" * 72)
        print(f"EPOCH {epoch}")
        print("=" * 72)
        print(f"lambda_min:       {geometry['lambda_min']:.6f}")
        print(f"lambda_max:       {geometry['lambda_max']:.6f}")
        print(f"anisotropy:       {geometry['anisotropy']:.6f}")
        print(f"epsilon alpha:    {geometry['epsilon_alpha']:.6f}")
        print(f"epsilon beta:     {geometry['epsilon_beta']:.6f}")

        model, train_set, test_set, train_loader, _, device = base.setup(checkpoint_path)

        if selected_indices is None:
            selected_indices = (
                list(EXPLICIT_TEST_INDICES)
                if EXPLICIT_TEST_INDICES is not None
                else base.select_balanced_indices(test_set, IMAGES_PER_CLASS)
            )
            with (RESULT_DIR / "selected_test_indices.json").open("w", encoding="utf-8") as handle:
                json.dump(selected_indices, handle, indent=2)
            print(f"Selected {len(selected_indices)} fixed test images: {selected_indices}")

        original_state = copy.deepcopy(model.state_dict())
        center_weights = base.get_params_vector(model).clone()
        d1 = torch.from_numpy(d1_numpy).to(device=device, dtype=center_weights.dtype)
        d2 = torch.from_numpy(d2_numpy).to(device=device, dtype=center_weights.dtype)
        if d1.numel() != center_weights.numel() or d2.numel() != center_weights.numel():
            raise ValueError(
                "Saved perturbation directions do not match the model parameter count: "
                f"d1={d1.numel()}, d2={d2.numel()}, model={center_weights.numel()}"
            )

        perturbations = {
            "plus_d1": center_weights + float(geometry["epsilon_alpha"]) * d1,
            "minus_d1": center_weights - float(geometry["epsilon_alpha"]) * d1,
            "plus_d2": center_weights + float(geometry["epsilon_beta"]) * d2,
            "minus_d2": center_weights - float(geometry["epsilon_beta"]) * d2,
        }

        epoch_sensitivity: List[float] = []
        epoch_signed_sensitivity: List[float] = []
        epoch_prediction_stability: List[float] = []
        epoch_center_correct: List[float] = []

        for image_number, image_index in enumerate(selected_indices, start=1):
            image, label = test_set[image_index]
            ground_truth = int(label.item())
            if TARGET_MODE != "ground_truth":
                raise ValueError("Only TARGET_MODE='ground_truth' is implemented.")
            target = ground_truth
            print(f"\n[{image_number:02d}/{len(selected_indices)}] epoch={epoch}, image={image_index}, target={target}")

            base.prepare_model_state(model, original_state, center_weights, train_loader, device)
            center_attr, _ = grad_cam_attribution(model, image, target, device)
            center_prediction_info = base.prediction_info(model, image, target, device)
            center_signed, center_magnitude = base.attribution_arrays(center_attr)
            center_attributions[(epoch, image_index)] = center_signed.copy()

            magnitude_maps = {"center": center_magnitude}
            metric_rows_for_figure: Dict[str, Dict[str, float]] = {"center": {}}
            raw_attributions = {"center": center_signed}
            image_metric_rows: List[Dict[str, object]] = []

            for perturbation_name, perturbed_weights in perturbations.items():
                base.prepare_model_state(model, original_state, perturbed_weights, train_loader, device)
                perturbed_attr, _ = grad_cam_attribution(model, image, target, device)
                perturbed_prediction_info = base.prediction_info(model, image, target, device)
                metrics = base.compare_attributions(center_attr, perturbed_attr)
                pert_signed, pert_magnitude = base.attribution_arrays(perturbed_attr)
                magnitude_maps[perturbation_name] = pert_magnitude
                metric_rows_for_figure[perturbation_name] = metrics
                raw_attributions[perturbation_name] = pert_signed

                row: Dict[str, object] = {
                    "experiment": EXPERIMENT_NAME,
                    "epoch": epoch,
                    "image_index": image_index,
                    "ground_truth": ground_truth,
                    "target": target,
                    "landscape_split": LANDSCAPE_SPLIT,
                    "bn_mode": BN_MODE,
                    "target_layer": TARGET_LAYER or "last_conv2d",
                    "epsilon_alpha": geometry["epsilon_alpha"],
                    "epsilon_beta": geometry["epsilon_beta"],
                    "lambda_min": geometry["lambda_min"],
                    "lambda_max": geometry["lambda_max"],
                    "anisotropy": geometry["anisotropy"],
                    "center_prediction": center_prediction_info["prediction"],
                    "center_confidence": center_prediction_info["confidence"],
                    "center_target_probability": center_prediction_info["target_probability"],
                    "perturbed_prediction": perturbed_prediction_info["prediction"],
                    "perturbed_confidence": perturbed_prediction_info["confidence"],
                    "perturbed_target_probability": perturbed_prediction_info["target_probability"],
                    "prediction_unchanged": int(center_prediction_info["prediction"] == perturbed_prediction_info["prediction"]),
                    "center_correct": int(center_prediction_info["prediction"] == ground_truth),
                    "perturbed_correct": int(perturbed_prediction_info["prediction"] == ground_truth),
                    **metrics,
                }
                per_perturbation_rows.append(row)
                image_metric_rows.append(row)
                print(
                    f"  {perturbation_name:9s} | "
                    f"1-cos(mag)={metrics['magnitude_cosine_distance']:.4f} | "
                    f"top-k={metrics['top_k_overlap']:.3f} | "
                    f"pred same={row['prediction_unchanged']}"
                )

            model.load_state_dict(original_state, strict=True)
            model.eval()

            magnitude_distances = np.array([float(row["magnitude_cosine_distance"]) for row in image_metric_rows])
            signed_distances = np.array([float(row["signed_cosine_distance"]) for row in image_metric_rows])
            prediction_unchanged_values = np.array([float(row["prediction_unchanged"]) for row in image_metric_rows])
            image_summary = {
                "experiment": EXPERIMENT_NAME,
                "epoch": epoch,
                "image_index": image_index,
                "ground_truth": ground_truth,
                "center_prediction": center_prediction_info["prediction"],
                "center_confidence": center_prediction_info["confidence"],
                "center_correct": int(center_prediction_info["prediction"] == ground_truth),
                "lambda_min": geometry["lambda_min"],
                "lambda_max": geometry["lambda_max"],
                "anisotropy": geometry["anisotropy"],
                "mean_magnitude_cosine_distance": float(np.nanmean(magnitude_distances)),
                "max_magnitude_cosine_distance": float(np.nanmax(magnitude_distances)),
                "mean_signed_cosine_distance": float(np.nanmean(signed_distances)),
                "max_signed_cosine_distance": float(np.nanmax(signed_distances)),
                "prediction_stability": float(np.nanmean(prediction_unchanged_values)),
            }
            per_image_rows.append(image_summary)
            epoch_sensitivity.append(image_summary["mean_magnitude_cosine_distance"])
            epoch_signed_sensitivity.append(image_summary["mean_signed_cosine_distance"])
            epoch_prediction_stability.append(image_summary["prediction_stability"])
            epoch_center_correct.append(image_summary["center_correct"])

            raw_path = RESULT_DIR / "raw_attributions" / f"epoch_{epoch:03d}" / f"image_{image_index:05d}.npz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
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
                save_perturbation_figure(
                    image=image,
                    magnitude_maps=magnitude_maps,
                    metric_rows=metric_rows_for_figure,
                    epoch=epoch,
                    image_index=image_index,
                    target=target,
                    center_prediction=int(center_prediction_info["prediction"]),
                    lambda_max=float(geometry["lambda_max"]),
                    save_path=PLOT_DIR / f"epoch_{epoch:03d}" / f"image_{image_index:05d}_grad_cam_sensitivity.png",
                )

        epoch_summary_rows.append({
            "experiment": EXPERIMENT_NAME,
            "epoch": epoch,
            "num_images": len(selected_indices),
            "landscape_split": LANDSCAPE_SPLIT,
            "bn_mode": BN_MODE,
            "target_layer": TARGET_LAYER or "last_conv2d",
            "epsilon_alpha": geometry["epsilon_alpha"],
            "epsilon_beta": geometry["epsilon_beta"],
            "center_loss": geometry["center_loss"],
            "lambda_min": geometry["lambda_min"],
            "lambda_max": geometry["lambda_max"],
            "anisotropy": geometry["anisotropy"],
            "mean_grad_cam_sensitivity": float(np.nanmean(epoch_sensitivity)),
            "std_grad_cam_sensitivity": float(np.nanstd(epoch_sensitivity, ddof=1) if len(epoch_sensitivity) > 1 else 0.0),
            "mean_signed_grad_cam_sensitivity": float(np.nanmean(epoch_signed_sensitivity)),
            "prediction_stability": float(np.nanmean(epoch_prediction_stability)),
            "selected_image_accuracy": float(np.nanmean(epoch_center_correct)),
        })

    cross_epoch_rows: List[Dict[str, object]] = []
    if len(EPOCHS) >= 2 and selected_indices is not None:
        first_epoch, last_epoch = EPOCHS[0], EPOCHS[-1]
        for image_index in selected_indices:
            first_tensor = torch.from_numpy(center_attributions[(first_epoch, image_index)]).unsqueeze(0)
            last_tensor = torch.from_numpy(center_attributions[(last_epoch, image_index)]).unsqueeze(0)
            cross_epoch_rows.append({
                "experiment": EXPERIMENT_NAME,
                "first_epoch": first_epoch,
                "last_epoch": last_epoch,
                "image_index": image_index,
                **base.compare_attributions(first_tensor, last_tensor),
            })

    base.write_csv(RESULT_DIR / "per_perturbation_metrics.csv", per_perturbation_rows)
    base.write_csv(RESULT_DIR / "per_image_summary.csv", per_image_rows)
    base.write_csv(RESULT_DIR / "epoch_summary.csv", epoch_summary_rows)
    base.write_csv(RESULT_DIR / "cross_epoch_grad_cam_drift.csv", cross_epoch_rows)

    epochs = np.array([int(row["epoch"]) for row in epoch_summary_rows])
    lambda_max_values = np.array([float(row["lambda_max"]) for row in epoch_summary_rows])
    sensitivity_values = np.array([float(row["mean_grad_cam_sensitivity"]) for row in epoch_summary_rows])
    sensitivity_std = np.array([float(row["std_grad_cam_sensitivity"]) for row in epoch_summary_rows])
    order = np.argsort(epochs)

    fig, axis_left = plt.subplots(figsize=(8, 5))
    axis_left.plot(epochs[order], lambda_max_values[order], marker="o", label="Test landscape lambda_max")
    axis_left.set_xlabel("Epoch")
    axis_left.set_ylabel("Local test-loss sharpness (lambda_max)")
    axis_right = axis_left.twinx()
    axis_right.errorbar(epochs[order], sensitivity_values[order], yerr=sensitivity_std[order], marker="s", capsize=4, label="Mean local Grad-CAM sensitivity")
    axis_right.set_ylabel("Mean Grad-CAM sensitivity: 1 - cosine")
    axis_left.grid(alpha=0.3)
    handles_left, labels_left = axis_left.get_legend_handles_labels()
    handles_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    fig.suptitle(f"Local loss sharpness vs Grad-CAM weight sensitivity\n{EXPERIMENT_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sharpness_vs_grad_cam_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"Results: {RESULT_DIR}")
    print(f"Plots:   {PLOT_DIR}")
    print("\nWith only two epochs, interpret the result descriptively; do not report a correlation coefficient as evidence of a trend.")


if __name__ == "__main__":
    main()
