import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepcfd_models import PhysicsInformedLossV2, build_model


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    mse_total_sum = 0.0
    mse_ux_sum = 0.0
    mse_uy_sum = 0.0
    mse_p_sum = 0.0

    physics = PhysicsInformedLossV2().to(device)
    phys_sum = 0.0

    n_total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        n = len(x)
        n_total += n

        mse_total_sum += F.mse_loss(pred, y).item() * n
        mse_ux_sum += F.mse_loss(pred[:, 0], y[:, 0]).item() * n
        mse_uy_sum += F.mse_loss(pred[:, 1], y[:, 1]).item() * n
        mse_p_sum += F.mse_loss(pred[:, 2], y[:, 2]).item() * n

        phys = physics(pred, x, {
            "continuity": 1.0,
            "momentum": 1.0,
            "boundary": 1.0,
            "pressure_gauge": 1.0,
        })
        phys_sum += phys["total_physics"].item() * n

    return {
        "mse": mse_total_sum / n_total,
        "mse_ux": mse_ux_sum / n_total,
        "mse_uy": mse_uy_sum / n_total,
        "mse_p": mse_p_sum / n_total,
        "physics_total": phys_sum / n_total,
    }


@torch.no_grad()
def save_visual_comparison(models, test_x, test_y, out_dir, sample_idx=0, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    gt = test_y[sample_idx].numpy()

    keys = list(models.keys())
    fields = ["Ux", "Uy", "p"]

    fig, axes = plt.subplots(3, len(keys) + 1, figsize=(4 * (len(keys) + 1), 10))

    for row, field in enumerate(fields):
        vmin, vmax = gt[row].min(), gt[row].max()
        im = axes[row, 0].imshow(gt[row].T, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"GT {field}")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)

        for col, key in enumerate(keys, start=1):
            model = models[key]
            pred = model(test_x[sample_idx : sample_idx + 1].to(device)).cpu().numpy()[0]
            im = axes[row, col].imshow(pred[row].T, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{key} {field}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"comparison_sample_{sample_idx}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Test retrained DeepCFD models")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-dir", type=str, default="models/test_outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    x = pickle.load(open(os.path.join(args.dataset_dir, "dataX.pkl"), "rb"))
    y = pickle.load(open(os.path.join(args.dataset_dir, "dataY.pkl"), "rb"))
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    split_idx = int(0.7 * len(x))
    test_x, test_y = x[split_idx:], y[split_idx:]
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    candidates = {
        "base_data": "model_base_data.pt",
        "base_phys": "model_base_phys.pt",
        "attention_data": "model_attention_data.pt",
        "attention_phys": "model_attention_phys.pt",
        "transformer_data": "model_transformer_data.pt",
        "transformer_phys": "model_transformer_phys.pt",
    }

    loaded = {}
    metrics = {}

    for logical_name, filename in candidates.items():
        path = os.path.join(args.models_dir, filename)
        if not os.path.exists(path):
            continue

        if "base" in logical_name:
            model = build_model("base")
        elif "attention" in logical_name:
            model = build_model("attention")
        else:
            model = build_model("transformer")

        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        loaded[logical_name] = model
        metrics[logical_name] = evaluate_model(model, test_loader, device)

    if not loaded:
        raise RuntimeError("No compatible model checkpoints found in models directory.")

    save_visual_comparison(loaded, test_x, test_y, args.out_dir, sample_idx=0, device=device)

    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
 
