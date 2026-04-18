import argparse
import json
import os
import pickle
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepcfd_models import PhysicsInformedLossV2


class ConvBlockCompat(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class EncoderBlockCompat(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv1 = ConvBlockCompat(in_ch, out_ch, kernel_size)
        self.conv2 = ConvBlockCompat(out_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        size = x.size()
        x_pooled, indices = self.pool(x)
        return x_pooled, x, indices, size


class DecoderBlockCompat(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = ConvBlockCompat(in_ch * 2, out_ch, kernel_size)
        self.conv2 = ConvBlockCompat(out_ch, out_ch, kernel_size)

    def forward(self, x, skip, indices, size):
        x = self.unpool(x, indices, output_size=size)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeepCFDCompat(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, filters=None):
        super().__init__()
        if filters is None:
            filters = [8, 16, 32, 64]

        self.encoder1 = EncoderBlockCompat(in_channels, filters[0], kernel_size)
        self.encoder2 = EncoderBlockCompat(filters[0], filters[1], kernel_size)
        self.encoder3 = EncoderBlockCompat(filters[1], filters[2], kernel_size)
        self.encoder4 = EncoderBlockCompat(filters[2], filters[3], kernel_size)

        self.bottleneck = nn.Sequential(
            ConvBlockCompat(filters[3], filters[3], kernel_size),
            ConvBlockCompat(filters[3], filters[3], kernel_size),
        )

        self.decoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderBlockCompat(filters[3], filters[2], kernel_size),
                        DecoderBlockCompat(filters[2], filters[1], kernel_size),
                        DecoderBlockCompat(filters[1], filters[0], kernel_size),
                        DecoderBlockCompat(filters[0], filters[0], kernel_size),
                        nn.Conv2d(filters[0], 1, kernel_size=1),
                    ]
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x):
        x1, s1, i1, sz1 = self.encoder1(x)
        x2, s2, i2, sz2 = self.encoder2(x1)
        x3, s3, i3, sz3 = self.encoder3(x2)
        x4, s4, i4, sz4 = self.encoder4(x3)
        z = self.bottleneck(x4)

        outputs = []
        for dec in self.decoders:
            d = dec[0](z, s4, i4, sz4)
            d = dec[1](d, s3, i3, sz3)
            d = dec[2](d, s2, i2, sz2)
            d = dec[3](d, s1, i1, sz1)
            d = dec[4](d)
            outputs.append(d)
        return torch.cat(outputs, dim=1)


def load_test_split(dataset_dir):
    x = pickle.load(open(os.path.join(dataset_dir, "dataX.pkl"), "rb"))
    y = pickle.load(open(os.path.join(dataset_dir, "dataY.pkl"), "rb"))
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    split_idx = int(0.7 * len(x))
    return x[split_idx:], y[split_idx:]


def build_model_from_state_dict(state_dict, filters=None, kernel_size=5):
    model = DeepCFDCompat(filters=filters, kernel_size=kernel_size)
    model.load_state_dict(state_dict, strict=True)
    return model
def discover_models(models_dir, device):
    discovered = OrderedDict()

    main_ckpt_path = os.path.join(models_dir, "deepcfd_model.pt")
    if os.path.exists(main_ckpt_path):
        ckpt = torch.load(main_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            discovered["deepcfd_data"] = build_model_from_state_dict(
                ckpt["state_dict"], 
                filters=ckpt.get("filters"), 
                kernel_size=ckpt.get("kernel_size", 5)
            ).to(device)

    ab_path = os.path.join(models_dir, "deepcfd_ab_results.pt")
    if os.path.exists(ab_path):
        ab = torch.load(ab_path, map_location=device)
        ab_results = ab.get("ab_results", {}) if isinstance(ab, dict) else {}
        for key in ["data_only", "physics_informed"]:
            if key in ab_results and "state_dict" in ab_results[key]:
                label = f"ab_{key}"
                discovered[label] = build_model_from_state_dict(
                    ab_results[key]["state_dict"],
                    filters=ab_results[key].get("filters"),
                    kernel_size=ab_results[key].get("kernel_size", 5)
                ).to(device)

    return discovered


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    physics = PhysicsInformedLossV2().to(device)

    mse_total_sum = 0.0
    mse_ux_sum = 0.0
    mse_uy_sum = 0.0
    mse_p_sum = 0.0
    mae_p_sum = 0.0
    rel_err_sum = 0.0
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
        mae_p_sum += F.l1_loss(pred[:, 2], y[:, 2]).item() * n

        rel_err = torch.abs(pred - y) / (torch.abs(y) + 1e-6)
        rel_err_sum += rel_err.mean().item() * n

        phys = physics(
            pred,
            x,
            {
                "continuity": 1.0,
                "momentum": 1.0,
                "boundary": 1.0,
                "pressure_gauge": 1.0,
            },
        )
        phys_sum += phys["total_physics"].item() * n

    return {
        "mse": mse_total_sum / n_total,
        "rmse": np.sqrt(mse_total_sum / n_total),
        "mse_ux": mse_ux_sum / n_total,
        "mse_uy": mse_uy_sum / n_total,
        "mse_p": mse_p_sum / n_total,
        "mae_p": mae_p_sum / n_total,
        "relative_error": rel_err_sum / n_total,
        "physics_total": phys_sum / n_total,
    }


@torch.no_grad()
def benchmark_latency(model, test_x, device, warmup=10, repeat=50):
    model.eval()
    x = test_x[:1].to(device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / repeat


@torch.no_grad()
def collect_relative_errors(model, test_x, test_y, device, max_samples=128):
    model.eval()
    x = test_x[:max_samples].to(device)
    y = test_y[:max_samples].to(device)
    pred = model(x)
    rel = torch.abs(pred - y) / (torch.abs(y) + 1e-6)
    return rel.detach().cpu().numpy().reshape(-1)


@torch.no_grad()
def save_field_triptychs(models, test_x, test_y, out_dir, sample_idx=0, max_models=4):
    keys = list(models.keys())[:max_models]
    gt = test_y[sample_idx].numpy()
    channel_names = ["Ux", "Uy", "Pressure"]

    for c, cname in enumerate(channel_names):
        fig, axes = plt.subplots(len(keys), 3, figsize=(12, 3.5 * max(1, len(keys))))
        if len(keys) == 1:
            axes = np.expand_dims(axes, axis=0)

        vmin, vmax = gt[c].min(), gt[c].max()

        for r, key in enumerate(keys):
            model = models[key]
            pred = model(test_x[sample_idx : sample_idx + 1].to(next(model.parameters()).device)).cpu().numpy()[0]
            err = np.abs(pred[c] - gt[c])

            im0 = axes[r, 0].imshow(gt[c].T, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
            axes[r, 0].set_title(f"GT {cname}")
            axes[r, 0].set_ylabel(key)

            im1 = axes[r, 1].imshow(pred[c].T, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
            axes[r, 1].set_title(f"Prediction {cname}")

            im2 = axes[r, 2].imshow(err.T, cmap="hot", origin="lower")
            axes[r, 2].set_title(f"|Error| {cname}")

            plt.colorbar(im0, ax=axes[r, 0], fraction=0.046)
            plt.colorbar(im1, ax=axes[r, 1], fraction=0.046)
            plt.colorbar(im2, ax=axes[r, 2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"field_triptych_{cname.lower()}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_compact_metric_plots(metrics, out_dir):
    names = list(metrics.keys())
    rmse_ux = [np.sqrt(metrics[k]["mse_ux"]) for k in names]
    rmse_uy = [np.sqrt(metrics[k]["mse_uy"]) for k in names]
    rmse_p = [np.sqrt(metrics[k]["mse_p"]) for k in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, rmse_ux, width=w, label="RMSE Ux")
    ax.bar(x, rmse_uy, width=w, label="RMSE Uy")
    ax.bar(x + w, rmse_p, width=w, label="RMSE p")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Comparative RMSE by Field")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_compact_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pareto_plot(metrics, lat_ms, out_dir):
    names = list(metrics.keys())
    xs = [lat_ms[k] for k in names]
    ys = [metrics[k]["rmse"] for k in names]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=80)
    for n, x, y in zip(names, xs, ys):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.set_xlabel("Latency [ms / sample]")
    ax.set_ylabel("RMSE total")
    ax.set_title("Pareto: Accuracy vs Inference Latency")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_latency_rmse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_relative_error_hist(rel_errors, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, arr in rel_errors.items():
        arr = np.clip(arr, 0, np.percentile(arr, 99.5))
        ax.hist(arr, bins=80, alpha=0.35, density=True, label=name)
    ax.set_xlabel("Relative error")
    ax.set_ylabel("Density")
    ax.set_title("Relative Error Distribution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relative_error_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Comparative testing with compact literature-style visualizations")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-dir", type=str, default="models/test_outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-models-fields", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    test_x, test_y = load_test_split(args.dataset_dir)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    models = discover_models(args.models_dir, device)
    if not models:
        raise RuntimeError("No compatible checkpoints found. Expected deepcfd_model.pt and/or deepcfd_ab_results.pt")

    metrics = OrderedDict()
    lat_ms = OrderedDict()
    rel_errors = OrderedDict()

    for name, model in models.items():
        metrics[name] = evaluate_model(model, test_loader, device)
        lat_ms[name] = benchmark_latency(model, test_x, device)
        rel_errors[name] = collect_relative_errors(model, test_x, test_y, device)

    for n in metrics:
        metrics[n]["latency_ms"] = lat_ms[n]

    save_compact_metric_plots(metrics, args.out_dir)
    save_pareto_plot(metrics, lat_ms, args.out_dir)
    save_relative_error_hist(rel_errors, args.out_dir)
    save_field_triptychs(models, test_x, test_y, args.out_dir, sample_idx=args.sample_idx, max_models=args.max_models_fields)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved comparative outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
 
 
