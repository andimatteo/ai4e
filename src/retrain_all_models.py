import argparse
import json
import os
import pickle
import random
import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepcfd_models import PhysicsInformedLossV2, build_model, count_params


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset_dir: str, device: torch.device):
    x = pickle.load(open(os.path.join(dataset_dir, "dataX.pkl"), "rb"))
    y = pickle.load(open(os.path.join(dataset_dir, "dataY.pkl"), "rb"))

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    batch, nx, ny = y.shape[0], y.shape[2], y.shape[3]
    channel_weights = torch.sqrt(
        torch.mean(y.permute(0, 2, 3, 1).reshape((batch * nx * ny, 3)) ** 2, dim=0)
    ).view(1, -1, 1, 1).to(device)

    return x, y, channel_weights


def split_data(x: torch.Tensor, y: torch.Tensor, split_ratio: float = 0.7, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=g)
    x = x[indices]
    y = y[indices]

    split_idx = int(split_ratio * len(x))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def data_loss_fn(output: torch.Tensor, target: torch.Tensor, channel_weights: torch.Tensor) -> torch.Tensor:
    lossu = ((output[:, 0] - target[:, 0]) ** 2).unsqueeze(1)
    lossv = ((output[:, 1] - target[:, 1]) ** 2).unsqueeze(1)
    lossp = torch.abs(output[:, 2] - target[:, 2]).unsqueeze(1)
    loss = torch.cat([lossu, lossv, lossp], dim=1) / channel_weights
    return torch.mean(loss)


def physics_ramp(epoch: int, ramp_start: int = 30, ramp_end: int = 220) -> float:
    if epoch < ramp_start:
        return 0.0
    if epoch > ramp_end:
        return 1.0
    return (epoch - ramp_start) / max(1, (ramp_end - ramp_start))


def compute_gradnorm_lambda(l_data: torch.Tensor, l_phys: torch.Tensor, output: torch.Tensor, eps: float = 1e-8):
    grad_data = torch.autograd.grad(l_data, output, retain_graph=True, create_graph=False)[0]
    grad_phys = torch.autograd.grad(l_phys, output, retain_graph=True, create_graph=False)[0]

    norm_data = grad_data.abs().mean()
    norm_phys = grad_phys.abs().mean()
    lam = (norm_data / (norm_phys + eps)).detach()
    return torch.clamp(lam, min=1e-3, max=10.0)


def train_one_epoch(
    model,
    loader,
    optimizer,
    channel_weights,
    physics_loss_fn,
    lambda_phys,
    epoch,
    adaptive_phys_weight,
    phys_weights,
):
    model.train()
    total_loss = 0.0
    loss_data_sum = 0.0
    loss_phys_sum = 0.0
    lambda_eff_sum = 0.0

    ramp = physics_ramp(epoch)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(channel_weights.device)
        batch_y = batch_y.to(channel_weights.device)

        optimizer.zero_grad()
        output = model(batch_x)

        l_data = data_loss_fn(output, batch_y, channel_weights)
        l_phys = torch.tensor(0.0, device=channel_weights.device)
        lambda_dyn = torch.tensor(1.0, device=channel_weights.device)

        if physics_loss_fn is not None and ramp > 0:
            phys_dict = physics_loss_fn(output, batch_x, phys_weights)
            l_phys = phys_dict["total_physics"]
            if adaptive_phys_weight:
                lambda_dyn = compute_gradnorm_lambda(l_data, l_phys, output)

        lambda_eff = lambda_phys * ramp * lambda_dyn
        loss = l_data + lambda_eff * l_phys

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = len(batch_x)
        total_loss += loss.item() * n
        loss_data_sum += l_data.item() * n
        loss_phys_sum += l_phys.item() * n
        lambda_eff_sum += lambda_eff.item() * n

    n_total = len(loader.dataset)
    return {
        "loss": total_loss / n_total,
        "loss_data": loss_data_sum / n_total,
        "loss_physics": loss_phys_sum / n_total,
        "lambda_eff": lambda_eff_sum / n_total,
    }


@torch.no_grad()
def evaluate_model(model, loader, channel_weights, physics_loss_fn, phys_weights):
    model.eval()
    total_loss = 0.0
    mse_total_sum = 0.0
    mse_ux_sum = 0.0
    mse_uy_sum = 0.0
    mse_p_sum = 0.0
    loss_phys_sum = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(channel_weights.device)
        batch_y = batch_y.to(channel_weights.device)
        output = model(batch_x)

        l_data = data_loss_fn(output, batch_y, channel_weights)
        total_loss += l_data.item() * len(batch_x)

        if physics_loss_fn is not None:
            l_phys = physics_loss_fn(output, batch_x, phys_weights)["total_physics"]
            loss_phys_sum += l_phys.item() * len(batch_x)

        mse_total_sum += F.mse_loss(output, batch_y).item() * len(batch_x)
        mse_ux_sum += F.mse_loss(output[:, 0], batch_y[:, 0]).item() * len(batch_x)
        mse_uy_sum += F.mse_loss(output[:, 1], batch_y[:, 1]).item() * len(batch_x)
        mse_p_sum += F.mse_loss(output[:, 2], batch_y[:, 2]).item() * len(batch_x)

    n_total = len(loader.dataset)
    return {
        "loss": total_loss / n_total,
        "loss_physics": loss_phys_sum / n_total,
        "mse": mse_total_sum / n_total,
        "mse_ux": mse_ux_sum / n_total,
        "mse_uy": mse_uy_sum / n_total,
        "mse_p": mse_p_sum / n_total,
    }


def run_experiment(
    model_name: str,
    use_physics: bool,
    train_loader,
    test_loader,
    channel_weights,
    epochs: int,
    lr: float,
    patience: int,
    lambda_phys: float,
    adaptive_phys_weight: bool,
    phys_weights: Dict[str, float],
):
    model = build_model(model_name).to(channel_weights.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=18)

    physics_fn = PhysicsInformedLossV2().to(channel_weights.device) if use_physics else None

    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(
            model,
            train_loader,
            optimizer,
            channel_weights,
            physics_fn,
            lambda_phys,
            epoch,
            adaptive_phys_weight,
            phys_weights,
        )
        val_res = evaluate_model(model, test_loader, channel_weights, physics_fn, phys_weights)
        scheduler.step(val_res["loss"])

        history["train"].append(train_res)
        history["val"].append(val_res)

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"[{model_name} | {'PINN' if use_physics else 'DATA'}] "
                f"Epoch {epoch:4d}/{epochs} | train={train_res['loss']:.6f} | val={val_res['loss']:.6f} | mse={val_res['mse']:.6f}"
            )

        if val_res["loss"] < best_val:
            best_val = val_res["loss"]
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: {model_name} ({'PINN' if use_physics else 'DATA'}) at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    key_suffix = "phys" if use_physics else "data"
    result_key = f"{model_name}_{key_suffix}"

    return result_key, {
        "model": model,
        "history": history,
        "best_val_loss": best_val,
        "params": count_params(model),
        "mode": key_suffix,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain DeepCFD models with improved PINN.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")

    # More epochs for PINN and especially transformer, as requested.
    parser.add_argument("--epochs-data", type=int, default=320)
    parser.add_argument("--epochs-phys", type=int, default=650)
    parser.add_argument("--epochs-trans-data", type=int, default=380)
    parser.add_argument("--epochs-trans-phys", type=int, default=800)

    parser.add_argument("--lambda-phys", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    x, y, channel_weights = load_dataset(args.dataset_dir, device)
    train_x, train_y, test_x, test_y = split_data(x, y, split_ratio=0.7, seed=args.seed)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    phys_weights = {
        "continuity": 1.0,
        "momentum": 0.08,
        "boundary": 1.0,
        "pressure_gauge": 0.05,
    }

    plan = [
        ("base", False, args.epochs_data),
        ("base", True, args.epochs_phys),
        ("attention", False, args.epochs_data),
        ("attention", True, args.epochs_phys),
        ("transformer", False, args.epochs_trans_data),
        ("transformer", True, args.epochs_trans_phys),
    ]

    all_results = {}
    started = time.time()

    for model_name, use_physics, epochs in plan:
        key, result = run_experiment(
            model_name=model_name,
            use_physics=use_physics,
            train_loader=train_loader,
            test_loader=test_loader,
            channel_weights=channel_weights,
            epochs=epochs,
            lr=args.lr,
            patience=args.patience,
            lambda_phys=args.lambda_phys,
            adaptive_phys_weight=True,
            phys_weights=phys_weights,
        )
        all_results[key] = result

        ckpt = {
            "state_dict": result["model"].state_dict(),
            "name": key,
            "best_val_loss": result["best_val_loss"],
            "params": result["params"],
            "mode": result["mode"],
            "physics_weights": phys_weights,
            "lambda_phys": args.lambda_phys,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"model_{key}.pt"))

    history = {k: v["history"] for k, v in all_results.items()}
    with open(os.path.join(args.out_dir, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    summary = {}
    for k, v in all_results.items():
        val_final = evaluate_model(
            v["model"],
            test_loader,
            channel_weights,
            PhysicsInformedLossV2().to(device) if v["mode"] == "phys" else None,
            phys_weights,
        )
        summary[k] = {
            "best_val_loss": float(v["best_val_loss"]),
            "mse": float(val_final["mse"]),
            "mse_ux": float(val_final["mse_ux"]),
            "mse_uy": float(val_final["mse_uy"]),
            "mse_p": float(val_final["mse_p"]),
            "params": int(v["params"]),
            "mode": v["mode"],
        }

    summary["meta"] = {
        "device": str(device),
        "elapsed_sec": round(time.time() - started, 2),
        "train_size": int(len(train_x)),
        "test_size": int(len(test_x)),
    }

    with open(os.path.join(args.out_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
 
