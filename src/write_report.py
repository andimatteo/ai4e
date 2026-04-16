import argparse
import datetime as dt
import json
import os


def build_markdown(summary, test_metrics):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# CFD AI Exam Report (Retraining)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Objective")
    lines.append("Retrain DeepCFD variants after partial repository recovery, with improved PINN documentation and reproducible testing.")
    lines.append("")
    lines.append("## Models")
    lines.append("- UNet-Base")
    lines.append("- UNet-Attention")
    lines.append("- UNet-Transformer (lightly refined pre-transform stage)")
    lines.append("")
    lines.append("## PINN Improvements Implemented")
    lines.append("- Residual scaling for continuity and momentum to reduce imbalance.")
    lines.append("- Stronger interior mask (2-cell margin) for derivative stability.")
    lines.append("- Outlet conditions expanded to dUx/dx=0, dUy/dx=0, dp/dx=0.")
    lines.append("- Pressure gauge penalty to reduce pressure drift/null-space issues.")
    lines.append("- Dynamic lambda balancing via gradient norms and longer PINN schedules.")
    lines.append("")

    lines.append("## Training Summary")
    meta = summary.get("meta", {})
    if meta:
        lines.append(f"- Device: {meta.get('device', 'n/a')}")
        lines.append(f"- Train samples: {meta.get('train_size', 'n/a')}")
        lines.append(f"- Test samples: {meta.get('test_size', 'n/a')}")
        lines.append(f"- Elapsed seconds: {meta.get('elapsed_sec', 'n/a')}")
    lines.append("")

    lines.append("## Validation Metrics (from training)")
    lines.append("| Model | Best Val Loss | MSE | MSE Ux | MSE Uy | MSE p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in summary.items():
        if k == "meta":
            continue
        lines.append(
            f"| {k} | {v.get('best_val_loss', 0):.6f} | {v.get('mse', 0):.6f} | "
            f"{v.get('mse_ux', 0):.6f} | {v.get('mse_uy', 0):.6f} | {v.get('mse_p', 0):.6f} |"
        )
    lines.append("")

    if test_metrics:
        lines.append("## Test Metrics (post-training evaluation)")
        lines.append("| Model | MSE | MSE Ux | MSE Uy | MSE p | Physics total |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k, v in test_metrics.items():
            lines.append(
                f"| {k} | {v.get('mse', 0):.6f} | {v.get('mse_ux', 0):.6f} | {v.get('mse_uy', 0):.6f} | "
                f"{v.get('mse_p', 0):.6f} | {v.get('physics_total', 0):.6f} |"
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("- Checkpoints: models/model_*.pt")
    lines.append("- Training history: models/training_history.pkl")
    lines.append("- Training summary: models/results_summary.json")
    lines.append("- Test outputs: models/test_outputs/")
    lines.append("")

    lines.append("## Notes")
    lines.append("This report is regenerated from current run artifacts to keep reproducibility after repository reconstruction.")

    return "\n".join(lines) + "\n"



def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from retraining artifacts")
    parser.add_argument("--summary", type=str, default="models/results_summary.json")
    parser.add_argument("--test-metrics", type=str, default="models/test_outputs/metrics.json")
    parser.add_argument("--out", type=str, default="report/CFD_AI_exam_report.md")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        raise FileNotFoundError(f"Missing summary file: {args.summary}")

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    test_metrics = {}
    if os.path.exists(args.test_metrics):
        with open(args.test_metrics, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report_text = build_markdown(summary, test_metrics)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
 
