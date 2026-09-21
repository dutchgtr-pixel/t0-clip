"""Create a publication-friendly SVG from reviewed aggregate benchmark JSON."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if any(summary["study_kind"] != "synthetic_smoke" for summary in summaries):
        raise ValueError("This demonstration plot is restricted to synthetic smoke summaries")
    names = list(summaries[0]["results"])
    labels = {"km": "Kaplan–Meier", "coxph": "Cox PH", "rsf": "Survival forest",
              "gbsa": "Boosted trees", "mlp": "Survival MLP", "perceiver_moe": "Perceiver mixture"}
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(1, len(summaries), figsize=(6*len(summaries), 4.8), squeeze=False)
    for ax, summary, path in zip(axes.flat, summaries, args.summaries):
        values = [summary["results"][name]["mean_test_ibs"] for name in names]
        ax.barh(np.arange(len(names)), values, color=["#26374A" if name == "coxph" else "#4C8791" for name in names], height=.62)
        ax.set_yticks(np.arange(len(names)), [labels[name] for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.23)
        ax.set_xlabel("Integrated Brier score · lower is better")
        ax.set_title("PH fixture" if "ph_smoke" in path.as_posix() else "Nonlinear / non-PH fixture", loc="left", pad=14)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for i, value in enumerate(values):
            ax.text(value+.004, i, f"{value:.4f}", va="center", fontsize=9)
    fig.suptitle("Synthetic reproducibility demonstration", x=.03, y=.98, ha="left", fontsize=17, fontweight="bold")
    fig.text(.03, .015, "1,200 artificial entities per fixture · fixed settings · one model seed\nNot evidence of effectiveness or superiority on real marketplace data", fontsize=9, color="#4A5560")
    fig.tight_layout(rect=(0, .1, 1, .92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Creator": "MarketNeural aggregate plotting script", "Date": None})
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
