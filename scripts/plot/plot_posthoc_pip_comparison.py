"""
Compare four model types side by side:
  POMO  /  POMO*  /  POMO*+PIP (post-hoc)  /  POMO*+PIP (jointly trained)

Produces triple-panel (easy/medium/hard) ins_infeasible plots for:
  - V1 post-decision (reveal=False)
  - V1 pre-decision  (reveal=True)
  - V2 gamma
  - V2 two_point
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_SWEEP   = PROJECT_ROOT / "results" / "csv" / "test_sweep_v1_v2.csv"
CSV_POSTHOC = PROJECT_ROOT / "results" / "csv" / "test_pomo_star_posthoc_pip.csv"
FIG_DIR     = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df_sweep   = pd.read_csv(CSV_SWEEP)
df_posthoc = pd.read_csv(CSV_POSTHOC)
df = pd.concat([df_sweep, df_posthoc], ignore_index=True)

HARDNESS = ["easy", "medium", "hard"]

MODEL_STYLES = [
    ("POMO",                 "POMO",                  "tab:blue",   "-",  1.6),
    ("POMO*",                "POMO_STAR",              "tab:orange", "-",  1.6),
    ("POMO*+PIP (post-hoc)", "POMO_STAR_PIP_posthoc", "tab:green",  "--", 1.8),
    ("POMO*+PIP (trained)",  "POMO_STAR_PIP",          "tab:green",  "-",  1.8),
]


def triple_panel(subsets_by_hardness, x_col, xlabel, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, h in zip(axes, HARDNESS):
        sub = subsets_by_hardness[h]
        for label, mt, color, ls, lw in MODEL_STYLES:
            g = sub[sub["model_type"] == mt].sort_values(x_col)
            if g.empty or g["ins_infeasible_pct"].isna().all():
                continue
            ax.plot(g[x_col], g["ins_infeasible_pct"],
                    label=label, color=color, linestyle=ls, linewidth=lw)
        ax.set_title(h)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelleft=True)
        if h == "easy":
            ax.set_ylabel("instance-level infeasible rate (%)")

    fig.suptitle(title, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    # collect from all axes to catch models that only appear in some panels
    seen = set()
    all_h, all_l = [], []
    for ax in axes:
        for h_i, l_i in zip(*ax.get_legend_handles_labels()):
            if l_i not in seen:
                seen.add(l_i)
                all_h.append(h_i)
                all_l.append(l_i)
    fig.legend(all_h, all_l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── V1 post-decision (reveal=False) ──────────────────────────────────────────
v1 = df[df["sweep"] == "v1"]
v1_post = v1[v1["reveal"] == False]
triple_panel(
    {h: v1_post[v1_post["hardness"] == h] for h in HARDNESS},
    x_col="delay_scale",
    xlabel="delay scale",
    title="V1 — post-decision noise",
    out_path=FIG_DIR / "posthoc_pip_v1_post_triple.png",
)

# ── V1 pre-decision (reveal=True) ────────────────────────────────────────────
v1_pre = v1[v1["reveal"] == True]
triple_panel(
    {h: v1_pre[v1_pre["hardness"] == h] for h in HARDNESS},
    x_col="delay_scale",
    xlabel="delay scale",
    title="V1 — pre-decision noise",
    out_path=FIG_DIR / "posthoc_pip_v1_pre_triple.png",
)

# ── V2 gamma ─────────────────────────────────────────────────────────────────
v2 = df[df["sweep"] == "v2"]
v2_gamma = v2[v2["noise_type"] == "gamma"]
triple_panel(
    {h: v2_gamma[v2_gamma["hardness"] == h] for h in HARDNESS},
    x_col="cv",
    xlabel="CV",
    title="V2 — gamma noise",
    out_path=FIG_DIR / "posthoc_pip_v2_gamma_triple.png",
)

# ── V2 two_point ─────────────────────────────────────────────────────────────
v2_tp = v2[v2["noise_type"] == "two_point"]
triple_panel(
    {h: v2_tp[v2_tp["hardness"] == h] for h in HARDNESS},
    x_col="cv",
    xlabel="CV",
    title="V2 — two-point noise",
    out_path=FIG_DIR / "posthoc_pip_v2_two_point_triple.png",
)
