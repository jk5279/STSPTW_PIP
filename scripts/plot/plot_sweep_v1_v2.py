"""
Visualize test_sweep_v1_v2.csv results.

V1: x=delay_scale, lines=model_type × reveal(pre/post), per hardness
V2: x=cv,          lines=model_type × noise_type(gamma/two_point), per hardness

Produces (all saved to results/figures/):
  sweep_v1_ins_infeasible_triple.png
  sweep_v1_obj_infeas_{h}_twin.png   (easy/medium/hard)
  sweep_v2_ins_infeasible_gamma_triple.png
  sweep_v2_ins_infeasible_two_point_triple.png
  sweep_v2_obj_infeas_{h}_twin.png   (easy/medium/hard)
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "results" / "csv" / "test_sweep_v1_v2.csv"
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df_v1 = df[df["sweep"] == "v1"].copy()
df_v2 = df[df["sweep"] == "v2"].copy()

HARDNESS = ["easy", "medium", "hard"]
MODEL_STYLES = [
    ("POMO",        "POMO",         "tab:blue"),
    ("POMO*",       "POMO_STAR",    "tab:orange"),
    ("POMO*+PIP",   "POMO_STAR_PIP","tab:green"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _twin_plot(ax_left, ax_right, x, infeas, obj, label, color, linestyle):
    if infeas is not None:
        ax_left.plot(x, infeas, label=f"{label} infeas",
                     color=color, linestyle=linestyle, linewidth=1.8)
    if obj is not None:
        ax_right.plot(x, obj, label=f"{label} obj",
                      color=color, linestyle=linestyle, linewidth=1.0, alpha=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# V1 plots
# ══════════════════════════════════════════════════════════════════════════════

# ── V1 triple: ins_infeasible vs delay_scale, solid=post / dashed=pre ────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h in zip(axes, HARDNESS):
    sub = df_v1[df_v1["hardness"] == h]
    for label, mt, color in MODEL_STYLES:
        g_post = sub[(sub["model_type"] == mt) & (sub["reveal"] == False)].sort_values("delay_scale")
        g_pre  = sub[(sub["model_type"] == mt) & (sub["reveal"] == True)].sort_values("delay_scale")
        if not g_post.empty:
            ax.plot(g_post["delay_scale"], g_post["ins_infeasible_pct"],
                    label=f"{label} (post)", color=color, linestyle="-", linewidth=1.8)
        if not g_pre.empty:
            ax.plot(g_pre["delay_scale"], g_pre["ins_infeasible_pct"],
                    label=f"{label} (pre)", color=color, linestyle="--", linewidth=1.8)
    ax.set_title(h)
    ax.set_xlabel("delay scale")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelleft=True)
    if h == "easy":
        ax.set_ylabel("instance-level infeasible rate (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
out = FIG_DIR / "sweep_v1_ins_infeasible_triple.png"
plt.savefig(out, dpi=200)
print(f"Saved {out}")
plt.close(fig)


# ── V1 twin-axis: infeasible (left) + obj (right) vs delay_scale per hardness ─
for h in HARDNESS:
    sub = df_v1[df_v1["hardness"] == h]
    fig, ax_l = plt.subplots(figsize=(7, 4))
    ax_r = ax_l.twinx()
    for label, mt, color in MODEL_STYLES:
        g_post = sub[(sub["model_type"] == mt) & (sub["reveal"] == False)].sort_values("delay_scale")
        g_pre  = sub[(sub["model_type"] == mt) & (sub["reveal"] == True)].sort_values("delay_scale")
        for g, ls, tag in [(g_post, "-", "post"), (g_pre, "--", "pre")]:
            if g.empty:
                continue
            infeas = g["ins_infeasible_pct"].values if g["ins_infeasible_pct"].notna().any() else None
            obj    = g["aug_score"].values          if g["aug_score"].notna().any()          else None
            _twin_plot(ax_l, ax_r, g["delay_scale"].values,
                       infeas, obj, f"{label} ({tag})", color, ls)
    ax_l.set_title(h)
    ax_l.set_xlabel("delay scale")
    ax_l.set_ylabel("instance-level infeasible rate (%)")
    ax_r.set_ylabel("objective (aug_score)")
    ax_l.grid(True, alpha=0.3)
    h_l, lb_l = ax_l.get_legend_handles_labels()
    h_r, lb_r = ax_r.get_legend_handles_labels()
    ax_l.legend(h_l + h_r, lb_l + lb_r, loc="best", fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / f"sweep_v1_obj_infeas_{h}_twin.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# V2 plots
# ══════════════════════════════════════════════════════════════════════════════

# ── V2 triple per noise_type: ins_infeasible vs cv ───────────────────────────
for noise in ["gamma", "two_point"]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, h in zip(axes, HARDNESS):
        sub = df_v2[(df_v2["hardness"] == h) & (df_v2["noise_type"] == noise)]
        for label, mt, color in MODEL_STYLES:
            g = sub[sub["model_type"] == mt].sort_values("cv")
            if g.empty:
                continue
            ax.plot(g["cv"], g["ins_infeasible_pct"],
                    label=label, color=color, linewidth=1.8)
        ax.set_title(h)
        ax.set_xlabel("CV")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelleft=True)
        if h == "easy":
            ax.set_ylabel("instance-level infeasible rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"V2 — noise: {noise}", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    out = FIG_DIR / f"sweep_v2_ins_infeasible_{noise}_triple.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── V2 twin-axis: infeasible (left) + obj (right) vs cv per hardness ─────────
# lines: model × noise_type (solid=gamma, dashed=two_point)
for h in HARDNESS:
    sub = df_v2[df_v2["hardness"] == h]
    fig, ax_l = plt.subplots(figsize=(7, 4))
    ax_r = ax_l.twinx()
    for label, mt, color in MODEL_STYLES:
        for noise, ls in [("gamma", "-"), ("two_point", "--")]:
            g = sub[(sub["model_type"] == mt) & (sub["noise_type"] == noise)].sort_values("cv")
            if g.empty:
                continue
            infeas = g["ins_infeasible_pct"].values if g["ins_infeasible_pct"].notna().any() else None
            obj    = g["aug_score"].values          if g["aug_score"].notna().any()          else None
            _twin_plot(ax_l, ax_r, g["cv"].values,
                       infeas, obj, f"{label} ({noise})", color, ls)
    ax_l.set_title(h)
    ax_l.set_xlabel("CV")
    ax_l.set_ylabel("instance-level infeasible rate (%)")
    ax_r.set_ylabel("objective (aug_score)")
    ax_l.grid(True, alpha=0.3)
    h_l, lb_l = ax_l.get_legend_handles_labels()
    h_r, lb_r = ax_r.get_legend_handles_labels()
    ax_l.legend(h_l + h_r, lb_l + lb_r, loc="best", fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / f"sweep_v2_obj_infeas_{h}_twin.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.close(fig)
