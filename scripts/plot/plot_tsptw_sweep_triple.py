import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../../results/csv/test_tsptw_on_stsptw_dw_sweep.csv"
df = pd.read_csv(csv_path)

hardness_list = ["easy", "medium", "hard"]
titles = {"easy": "easy", "medium": "medium", "hard": "hard"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, h in zip(axes, hardness_list):
    sub = df[(df["hardness"] == h) & df["ins_infeasible_pct"].notna()]

    for label, mt, color in [
        ("POMO", "POMO", "tab:blue"),
        ("POMO*", "POMO_STAR", "tab:orange"),
        ("POMO*+PIP", "POMO_STAR_PIP", "tab:green"),
    ]:
        g = sub[sub["model_type"] == mt].sort_values("delay_scale")
        if g.empty:
            continue
        ax.plot(
            g["delay_scale"],
            g["ins_infeasible_pct"],
            label=label,
            linewidth=1.8,
            color=color,
        )

    ax.set_title(titles[h])
    ax.set_xlabel("delay weight")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelleft=True)
    if h == "easy":
        ax.set_ylabel("instance-level infeasible rate (%)")

# Single shared legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig("../../results/figures/tsptw_on_stsptw_ins_infeasible_triple.png", dpi=200)
plt.show()