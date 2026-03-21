import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../../results/csv/test_tsptw_on_stsptw_dw_sweep.csv"
df = pd.read_csv(csv_path)

hardness_list = ["easy", "medium", "hard"]

for h in hardness_list:
    sub = df[(df["hardness"] == h) & df["ins_infeasible_pct"].notna()]

    plt.figure(figsize=(6, 4))

    for label, mt, color in [
        ("POMO", "POMO", "tab:blue"),
        ("POMO*", "POMO_STAR", "tab:orange"),
        ("POMO*+PIP", "POMO_STAR_PIP", "tab:green"),
    ]:
        g = sub[sub["model_type"] == mt].sort_values("delay_scale")
        if g.empty:
            continue
        plt.plot(
            g["delay_scale"],
            g["ins_infeasible_pct"],
            label=label,
            linewidth=1.8,
            color=color,
        )

    plt.xlabel("delay weight")
    plt.ylabel("instance-level infeasible rate (%)")
    plt.title(f"TSPTW models on STSPTW ({h})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = f"../../results/figures/tsptw_on_stsptw_ins_infeasible_{h}.png"
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    plt.close()