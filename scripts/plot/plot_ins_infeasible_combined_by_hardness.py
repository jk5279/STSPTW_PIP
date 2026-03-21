import pandas as pd
import matplotlib.pyplot as plt


def main():
    sts_csv = "../../results/csv/test_stsptw_matched.csv"
    ts_csv = "../../results/csv/test_tsptw_on_stsptw_dw_sweep.csv"

    df_sts = pd.read_csv(sts_csv)
    df_ts = pd.read_csv(ts_csv)

    # Rename delay_scale to delay_weight to align with STSPTW matched CSV
    df_ts = df_ts.rename(columns={"delay_scale": "delay_weight"})

    hardness_list = ["easy", "medium", "hard"]
    titles = {"easy": "easy", "medium": "medium", "hard": "hard"}

    for h in hardness_list:
        sub_sts = df_sts[(df_sts["hardness"] == h) & df_sts["ins_infeasible_pct"].notna()]
        sub_ts = df_ts[(df_ts["hardness"] == h) & df_ts["ins_infeasible_pct"].notna()]

        fig, ax = plt.subplots(figsize=(6, 4))

        for label, mt, color in [
            ("POMO", "POMO", "tab:blue"),
            ("POMO*", "POMO_STAR", "tab:orange"),
            ("POMO*+PIP", "POMO_STAR_PIP", "tab:green"),
        ]:
            g_sts = sub_sts[sub_sts["model_type"] == mt].sort_values("delay_weight")
            g_ts = sub_ts[sub_ts["model_type"] == mt].sort_values("delay_weight")

            if not g_sts.empty:
                ax.plot(
                    g_sts["delay_weight"],
                    g_sts["ins_infeasible_pct"],
                    label=f"{label} (STSPTW train)",
                    linewidth=1.8,
                    color=color,
                    linestyle="-",
                )

            if not g_ts.empty:
                ax.plot(
                    g_ts["delay_weight"],
                    g_ts["ins_infeasible_pct"],
                    label=f"{label} (TSPTW train)",
                    linewidth=1.8,
                    color=color,
                    linestyle="--",
                )

        ax.set_title(titles[h])
        ax.set_xlabel("delay weight")
        ax.set_ylabel("instance-level infeasible rate (%)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelleft=True)
        ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        out_name = f"../../results/figures/ins_infeasible_{h}_combined.png"
        plt.savefig(out_name, dpi=200)
        print(f"Saved {out_name}")
        plt.close(fig)


if __name__ == "__main__":
    main()

