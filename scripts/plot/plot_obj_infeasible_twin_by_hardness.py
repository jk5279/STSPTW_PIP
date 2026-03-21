import pandas as pd
import matplotlib.pyplot as plt


def main():
    sts_csv = "../../results/csv/test_stsptw_matched.csv"
    ts_csv = "../../results/csv/test_tsptw_on_stsptw_dw_sweep.csv"

    df_sts = pd.read_csv(sts_csv)
    df_ts = pd.read_csv(ts_csv)

    # unify column name
    df_ts = df_ts.rename(columns={"delay_scale": "delay_weight"})

    hardness_list = ["easy", "medium", "hard"]
    titles = {"easy": "easy", "medium": "medium", "hard": "hard"}

    for h in hardness_list:
        sub_sts = df_sts[(df_sts["hardness"] == h)]
        sub_ts = df_ts[(df_ts["hardness"] == h)]

        fig, ax_left = plt.subplots(figsize=(7, 4))
        ax_right = ax_left.twinx()

        # left y: infeasible (%), right y: objective (aug_score)
        for label, mt, color in [
            ("POMO", "POMO", "tab:blue"),
            ("POMO*", "POMO_STAR", "tab:orange"),
            ("POMO*+PIP", "POMO_STAR_PIP", "tab:green"),
        ]:
            g_sts = sub_sts[sub_sts["model_type"] == mt].sort_values("delay_weight")
            g_ts = sub_ts[sub_ts["model_type"] == mt].sort_values("delay_weight")

            # infeasible rate (left y)
            if g_sts["ins_infeasible_pct"].notna().any():
                ax_left.plot(
                    g_sts["delay_weight"],
                    g_sts["ins_infeasible_pct"],
                    label=f"{label} infeas (STSPTW)",
                    color=color,
                    linestyle="-",
                    linewidth=1.6,
                )
            if g_ts["ins_infeasible_pct"].notna().any():
                ax_left.plot(
                    g_ts["delay_weight"],
                    g_ts["ins_infeasible_pct"],
                    label=f"{label} infeas (TSPTW)",
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                )

            # objective (right y)
            if g_sts["aug_score"].notna().any():
                ax_right.plot(
                    g_sts["delay_weight"],
                    g_sts["aug_score"],
                    label=f"{label} obj (STSPTW)",
                    color=color,
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.5,
                )
            if g_ts["aug_score"].notna().any():
                ax_right.plot(
                    g_ts["delay_weight"],
                    g_ts["aug_score"],
                    label=f"{label} obj (TSPTW)",
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

        ax_left.set_title(titles[h])
        ax_left.set_xlabel("delay weight")
        ax_left.set_ylabel("instance-level infeasible rate (%)")
        ax_right.set_ylabel("objective (aug_score)")

        ax_left.grid(True, alpha=0.3)
        ax_left.tick_params(labelleft=True)

        # Combine legends from both axes
        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        handles = handles_left + handles_right
        labels = labels_left + labels_right
        ax_left.legend(handles, labels, loc="best", fontsize=7)

        plt.tight_layout()
        out_name = f"../../results/figures/obj_infeas_{h}_twin.png"
        plt.savefig(out_name, dpi=200)
        print(f"Saved {out_name}")
        plt.close(fig)


if __name__ == "__main__":
    main()

