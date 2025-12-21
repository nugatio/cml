import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from simulate import THETA_GT


def main(csv_path):
    """
    Plots the distribution of estimators in two separate figures,
    matching the clean academic visual style.
    """
    data = pd.read_csv(csv_path)
    std_cf = data["theta_dml_cf"].std()
    theta_naive_norm = (data["theta_naive"] - THETA_GT) / std_cf
    theta_of_norm = (data["theta_dml_of"] - THETA_GT) / std_cf
    theta_cf_norm = (data["theta_dml_cf"] - THETA_GT) / std_cf

    plt.style.use("default")
    color_naive = "#E74C3C"
    color_of = "#F39C12"
    color_dml = "#2E86C1"
    color_ref = "#333333"
    x_grid = np.linspace(-15, 15, 500)
    y_gauss = norm.pdf(x_grid, 0, 1)

    def setup_figure(filename_base):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel(
            r"Normalized Scaled ($(\hat{\theta}_0-\theta_0)/\hat{\sigma}$)",
            fontsize=11,
            color="#333333",
        )
        ax.set_ylabel("Density", fontsize=11, color="#333333")

        ax.tick_params(axis="both", colors="black", labelsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.axvline(0, color=color_ref, linestyle="-", linewidth=0.8, alpha=0.3)

        return fig, ax

    def save_plot(filename_base):
        plt.tight_layout()
        plt.savefig(
            f"{filename_base}.pdf", bbox_inches="tight", dpi=500, facecolor="white"
        )
        plt.close()

    # Plot 1: Regularization Bias
    fig1, ax1 = setup_figure("regularization_bias")
    sns.kdeplot(
        theta_naive_norm,
        ax=ax1,
        fill=True,
        color=color_naive,
        alpha=0.3,
        linewidth=2,
        label="Naive (Non-Orthogonal)",
    )
    sns.kdeplot(
        theta_cf_norm,
        ax=ax1,
        fill=True,
        color=color_dml,
        alpha=0.3,
        linewidth=2,
        label="DML (Orthogonal)",
    )
    ax1.plot(
        x_grid,
        y_gauss,
        color=color_ref,
        linestyle="--",
        linewidth=1.5,
        label="Standard Normal",
        zorder=5,
    )
    ax1.set_xlim(-6, 6)
    ax1.legend(
        loc="upper left",
        fancybox=True,
        facecolor="white",
        framealpha=1,
        edgecolor="#333333",
        fontsize=9,
    )
    save_plot("regularization_bias")

    # Plot 2: Overfitting Bias
    fig2, ax2 = setup_figure("overfitting_bias")
    sns.kdeplot(
        theta_of_norm,
        ax=ax2,
        fill=True,
        color=color_of,
        alpha=0.3,
        linewidth=2,
        label="No-Split (Overfitting)",
    )
    sns.kdeplot(
        theta_cf_norm,
        ax=ax2,
        fill=True,
        color=color_dml,
        alpha=0.3,
        linewidth=2,
        label="Cross-Fit",
    )
    ax2.plot(
        x_grid,
        y_gauss,
        color=color_ref,
        linestyle="--",
        linewidth=1.5,
        label="Standard Normal",
        zorder=5,
    )
    ax2.set_xlim(-15, 5)
    ax2.legend(
        loc="upper left",
        fancybox=True,
        facecolor="white",
        framealpha=1,
        edgecolor="#333333",
        fontsize=9,
    )
    save_plot("overfitting_bias")


if __name__ == "__main__":
    main("results.csv")
