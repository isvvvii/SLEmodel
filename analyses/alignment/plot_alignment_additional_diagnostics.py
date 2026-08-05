# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import re
from matplotlib.ticker import FuncFormatter

BASE_RESULTS_DIR = Path("./interpretability_results")
FINAL_FIGURES_DIR = Path("./final_figures")
FINAL_FIGURES_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Mass": "#B4AED3",
    "RNA": "#E8B2A6",
    "Correct": "#B7D2CD",
    "Incorrect": "#E8B2A6",
}

PALETTE_QUALITY = {
    "Mass": "#C89B96",
    "RNA": "#A092AA",
}

FONT_SIZES = {"title": 9, "label": 8, "tick": 7, "legend": 7}

def set_style():
    """
    Global style for additional alignment diagnostics:
    - Helvetica/Arial/DejaVu Sans
    - font.size=8; title=9; ticks=7; legend=7
    - axes/ticks linewidth=0.6; tick size=3.5
    - PDF fonttype=42 for editable text
    """
    sns.set_style("ticks")
    plt.style.use("default")

    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes
        "font.size": 8,
        "axes.labelsize": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],

        # Lines
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # ---- Save ----
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.transparent": True,
    })

def _save(fig, name):
    png_path = FINAL_FIGURES_DIR / f"{name}.png"
    pdf_path = FINAL_FIGURES_DIR / f"{name}.pdf"

    FINAL_FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    try:
        fig.savefig(str(png_path), facecolor='white', dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {png_path} (exists: {png_path.exists()}, size: {png_path.stat().st_size if png_path.exists() else 0} bytes)")

        fig.savefig(str(pdf_path), facecolor='white', bbox_inches='tight')
        print(f"  [SAVED] {pdf_path} (exists: {pdf_path.exists()})")

    except Exception as e:
        print(f"  [ERROR] Failed to save {name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)

def collect_advanced(base_dir: Path):
    """
    Load data for the additional diagnostics and add to the performance table:
      - seed, fold
      - n_source_mass, n_source_rna
      - entropy_mass_norm = entropy_mass / log(n_source_mass)
      - entropy_rna_norm  = entropy_rna  / log(n_source_rna)

    Row entropy is normalized by its theoretical maximum, log(n_source), to permit
    comparison of coupling locality between modalities.
    """
    print(f"[DEBUG] Searching in: {base_dir.resolve()}")

    SEED_RE = re.compile(r"seed_(\d+)")
    FOLD_RE = re.compile(r"(Repeat_\d+-Fold_\d+)")

    def _parse_seed_fold(path: Path):
        seed, fold = None, None
        for part in path.parts:
            m = SEED_RE.search(part)
            if m:
                try:
                    seed = int(m.group(1))
                except Exception:
                    seed = None
            m = FOLD_RE.search(part)
            if m:
                fold = m.group(1)
        return seed, fold

    def _safe_log(x):
        x = np.asarray(x, dtype=float)
        return np.log(np.clip(x, 2.0, None))

    # ---------------------------
    # (1) per-sample performance
    # ---------------------------
    perf_files = list(base_dir.rglob("per_sample_performance.csv"))
    print(f"[DEBUG] Found {len(perf_files)} per_sample_performance.csv files")
    if perf_files:
        print(f"  Example: {perf_files[0]}")

    perf_dfs = []
    for f in perf_files:
        try:
            df = pd.read_csv(f)
            seed, fold = _parse_seed_fold(f)
            df["seed"] = seed
            df["fold"] = fold
            perf_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to read {f}: {e}")

    perf = pd.concat(perf_dfs, ignore_index=True) if perf_dfs else pd.DataFrame()
    print(f"[DEBUG] perf DataFrame shape: {perf.shape}")
    if not perf.empty:
        print(f"[DEBUG] perf columns: {perf.columns.tolist()}")

    # --------------------------------
    # (2) OT donor pool sizes (n_source)
    # --------------------------------
    ot_files = list(base_dir.rglob("ot_repair_stats.csv"))
    print(f"[DEBUG] Found {len(ot_files)} ot_repair_stats.csv files")
    if ot_files:
        print(f"  Example: {ot_files[0]}")

    ot_dfs = []
    for f in ot_files:
        try:
            df = pd.read_csv(f)
            seed, fold = _parse_seed_fold(f)
            if "seed" not in df.columns:
                df["seed"] = seed
            if "fold" not in df.columns:
                df["fold"] = fold
            ot_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to read {f}: {e}")

    ot = pd.concat(ot_dfs, ignore_index=True) if ot_dfs else pd.DataFrame()
    if not ot.empty:
        for c in ["seed", "fold", "modality", "n_source"]:
            if c not in ot.columns:
                print(f"[WARN] ot_repair_stats missing column: {c}")

    ot_pivot = pd.DataFrame()
    if not ot.empty and {"seed", "fold", "modality", "n_source"}.issubset(set(ot.columns)):
        tmp = ot.copy()
        tmp["n_source"] = pd.to_numeric(tmp["n_source"], errors="coerce")
        tmp = tmp[tmp["modality"].isin(["Mass", "RNA"])].copy()
        tmp = tmp.groupby(["seed", "fold", "modality"], as_index=False)["n_source"].median()
        ot_pivot = tmp.pivot_table(index=["seed", "fold"], columns="modality", values="n_source", aggfunc="median").reset_index()
        ot_pivot.rename(columns={"Mass": "n_source_mass", "RNA": "n_source_rna"}, inplace=True)

    if not perf.empty:
        if not ot_pivot.empty:
            perf = perf.merge(ot_pivot, on=["seed", "fold"], how="left")
        else:
            perf["n_source_mass"] = np.nan
            perf["n_source_rna"] = np.nan

        if not ot_pivot.empty:
            mass_med = float(np.nanmedian(ot_pivot["n_source_mass"].values)) if "n_source_mass" in ot_pivot.columns else np.nan
            rna_med = float(np.nanmedian(ot_pivot["n_source_rna"].values)) if "n_source_rna" in ot_pivot.columns else np.nan
        else:
            mass_med, rna_med = np.nan, np.nan

        if "n_source_mass" in perf.columns:
            perf["n_source_mass"] = pd.to_numeric(perf["n_source_mass"], errors="coerce").fillna(mass_med)
        if "n_source_rna" in perf.columns:
            perf["n_source_rna"] = pd.to_numeric(perf["n_source_rna"], errors="coerce").fillna(rna_med)

        if "entropy_mass" in perf.columns:
            perf["entropy_mass_norm"] = pd.to_numeric(perf["entropy_mass"], errors="coerce") / _safe_log(perf["n_source_mass"].values)
        if "entropy_rna" in perf.columns:
            perf["entropy_rna_norm"] = pd.to_numeric(perf["entropy_rna"], errors="coerce") / _safe_log(perf["n_source_rna"].values)

    # ---------------------------
    # ---------------------------
    ps_files = list(base_dir.rglob("ps_distributions.npz"))
    print(f"[DEBUG] Found {len(ps_files)} ps_distributions.npz files")
    if ps_files:
        print(f"  Example: {ps_files[0]}")

    mass_dfs, rna_dfs = [], []
    for f in ps_files:
        try:
            d = np.load(f)
            if f == ps_files[0]:
                print(f"[DEBUG] npz keys: {list(d.keys())}")

            n_classes = d['ps_gly_val'].shape[1]
            for i in range(n_classes):
                mass_dfs.append(pd.DataFrame({
                    'ps_value': np.concatenate([d['ps_gly_val'][:, i], d['ps_mass_pool'][:, i], d['ps_mass_matched'][:, i]]),
                    'group': (['Glycan (Target)'] * len(d['ps_gly_val']) +
                              ['Mass (Source)'] * len(d['ps_mass_pool']) +
                              ['Mass (Matched)'] * len(d['ps_mass_matched'])),
                    'class_idx': i
                }))
                rna_dfs.append(pd.DataFrame({
                    'ps_value': np.concatenate([d['ps_gly_val'][:, i], d['ps_rna_pool'][:, i], d['ps_rna_matched'][:, i]]),
                    'group': (['Glycan (Target)'] * len(d['ps_gly_val']) +
                              ['RNA (Source)'] * len(d['ps_rna_pool']) +
                              ['RNA (Matched)'] * len(d['ps_rna_matched'])),
                    'class_idx': i
                }))
        except Exception as e:
            print(f"  [WARN] Failed to read {f}: {e}")

    ps_mass = pd.concat(mass_dfs, ignore_index=True) if mass_dfs else pd.DataFrame()
    ps_rna = pd.concat(rna_dfs, ignore_index=True) if rna_dfs else pd.DataFrame()

    print(f"[DEBUG] ps_mass DataFrame shape: {ps_mass.shape}")
    print(f"[DEBUG] ps_rna DataFrame shape: {ps_rna.shape}")

    return {'performance': perf, 'ps_mass': ps_mass, 'ps_rna': ps_rna}

def plot_quality_vs_performance(df: pd.DataFrame):
    """
    Publication version:
    Two-panel layout (1×2):
      (a) Alignment uncertainty (entropy) vs prediction confidence
          - Mass + RNA overlayed in the same axis
      (b) Alignment uncertainty vs correctness
          - Mass + RNA shown together with outcome (Correct/Incorrect)

    Uses normalized entropy H/log(n_source) if available; otherwise falls back to raw entropy.
    """
    if df.empty:
        print("[SKIP] Quality vs Performance: No data")
        return

    required_cols = ["confidence", "is_correct"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[SKIP] Missing column: {c}")
            return

    use_mass = "entropy_mass_norm" if "entropy_mass_norm" in df.columns else "entropy_mass"
    use_rna = "entropy_rna_norm" if "entropy_rna_norm" in df.columns else "entropy_rna"

    if use_mass == "entropy_mass" or use_rna == "entropy_rna":
        print(
            "[WARN] Normalized entropy columns not found; falling back to raw entropy. "
            "Cross-modality uncertainty comparison requires ot_repair_stats.csv."
        )

    xlab = (
        r"Normalized coupling entropy ($H/\log n_{\mathrm{source}}$)"
        if ("norm" in use_mass and "norm" in use_rna)
        else "Coupling entropy"
    )

    # ---- Build a tidy long-form table: one row per sample per modality ----
    plot_df = df.copy()
    plot_df["confidence"] = pd.to_numeric(plot_df["confidence"], errors="coerce")
    plot_df["is_correct"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    plot_df[use_mass] = pd.to_numeric(plot_df[use_mass], errors="coerce")
    plot_df[use_rna] = pd.to_numeric(plot_df[use_rna], errors="coerce")

    long_df = pd.concat(
        [
            plot_df[["confidence", "is_correct", use_mass]].rename(columns={use_mass: "entropy"}).assign(modality="Mass"),
            plot_df[["confidence", "is_correct", use_rna]].rename(columns={use_rna: "entropy"}).assign(modality="RNA"),
        ],
        ignore_index=True,
    )

    long_df = long_df.dropna(subset=["confidence", "is_correct", "entropy"]).copy()
    long_df["outcome"] = long_df["is_correct"].replace({1.0: "Correct", 0.0: "Incorrect"})
    long_df = long_df[long_df["outcome"].isin(["Correct", "Incorrect"])].copy()

    if long_df.empty:
        print("[SKIP] Quality vs Performance: no valid rows after cleaning")
        return

    # Tick formatting and panel labels
    fmt_1dec = FuncFormatter(lambda x, pos: f"{x:.1f}")

    def _panel_label(ax, letter: str):
        ax.annotate(
            letter,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(-18, 18), textcoords="offset points",
            ha="left", va="top",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    # Compact figure canvas
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # =========================================================
    # (a) Uncertainty vs confidence (Mass + RNA in one axis)
    # =========================================================
    ax = axes[0]
    rho_text = []

    for modality in ["Mass", "RNA"]:
        sub = long_df[long_df["modality"] == modality]
        if sub.shape[0] < 6:
            continue

        # Spearman rho
        try:
            rho, _ = spearmanr(sub["entropy"].values, sub["confidence"].values)
            rho_text.append(f"{modality}: ρ={rho:.3f}")
        except Exception:
            rho_text.append(f"{modality}: ρ=N/A")

        sns.regplot(
            data=sub,
            x="entropy",
            y="confidence",
            ax=ax,
            ci=95,
            n_boot=300,
            truncate=False,
            scatter_kws={
                "alpha": 0.10,
                "s": 10,
                "color": PALETTE_QUALITY[modality],
                "edgecolors": "none",
            },
            line_kws={
                "color": PALETTE_QUALITY[modality],
                "lw": 2.0,
                "alpha": 0.95,
            },
        )

    ax.set_title("Uncertainty vs confidence", pad=8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Prediction confidence")
    ax.xaxis.set_major_formatter(fmt_1dec)
    ax.yaxis.set_major_formatter(fmt_1dec)
    ax.grid(True, ls="--", lw=0.6, alpha=0.18)
    sns.despine(ax=ax)

    if rho_text:
        ax.text(
            0.98,
            0.05,
            "\n".join(rho_text),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", lw=0.7, alpha=0.68),
        )

    # Legend as color key (clean, minimal)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=PALETTE_QUALITY["Mass"], lw=2.5, label="Mass"),
        Line2D([0], [0], color=PALETTE_QUALITY["RNA"], lw=2.5, label="RNA"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=2.2, borderaxespad=0.2)
    _panel_label(ax, "a")

    # =========================================================
    # (b) Uncertainty vs correctness (Mass + RNA in one axis)
    # =========================================================
    ax = axes[1]

    sns.violinplot(
        data=long_df,
        x="outcome",
        y="entropy",
        hue="modality",
        order=["Correct", "Incorrect"],
        hue_order=["Mass", "RNA"],
        palette={"Mass": PALETTE_QUALITY["Mass"], "RNA": PALETTE_QUALITY["RNA"]},
        cut=0,
        inner="quartile",
        dodge=True,
        linewidth=0.9,
        ax=ax,
    )

    ax.set_title("Uncertainty vs correctness", pad=8)
    ax.set_xlabel("Prediction outcome")
    ax.set_ylabel(xlab)

    ax.yaxis.set_major_formatter(fmt_1dec)
    ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.18)
    sns.despine(ax=ax)

    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.20),
          title=None, handlelength=2.2, borderaxespad=0.2)

    _panel_label(ax, "b")

    plt.tight_layout()
    _save(fig, "quality_vs_performance")


def plot_ps_distributions(df_mass: pd.DataFrame, df_rna: pd.DataFrame):
    """Plot propensity-score densities before and after alignment."""
    if df_mass.empty or df_rna.empty:
        print(f"[SKIP] PS KDE: mass empty={df_mass.empty}, rna empty={df_rna.empty}")
        return

    print("[INFO] Generating propensity-score density figure...")

    BIG_FONT = {
        'title': 18,
        'label': 16,
        'tick': 14,
        'legend': 14
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    def plot_modality(ax, df, modality_name, color_key):
        target = df[df["group"] == "Glycan (Target)"].copy()
        source = df[df["group"] == f"{modality_name} (Source)"].copy()
        matched = df[df["group"] == f"{modality_name} (Matched)"].copy()

        # Target: black solid line with grey fill
        if not target.empty:
            sns.kdeplot(
                data=target,
                x="ps_value",
                ax=ax,
                color="black",
                fill=True,
                alpha=0.20,
                linewidth=2.5,
                cut=0,
                clip=(0.0, 1.0),
                label="Glycan (Target)",
                zorder=1,
            )

        # Original source: coloured dashed line with light fill
        if not source.empty:
            sns.kdeplot(
                data=source,
                x="ps_value",
                ax=ax,
                color=PALETTE[color_key],
                linestyle="--",
                linewidth=2.5,
                fill=True,
                alpha=0.15,
                cut=0,
                clip=(0.0, 1.0),
                label=f"{modality_name} (Source)",
                zorder=2,
            )

        # OT-matched source: coloured solid line with darker fill
        if not matched.empty:
            sns.kdeplot(
                data=matched,
                x="ps_value",
                ax=ax,
                color=PALETTE[color_key],
                linestyle="-",
                linewidth=2.5,
                fill=True,
                alpha=0.30,
                cut=0,
                clip=(0.0, 1.0),
                label=f"{modality_name} (Matched)",
                zorder=3,
            )

        ax.set_title(
            f"Glycan–{modality_name} alignment",
            fontsize=BIG_FONT["title"],
            pad=15,
        )
        ax.set_xlabel(
            "Propensity score",
            fontsize=BIG_FONT["label"],
        )

        # Propensity scores are probabilities and must remain within 0–1
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=BIG_FONT["tick"],
            length=6,
            width=1.2,
        )

        sns.despine(ax=ax, top=True, right=True)

        ax.legend(
            fontsize=BIG_FONT["legend"],
            frameon=False,
            loc="upper right",
            handlelength=2.5,
        )

    plot_modality(axes[0], df_mass, "Mass", "Mass")
    axes[0].set_ylabel("Density", fontsize=BIG_FONT['label'])

    plot_modality(axes[1], df_rna, "RNA", "RNA")
    axes[1].set_ylabel("")

    plt.tight_layout()
    _save(fig, "ps_distributions")

def main():
    set_style()
    print("=" * 60)
    print("Generating additional alignment diagnostics")
    print("=" * 60)

    data = collect_advanced(BASE_RESULTS_DIR)

    print("\n" + "-" * 60)
    plot_quality_vs_performance(data['performance'])

    print("\n" + "-" * 60)
    plot_ps_distributions(data['ps_mass'], data['ps_rna'])

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FINAL_FIGURES_DIR.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
