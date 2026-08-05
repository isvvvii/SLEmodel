# run_sensitivity_analysis.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import ot
from scipy.stats import spearmanr

from slemodel import config as cfg
from slemodel.utils import train_classifier, get_propensity_scores, get_ot_coupling
from slemodel.models import DecoupledModel, PropensityScoreClassifier
from .interpretability import _get_global_fold_index, _pairwise_euclidean, _row_entropy

# --- 1. CONFIGURATION ---
FINAL_FIGURES_DIR = Path("./final_figures")
FINAL_FIGURES_DIR.mkdir(exist_ok=True)

SENSITIVITY_RESULTS_DIR = Path("./sensitivity_analysis_results")
SENSITIVITY_RESULTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Mass": "#B4AED3",
    "RNA": "#E8B2A6",
}

FONT_SIZES = {"title": 12, "label": 11, "tick": 10}

REG_PARAMS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

SEEDS = getattr(cfg, "SEEDS", [42, 7, 100, 123, 2025])

N_SPLITS = cfg.N_SPLITS_K_FOLD
N_REPEATS = cfg.N_REPEATS


# --- 2. HELPER FUNCTIONS ---
def set_publication_style():
    """
    NC-like plotting style with PDF text kept editable in Adobe Illustrator.
    """
    sns.set_style("ticks")
    plt.rcParams.update({
        # ---- font / text ----
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "Helvetica"],
        "font.size": FONT_SIZES["tick"],
        "axes.labelsize": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["tick"],
        "text.usetex": False,
        "axes.unicode_minus": False,

        # ---- editable vector export ----
        "pdf.fonttype": 42,          # keep text editable in Adobe Illustrator
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",
        "pdf.compression": 0,

        # ---- figure appearance ----
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        # ---- axes style ----
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _ps_space_transform(ps: np.ndarray, metric: str) -> np.ndarray:
    m = (metric or "sqeuclidean").lower().strip()
    if m == "hellinger":
        return np.sqrt(np.clip(ps, 0.0, 1.0))
    return ps


def _ps_cost_matrix(ps_t: np.ndarray, ps_s: np.ndarray, metric: str) -> np.ndarray:
    xt = _ps_space_transform(np.asarray(ps_t, dtype=np.float64), metric)
    xs = _ps_space_transform(np.asarray(ps_s, dtype=np.float64), metric)
    return ot.dist(xt, xs, metric="sqeuclidean")


def _savefig(fig, name, *, preview_png=True):
    """
    Save an Illustrator-editable vector PDF plus an optional PNG preview.
    """
    pdf_path = FINAL_FIGURES_DIR / f"{name}.pdf"
    png_path = FINAL_FIGURES_DIR / f"{name}.png"

    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    # Save vector PDF first (key output for Illustrator)
    fig.savefig(pdf_path, format="pdf", **common_kwargs)

    # Save PNG preview
    if preview_png:
        fig.savefig(png_path, format="png", dpi=600, **common_kwargs)

    plt.close(fig)


def _smd(a, b):
    """Calculate absolute standardized mean differences for 2D arrays."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    var_a = np.var(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mu_a)
    var_b = np.var(b, axis=0, ddof=1) if b.shape[0] > 1 else np.zeros_like(mu_b)
    sd_pool = np.sqrt((var_a + var_b) / 2)
    return np.abs((mu_a - mu_b) / (sd_pool + 1e-9))


def _compute_smd_metrics(ps_target_raw, ps_source_raw, G, labels_target, metric: str, n_classes=3):
    """
    Compute pre-/post-match SMD in the *same feature space* used by the OT cost metric.

    Key point (critical for Hellinger):
      - If metric == "hellinger", we compute SMD in sqrt(PS) space.
      - Post-match "matched PS" is computed as:  PS_matched_feat = G @ PS_source_feat
        where PS_source_feat = sqrt(PS_source) for Hellinger.
        (This matches how OT distance is defined; do NOT use sqrt(G @ PS_source) as a proxy.)

    Args:
        ps_target_raw: raw PS of target (gly) [N_target, K]
        ps_source_raw: raw PS of source (mass/rna) [N_source, K]
        G: row-stochastic OT coupling [N_target, N_source]
        labels_target: target labels [N_target]
        metric: OT cost metric ("sqeuclidean" or "hellinger")
        n_classes: number of classes K

    Returns:
        dict with overall and by_group SMD summaries (mean absolute SMD).
    """
    label_names = cfg.VisualizationConfig.LABEL_NAMES

    # ---- transform PS into the OT-cost feature space ----
    ps_target = _ps_space_transform(np.asarray(ps_target_raw, dtype=np.float64), metric)
    ps_source = _ps_space_transform(np.asarray(ps_source_raw, dtype=np.float64), metric)

    G = np.asarray(G, dtype=np.float64)
    ps_matched = G @ ps_source  # barycentric mapping in the *feature space*

    def _smd_abs_per_covariate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Absolute SMD per covariate (dimension-wise)."""
        mu_a = a.mean(axis=0)
        mu_b = b.mean(axis=0)
        var_a = np.var(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mu_a)
        var_b = np.var(b, axis=0, ddof=1) if b.shape[0] > 1 else np.zeros_like(mu_b)
        sd_pool = np.sqrt((var_a + var_b) / 2.0)
        return np.abs((mu_a - mu_b) / (sd_pool + 1e-9))

    # ---- overall SMD (target vs source; target vs matched) ----
    pre_smd_overall = float(np.mean(_smd_abs_per_covariate(ps_target, ps_source)))
    post_smd_overall = float(np.mean(_smd_abs_per_covariate(ps_target, ps_matched)))
    delta_smd_overall = post_smd_overall - pre_smd_overall

    results = {
        "overall": {
            "pre_smd": pre_smd_overall,
            "post_smd": post_smd_overall,
            "delta_smd": delta_smd_overall,
            "smd_space": f"ps_feature_space({metric})",
        },
        "by_group": {}
    }

    # ---- by-group SMD ----
    source_mean = ps_source.mean(axis=0)
    source_var = np.var(ps_source, axis=0, ddof=1) if ps_source.shape[0] > 1 else np.zeros_like(source_mean)

    labels_target = np.asarray(labels_target)
    for k in range(n_classes):
        mask = labels_target == k
        if mask.sum() == 0:
            continue

        group_name = label_names.get(k, f"Class_{k}")
        target_group = ps_target[mask]
        matched_group = ps_matched[mask]

        # Pre-match: group centroid vs source centroid (scaled by pooled SD)
        target_group_mean = target_group.mean(axis=0)
        target_group_var = np.var(target_group, axis=0, ddof=1) if target_group.shape[0] > 1 else np.zeros_like(target_group_mean)
        pooled_sd_pre = np.sqrt((target_group_var + source_var) / 2.0)
        pre_smd_group = float(np.mean(np.abs((target_group_mean - source_mean) / (pooled_sd_pre + 1e-9))))

        # Post-match: group distribution vs matched distribution (dimension-wise abs SMD)
        post_smd_group = float(np.mean(_smd_abs_per_covariate(target_group, matched_group)))
        delta_smd_group = post_smd_group - pre_smd_group

        results["by_group"][group_name] = {
            "pre_smd": pre_smd_group,
            "post_smd": post_smd_group,
            "delta_smd": delta_smd_group,
            "smd_space": f"ps_feature_space({metric})",
        }

    return results

# --- 3. ANALYSIS FUNCTION ---
def perform_sensitivity_analysis():
    print("Starting sensitivity analysis...")
    print(f"Testing reg values: {REG_PARAMS}")
    print(f"Seeds: {SEEDS}")

    gly_df = pd.read_csv(cfg.GLY_PATH)
    mass_df = pd.read_csv(cfg.MASS_PATH)
    rna_df = pd.read_csv(cfg.RNA_PATH)

    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    gly_y = gly_df.iloc[:, 1].map(label_map).values
    mass_y = mass_df.iloc[:, 1].map(label_map).values
    rna_y = rna_df.iloc[:, 1].map(label_map).values

    gly_x = gly_df.iloc[:, 2:].values
    mass_x = mass_df.iloc[:, 2:].values
    rna_x = rna_df.iloc[:, 2:].values

    # OT cost metrics (always defined; used for both OT and SMD space)
    metric_mass = getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean")
    metric_rna = getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean")

    all_results = []
    all_results_by_group = []

    for seed in SEEDS:
        print(f"\n========== Processing seed {seed} ==========")

        rskf_gly = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
        rskf_mass = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
        rskf_rna = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)

        mass_splits = list(rskf_mass.split(mass_x, mass_y))
        rna_splits = list(rskf_rna.split(rna_x, rna_y))

        for fold_idx, (gly_train_idx, gly_val_idx) in enumerate(rskf_gly.split(gly_x, gly_y)):
            fold_name = f"Repeat_1-Fold_{fold_idx + 1}"
            print(f"Seed {seed}: processing {fold_name}...")

            mass_train_idx, _ = mass_splits[fold_idx]
            rna_train_idx, _ = rna_splits[fold_idx]

            gly_scaler = StandardScaler().fit(gly_x[gly_train_idx])
            mass_scaler = StandardScaler().fit(mass_x[mass_train_idx])
            rna_scaler = StandardScaler().fit(rna_x[rna_train_idx])

            Xg_va_t = torch.tensor(gly_scaler.transform(gly_x[gly_val_idx]), dtype=torch.float32)
            Xg_tr_t = torch.tensor(gly_scaler.transform(gly_x[gly_train_idx]), dtype=torch.float32)
            Xm_tr_t = torch.tensor(mass_scaler.transform(mass_x[mass_train_idx]), dtype=torch.float32)
            Xr_tr_t = torch.tensor(rna_scaler.transform(rna_x[rna_train_idx]), dtype=torch.float32)

            yg_tr_t = torch.tensor(gly_y[gly_train_idx], dtype=torch.long)
            yg_va_t = torch.tensor(gly_y[gly_val_idx], dtype=torch.long)
            ym_tr_t = torch.tensor(mass_y[mass_train_idx], dtype=torch.long)
            yr_tr_t = torch.tensor(rna_y[rna_train_idx], dtype=torch.long)

            g_clf = train_classifier(
                PropensityScoreClassifier(Xg_tr_t.shape[1]),
                Xg_tr_t, yg_tr_t, cfg.DEVICE, epochs=cfg.EPOCHS_PS
            )
            m_clf = train_classifier(
                PropensityScoreClassifier(Xm_tr_t.shape[1]),
                Xm_tr_t, ym_tr_t, cfg.DEVICE, epochs=cfg.EPOCHS_PS
            )
            r_clf = train_classifier(
                PropensityScoreClassifier(Xr_tr_t.shape[1]),
                Xr_tr_t, yr_tr_t, cfg.DEVICE, epochs=cfg.EPOCHS_PS
            )

            ps_g_va = get_propensity_scores(g_clf, Xg_va_t, cfg.DEVICE).numpy()
            ps_m_tr = get_propensity_scores(m_clf, Xm_tr_t, cfg.DEVICE).numpy()
            ps_r_tr = get_propensity_scores(r_clf, Xr_tr_t, cfg.DEVICE).numpy()
            labels_va = yg_va_t.numpy()

            model_path = (
                Path(cfg.BASE_EXPERIMENT_DIR) / cfg.EXPERIMENT_TAG
                / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}"
                / fold_name / f"best_model_{fold_name}.pth"
            )
            has_model = model_path.exists()
            if has_model:
                model = DecoupledModel(gly_x.shape[1], mass_x.shape[1], rna_x.shape[1]).to(cfg.DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
                model.eval()

                with torch.no_grad():
                    zs_g, _ = model.gly_encoder(Xg_va_t.to(cfg.DEVICE))
                    zs_m, _ = model.mass_encoder(Xm_tr_t.to(cfg.DEVICE))
                    zs_r, _ = model.rna_encoder(Xr_tr_t.to(cfg.DEVICE))

                dist_m = _pairwise_euclidean(zs_g, zs_m)
                dist_r = _pairwise_euclidean(zs_g, zs_r)

                cost_m = _ps_cost_matrix(ps_g_va, ps_m_tr, metric_mass)
                cost_r = _ps_cost_matrix(ps_g_va, ps_r_tr, metric_rna)
            else:
                dist_m = dist_r = None
                cost_m = cost_r = None

            # sweep reg
            for reg in REG_PARAMS:
                Gm = get_ot_coupling(
                    torch.from_numpy(ps_g_va),
                    torch.from_numpy(ps_m_tr),
                    reg=reg,
                    method=getattr(cfg, "OT_METHOD_MASS", "standard"),
                    cost_metric=metric_mass,
                    purpose="eval",
                ).numpy()

                Gr = get_ot_coupling(
                    torch.from_numpy(ps_g_va),
                    torch.from_numpy(ps_r_tr),
                    reg=reg,
                    method=getattr(cfg, "OT_METHOD_RNA", "standard"),
                    cost_metric=metric_rna,
                    purpose="eval",
                ).numpy()

                # SMD metrics computed in OT-cost feature space
                metrics_m = _compute_smd_metrics(ps_g_va, ps_m_tr, Gm, labels_va, metric=metric_mass, n_classes=3)
                metrics_r = _compute_smd_metrics(ps_g_va, ps_r_tr, Gr, labels_va, metric=metric_rna, n_classes=3)

                # Cost-latent consistency (unchanged)
                if has_model:
                    rho_m, _ = spearmanr(cost_m.ravel(), dist_m.ravel())
                    rho_r, _ = spearmanr(cost_r.ravel(), dist_r.ravel())
                else:
                    rho_m, rho_r = np.nan, np.nan

                # Entropy (unchanged)
                entropy_m = float(np.mean(_row_entropy(Gm)))
                entropy_r = float(np.mean(_row_entropy(Gr)))

                # overall rows
                all_results.append({
                    "seed": seed, "reg": reg, "fold": fold_idx, "modality": "Mass",
                    "pre_smd": metrics_m["overall"]["pre_smd"],
                    "post_smd": metrics_m["overall"]["post_smd"],
                    "delta_smd": metrics_m["overall"]["delta_smd"],
                    "spearman_rho": rho_m,
                    "avg_entropy": entropy_m,
                    "smd_space": metrics_m["overall"].get("smd_space", f"ps_feature_space({metric_mass})"),
                    "cost_metric": metric_mass,
                })
                all_results.append({
                    "seed": seed, "reg": reg, "fold": fold_idx, "modality": "RNA",
                    "pre_smd": metrics_r["overall"]["pre_smd"],
                    "post_smd": metrics_r["overall"]["post_smd"],
                    "delta_smd": metrics_r["overall"]["delta_smd"],
                    "spearman_rho": rho_r,
                    "avg_entropy": entropy_r,
                    "smd_space": metrics_r["overall"].get("smd_space", f"ps_feature_space({metric_rna})"),
                    "cost_metric": metric_rna,
                })

                # by-group rows
                for group_name, group_metrics in metrics_m["by_group"].items():
                    all_results_by_group.append({
                        "seed": seed, "reg": reg, "fold": fold_idx,
                        "modality": "Mass", "group": group_name,
                        "pre_smd": group_metrics["pre_smd"],
                        "post_smd": group_metrics["post_smd"],
                        "delta_smd": group_metrics["delta_smd"],
                        "smd_space": group_metrics.get("smd_space", f"ps_feature_space({metric_mass})"),
                        "cost_metric": metric_mass,
                    })

                for group_name, group_metrics in metrics_r["by_group"].items():
                    all_results_by_group.append({
                        "seed": seed, "reg": reg, "fold": fold_idx,
                        "modality": "RNA", "group": group_name,
                        "pre_smd": group_metrics["pre_smd"],
                        "post_smd": group_metrics["post_smd"],
                        "delta_smd": group_metrics["delta_smd"],
                        "smd_space": group_metrics.get("smd_space", f"ps_feature_space({metric_rna})"),
                        "cost_metric": metric_rna,
                    })

    df_all = pd.DataFrame(all_results)
    df_by_group = pd.DataFrame(all_results_by_group)
    return df_all, df_by_group

def compute_summary_and_optimal(df_all: pd.DataFrame, df_by_group: pd.DataFrame):
    """Calculate summary statistics and identify the best regularization value."""
    if df_all.empty:
        return None, None, None

    summary_overall = (
        df_all.groupby(["reg", "modality"])
        .agg(
            pre_smd_mean=("pre_smd", "mean"), pre_smd_std=("pre_smd", "std"),
            post_smd_mean=("post_smd", "mean"), post_smd_std=("post_smd", "std"),
            delta_smd_mean=("delta_smd", "mean"), delta_smd_std=("delta_smd", "std"),
            spearman_rho_mean=("spearman_rho", "mean"), spearman_rho_std=("spearman_rho", "std"),
            avg_entropy_mean=("avg_entropy", "mean"), avg_entropy_std=("avg_entropy", "std"),
            n_samples=("delta_smd", "count"),
        ).reset_index()
    )

    summary_by_group = (
        df_by_group.groupby(["reg", "modality", "group"])
        .agg(
            pre_smd_mean=("pre_smd", "mean"), pre_smd_std=("pre_smd", "std"),
            post_smd_mean=("post_smd", "mean"), post_smd_std=("post_smd", "std"),
            delta_smd_mean=("delta_smd", "mean"), delta_smd_std=("delta_smd", "std"),
            n_samples=("delta_smd", "count"),
        ).reset_index()
    )

    optimal_results = []

    for modality in ["Mass", "RNA"]:
        mod_data = summary_by_group[summary_by_group["modality"] == modality]

        if mod_data.empty:
            continue

        unique_regs = mod_data["reg"].unique()

        best_reg = None
        best_score = float("inf")
        best_details = {}

        for reg in unique_regs:
            reg_data = mod_data[mod_data["reg"] == reg]

            if reg_data.empty:
                continue

            delta_smd_by_group = {}
            all_negative = True
            sum_delta = 0
            max_post_smd = 0

            for _, row in reg_data.iterrows():
                group = row["group"]
                delta = row["delta_smd_mean"]
                post = row["post_smd_mean"]

                if pd.isna(delta):
                    continue

                delta_smd_by_group[group] = delta
                sum_delta += delta
                max_post_smd = max(max_post_smd, post)
                if delta > 0:
                    all_negative = False

            if not delta_smd_by_group:
                continue

            worst_delta = max(delta_smd_by_group.values())
            penalty = max(0, worst_delta) * 10 if worst_delta > 0 else 0
            score = sum_delta + penalty

            if score < best_score:
                best_score = score
                best_reg = reg
                best_details = {
                    "all_groups_improved": all_negative,
                    "sum_delta_smd": sum_delta,
                    "worst_delta_smd": worst_delta,
                    "max_post_smd": max_post_smd,
                    **{f"delta_{g}": v for g, v in delta_smd_by_group.items()}
                }

        if best_reg is None:
            continue

        overall_at_best = summary_overall[
            (summary_overall["modality"] == modality) &
            (summary_overall["reg"] == best_reg)
        ]

        if not overall_at_best.empty:
            row = overall_at_best.iloc[0]
            optimal_results.append({
                "modality": modality,
                "optimal_reg": best_reg,
                "delta_smd_mean": row["delta_smd_mean"],
                "delta_smd_std": row["delta_smd_std"],
                "post_smd_mean": row["post_smd_mean"],
                "spearman_rho_mean": row["spearman_rho_mean"],
                "avg_entropy_mean": row["avg_entropy_mean"],
                **best_details
            })

    df_optimal = pd.DataFrame(optimal_results)

    return summary_overall, summary_by_group, df_optimal

def plot_sensitivity_analysis(df: pd.DataFrame, df_by_group: pd.DataFrame):
    """
    Plot the OT-regularization sensitivity analysis.
    Annotation boxes use relative panel coordinates for consistent placement.
    """
    if df.empty:
        print("Cannot plot: no data.")
        return

    df_seed = df.groupby(["seed", "reg", "modality"], as_index=False)[
        ["post_smd", "delta_smd", "spearman_rho", "avg_entropy"]
    ].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette_mod = {"Mass": PALETTE["Mass"], "RNA": PALETTE["RNA"]}

    reg_mass = getattr(cfg, "OT_REG_MASS", None)
    reg_rna = getattr(cfg, "OT_REG_RNA", None)

    Y_FRAC_MASS = 0.70
    Y_FRAC_RNA = 0.50

    def _add_selected_reg_lines(ax):
        """Draw the selected value and its annotation in relative panel coordinates."""
        ymin, ymax = ax.get_ylim()

        if reg_mass is not None:
            y_mass_data = ymin + Y_FRAC_MASS * (ymax - ymin)

            ax.axvline(reg_mass, ls="--", color=PALETTE["Mass"], lw=1.8, alpha=0.85, zorder=0)
            ax.text(
                reg_mass, y_mass_data,
                f"Mass\n(reg={reg_mass:.0e})",
                color=PALETTE["Mass"],
                ha="center", va="center",
                fontsize=9.5,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="white",
                    ec=PALETTE["Mass"],
                    lw=1.2,
                    alpha=0.92
                )
            )

        if reg_rna is not None:
            y_rna_data = ymin + Y_FRAC_RNA * (ymax - ymin)

            ax.axvline(reg_rna, ls="--", color=PALETTE["RNA"], lw=1.8, alpha=0.85, zorder=0)
            ax.text(
                reg_rna, y_rna_data,
                f"RNA\n(reg={reg_rna:.0e})",
                color=PALETTE["RNA"],
                ha="center", va="center",
                fontsize=9.5,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="white",
                    ec=PALETTE["RNA"],
                    lw=1.2,
                    alpha=0.92
                )
            )

    # ============================
    # (a) Post-match |SMD|
    # ============================
    sns.lineplot(data=df_seed, x="reg", y="post_smd", hue="modality", marker="o", ax=axes[0], palette=palette_mod)
    axes[0].set_xscale("log")
    axes[0].set_title("(a) Post-match |SMD|", fontsize=FONT_SIZES["title"])
    axes[0].set_xlabel("Regularization (log scale)")
    axes[0].set_ylabel("Mean |SMD| (lower is better)")
    axes[0].axhline(0.1, ls="--", color="0.6", lw=1.0, zorder=0)
    _add_selected_reg_lines(axes[0])
    axes[0].legend(frameon=False, title=None)

    # ============================
    # (b) Delta SMD
    # ============================
    sns.lineplot(data=df_seed, x="reg", y="delta_smd", hue="modality", marker="o", ax=axes[1], palette=palette_mod)
    axes[1].set_xscale("log")
    axes[1].set_title("(b) Δ|SMD| (Post − Pre)", fontsize=FONT_SIZES["title"])
    axes[1].set_xlabel("Regularization (log scale)")
    axes[1].set_ylabel("Δ|SMD| (negative is better)")
    axes[1].axhline(0, ls="--", color="red", lw=1.5, alpha=0.8, zorder=0)
    _add_selected_reg_lines(axes[1])
    axes[1].legend(frameon=False, title=None)

    # ============================
    # (c) Entropy
    # ============================
    sns.lineplot(data=df_seed, x="reg", y="avg_entropy", hue="modality", marker="o", ax=axes[2], palette=palette_mod)
    axes[2].set_xscale("log")
    axes[2].set_title("(c) Coupling entropy", fontsize=FONT_SIZES["title"])
    axes[2].set_xlabel("Regularization (log scale)")
    axes[2].set_ylabel("Mean row entropy")
    _add_selected_reg_lines(axes[2])
    axes[2].legend(frameon=False, title=None)

    fig.tight_layout()
    _savefig(fig, "ot_regularization_sensitivity")
    print("OT-regularization sensitivity plot saved.")

    if not df_by_group.empty:
        df_group_seed = df_by_group.groupby(["seed", "reg", "modality", "group"], as_index=False)["delta_smd"].mean()

        fig_group, axes_group = plt.subplots(1, 2, figsize=(14, 5))
        for idx, modality in enumerate(["Mass", "RNA"]):
            ax = axes_group[idx]
            mod_data = df_group_seed[df_group_seed["modality"] == modality]
            sns.lineplot(data=mod_data, x="reg", y="delta_smd", hue="group", marker="o", ax=ax)
            ax.set_xscale("log")
            ax.set_title(f"{modality}: Δ|SMD| by group", fontsize=FONT_SIZES["title"])
            ax.set_xlabel("Regularization (log scale)")
            ax.set_ylabel("Δ|SMD| (negative is better)")
            ax.axhline(0, ls="--", color="red", lw=1.5, alpha=0.8, zorder=0)

            ymin, ymax = ax.get_ylim()
            if modality == "Mass" and reg_mass is not None:
                y_mass_data = ymin + Y_FRAC_MASS * (ymax - ymin)
                ax.axvline(reg_mass, ls="--", color=PALETTE["Mass"], lw=1.6, alpha=0.9, zorder=0)
                ax.text(reg_mass, y_mass_data, f"reg={reg_mass:.0e}", color=PALETTE["Mass"],
                        ha="center", va="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["Mass"], lw=1.0, alpha=0.9))
            if modality == "RNA" and reg_rna is not None:
                y_rna_data = ymin + Y_FRAC_RNA * (ymax - ymin)
                ax.axvline(reg_rna, ls="--", color=PALETTE["RNA"], lw=1.6, alpha=0.9, zorder=0)
                ax.text(reg_rna, y_rna_data, f"reg={reg_rna:.0e}", color=PALETTE["RNA"],
                        ha="center", va="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["RNA"], lw=1.0, alpha=0.9))

            ax.legend(title="Group", frameon=False)

        fig_group.tight_layout()
        _savefig(fig_group, "ot_regularization_sensitivity_by_group")
        print("Group-specific OT-regularization sensitivity plot saved.")


# --- 4. MAIN ---
if __name__ == "__main__":
    set_publication_style()

    print("=" * 60)
    print("SENSITIVITY ANALYSIS FOR MODALITY-SPECIFIC REG SELECTION")
    print("=" * 60)

    df_all, df_by_group = perform_sensitivity_analysis()

    if df_all.empty:
        print("No results.")
    else:
        df_all.to_csv(SENSITIVITY_RESULTS_DIR / "sensitivity_raw_results.csv", index=False)
        df_by_group.to_csv(SENSITIVITY_RESULTS_DIR / "sensitivity_by_group_raw.csv", index=False)
        print(f"\nRaw results saved to {SENSITIVITY_RESULTS_DIR}")

        summary_overall, summary_by_group, df_optimal = compute_summary_and_optimal(df_all, df_by_group)

        if summary_overall is not None:
            summary_overall.to_csv(SENSITIVITY_RESULTS_DIR / "sensitivity_summary_overall.csv", index=False)
            summary_by_group.to_csv(SENSITIVITY_RESULTS_DIR / "sensitivity_summary_by_group.csv", index=False)
            df_optimal.to_csv(SENSITIVITY_RESULTS_DIR / "sensitivity_optimal_reg.csv", index=False)

            print("\n" + "=" * 60)
            print("OPTIMAL REG RECOMMENDATIONS")
            print("=" * 60)
            print(df_optimal.to_string(index=False))

            print("\n" + "-" * 60)
            print("RECOMMENDED CONFIG SETTINGS:")
            print("-" * 60)
            for _, row in df_optimal.iterrows():
                print(f"OT_REG_{row['modality'].upper()} = {row['optimal_reg']}")

        plot_sensitivity_analysis(df_all, df_by_group)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
