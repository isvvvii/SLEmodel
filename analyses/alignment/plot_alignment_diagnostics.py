# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from pathlib import Path
import re
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as patches

BASE_RESULTS_DIR = Path("./interpretability_results")
FINAL_FIGURES_DIR = Path("./final_figures")
FINAL_FIGURES_DIR.mkdir(exist_ok=True)

PALETTE = {
    'Mass': '#B4AED3',
    'RNA': '#E8B2A6',
    'PositiveDelta': '#B7D2CD',
    'NegativeDelta': '#E8B2A6',
    'NegCtrl': '0.65',
    'Correct': '#B7D2CD',
    'Incorrect': '#E8B2A6'
}

FONT_SIZES = {
    "title": 9,     # panel title
    "label": 8,     # axis label
    "tick": 7,      # tick labels
    "legend": 7,    # legend text
    "panel": 10,    # panel letter (a–f), bold
}

def set_style():
    """
    Compact publication typography.
    Only controls typography/axes; does NOT change plot geometry or color choices.
    """
    sns.set_style("ticks")
    plt.style.use("default")

    plt.rcParams.update({
        # ---- Fonts ----
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes
        "font.size": FONT_SIZES["label"],          # base
        "axes.labelsize": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],

        # Axes and ticks
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # ---- Save ----
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
    })

def _mean_t_ci(vals):
    vals = np.asarray(vals, dtype=float)[~np.isnan(np.asarray(vals, dtype=float))]
    n = len(vals)
    if n == 0: return np.nan, np.nan, np.nan
    m = np.mean(vals)
    if n < 2: return m, m, m
    se = np.std(vals, ddof=1) / np.sqrt(n)
    ci = se * t.ppf(0.975, n - 1)
    return m, m - ci, m + ci

def _scan(pattern): return list(BASE_RESULTS_DIR.glob(pattern))

def collect_all_results():
    seed_pat = re.compile(r"seed_(\d+)"); fold_pat = re.compile(r"(Repeat_\d+-Fold_\d+)")
    def _read(pattern):
        dfs = []
        for fp in _scan(pattern):
            try:
                df = pd.read_csv(fp)
                s = seed_pat.search(str(fp)); f = fold_pat.search(str(fp))
                if s: df['seed'] = int(s.group(1))
                if f: df['fold'] = f.group(1)
                dfs.append(df)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    data = {
        'smd': _read("**/smd_summary.csv"),
        'global': _read("**/global_alignment_metrics.csv"),
        'within_pair': _read("**/within_pair_summary.csv"),
        'cost_latent': _read("**/cost_latent_consistency.csv"),
        'locality_mass': _read("**/locality_mass.csv"),
        'locality_rna': _read("**/locality_rna.csv"),
        'hit_at_k': _read("**/hit_at_k.csv"),
        'cka': _read("**/cka_consistency.csv"),
        'cost_latent_neg': _read("**/cost_latent_consistency_negcontrol.csv"),
        'ot_repair': _read("**/ot_repair_stats.csv"),

        'class_flow_mass': _scan("**/class_flow_matrix_mass.csv"),
        'class_flow_rna' : _scan("**/class_flow_matrix_rna.csv"),
    }
    return data

def _save(fig, name):
    fig.savefig(FINAL_FIGURES_DIR / f"{name}.png", facecolor='white')
    fig.savefig(FINAL_FIGURES_DIR / f"{name}.pdf", facecolor='white')
    plt.close(fig)

class SplitColorPatch(patches.Patch):
    """Create a two-color rectangular legend handle."""
    def __init__(self, left_color, right_color, **kwargs):
        super().__init__(**kwargs)
        self.left_color = left_color
        self.right_color = right_color

    def __copy__(self):
        return SplitColorPatch(self.left_color, self.right_color)

def plot_smd_balance(df_smd: pd.DataFrame, name: str):
    """
    Love plot（Balance in PS space）：
    Connector opacity is matched to the global-distance panel (alpha=0.45).
    """
    if df_smd.empty:
        print("Love plot skipped.")
        return

    def _agg(g):
        vals = np.asarray(g, dtype=float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        if n < 2:
            ci = 0.0
        else:
            se = float(np.std(vals, ddof=1) / np.sqrt(n))
            ci = float(se * t.ppf(0.975, n - 1))
        return pd.Series({'mean': mean, 'ci': ci})

    pre = df_smd.groupby(['modality', 'group'])['pre_match_smd'].apply(_agg).unstack()
    post = df_smd.groupby(['modality', 'group'])['post_match_smd'].apply(_agg).unstack()
    delta = df_smd.groupby(['modality', 'group'])['delta_smd'].apply(_agg).unstack()
    agg = (pre.add_prefix('pre_').join(post.add_prefix('post_')).join(delta.add_prefix('delta_'))).reset_index()

    order = ['Stable', 'Active', 'Control']
    ymap = {g: i for i, g in enumerate(order)}
    agg['y'] = agg['group'].map(ymap)

    delta_mass_mean = df_smd[df_smd['modality'] == 'Mass']['delta_smd'].mean()
    delta_rna_mean = df_smd[df_smd['modality'] == 'RNA']['delta_smd'].mean()

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.0))
    ax.set_title("Balance in PS space (|SMD|)", fontsize=FONT_SIZES['title'] + 1, pad=14)

    y_offset = 0.15
    for _, row in agg.iterrows():
        mod = row['modality']
        y = row['y'] + (y_offset if mod == 'Mass' else -y_offset)
        color = PALETTE[mod]

        ax.plot([row['pre_mean'], row['post_mean']], [y, y], color='0.70', lw=1.1, alpha=0.45, zorder=1)

        ax.errorbar(row['pre_mean'], y, xerr=row['pre_ci'], fmt='o', ms=7, mfc='white',
                    mec=color, ecolor=color, capsize=3, zorder=3)
        ax.errorbar(row['post_mean'], y, xerr=row['post_ci'], fmt='s', ms=6, mfc=color,
                    mec=color, ecolor=color, capsize=3, zorder=3)

    ax.axvline(0.1, ls='--', color='0.55', lw=1.0, zorder=0)
    ax.axvline(0.05, ls=':', color='0.55', lw=1.0, zorder=0)

    ax.set_xlabel("|SMD| (mean ± 95% CI)")
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels([f"PS({g})" for g in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.5, -0.5)

    handles = [
        mlines.Line2D([], [], color='0.35', marker='o', ms=7, mfc='white', ls='none', label='Pre-match'),
        mlines.Line2D([], [], color='0.35', marker='s', ms=6, ls='none', label='Post-match'),
        mlines.Line2D([], [], color=PALETTE['Mass'], marker='s', ls='none',
                      label=f"Mass (Δ|SMD|: {delta_mass_mean:+.3f})"),
        mlines.Line2D([], [], color=PALETTE['RNA'], marker='s', ls='none',
                      label=f"RNA (Δ|SMD|: {delta_rna_mean:+.3f})"),
    ]
    ax.legend(handles=handles, frameon=False, loc='upper right', bbox_to_anchor=(1.02, 0.98), ncol=1)

    plt.tight_layout()
    _save(fig, name)

def plot_global_distances(df: pd.DataFrame, name: str):
    """Plot pre- and post-alignment global distance measures."""
    if df.empty:
        print("Global distance skipped.")
        return

    metrics = [
        ("emd", "EMD"),
        ("mmd2", r"MMD$^2$"),
        ("sink", "Sinkhorn\ndiv."),
    ]
    modalities = ["Mass", "RNA"]

    colors = {
        "Mass": {"pre": "#F5F3F9", "post": PALETTE["Mass"]},
        "RNA":  {"pre": "#FDF6F3", "post": PALETTE["RNA"]},
    }

    rows = []
    for mod in modalities:
        d = df[df["modality"] == mod].copy()
        if d.empty:
            continue
        for key, label in metrics:
            pre_col, post_col = f"{key}_pre", f"{key}_post"
            if pre_col not in d.columns or post_col not in d.columns:
                continue

            pre_vals = d[pre_col].astype(float).values
            post_vals = d[post_col].astype(float).values
            pre_m, pre_lo, pre_hi = _mean_t_ci(pre_vals)
            post_m, post_lo, post_hi = _mean_t_ci(post_vals)

            rows.append({
                "modality": mod,
                "metric_key": key,
                "metric_label": label,
                "pre_mean": pre_m, "pre_lo": pre_lo, "pre_hi": pre_hi,
                "post_mean": post_m, "post_lo": post_lo, "post_hi": post_hi,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("Global distance skipped (no usable columns).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"wspace": 0.20})
    fig.suptitle("Global distribution distance in PS space", fontsize=FONT_SIZES['title'] + 1, y=1.00)

    for ax_idx, mod in enumerate(modalities):
        ax = axes[ax_idx]
        sub = plot_df[plot_df["modality"] == mod].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["order"] = sub["metric_key"].map({k: i for i, (k, _) in enumerate(metrics)})
        sub.sort_values("order", inplace=True)

        y = np.arange(len(sub))

        if ax_idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(sub["metric_label"].tolist())
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])

        ax.invert_yaxis()

        for i, r in enumerate(sub.itertuples(index=False)):
            ax.plot([r.pre_mean, r.post_mean], [i, i], color="0.70", lw=1.2, alpha=0.45, zorder=1)

        ax.errorbar(
            sub["pre_mean"].values, y,
            xerr=[sub["pre_mean"].values - sub["pre_lo"].values, sub["pre_hi"].values - sub["pre_mean"].values],
            fmt="o", ms=6.8, mfc="white", mec=colors[mod]["post"], ecolor=colors[mod]["post"],
            elinewidth=1.2, capsize=2.5, zorder=3, label="Pre-match"
        )

        ax.errorbar(
            sub["post_mean"].values, y,
            xerr=[sub["post_mean"].values - sub["post_lo"].values, sub["post_hi"].values - sub["post_mean"].values],
            fmt="s", ms=6.2, mfc=colors[mod]["post"], mec=colors[mod]["post"], ecolor=colors[mod]["post"],
            elinewidth=1.2, capsize=2.5, zorder=4, label="Post-match"
        )

        ax.set_title(mod, fontsize=FONT_SIZES['title'])
        ax.set_xlabel("Distance (mean ± 95% CI)")
        ax.grid(axis="x", ls="--", lw=0.8, alpha=0.35)

        ax.set_ylim(2.3, -0.3)

        xmin = float(min(sub["pre_lo"].min(), sub["post_lo"].min()))
        xmax = float(max(sub["pre_hi"].max(), sub["post_hi"].max()))
        pad = 0.06 * (xmax - xmin + 1e-12)
        ax.set_xlim(xmin - pad, xmax + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2], labels[:2],
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=FONT_SIZES['tick']
    )

    # Avoid tight_layout warning with figure-level legend
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22, wspace=0.22)
    _save(fig, name)

def plot_pair_consistency(df_cos: pd.DataFrame, df_rho: pd.DataFrame, df_rho_neg: pd.DataFrame, name: str):
    """Plot within-pair similarity and cost-distance concordance."""
    if df_cos.empty or df_rho.empty:
        print("Consistency plot skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    y_positions = [0.18, 0.82]

    # ============================
    # ============================
    ax = axes[0]
    data_cos = pd.concat([
        pd.DataFrame({'Modality': 'Mass', 'Δcos': df_cos['delta_mean_mass'], 'p': df_cos['perm_p_mass']}),
        pd.DataFrame({'Modality': 'RNA',  'Δcos': df_cos['delta_mean_rna'],  'p': df_cos['perm_p_rna']})
    ], ignore_index=True)

    for j, mod in enumerate(['Mass', 'RNA']):
        vals = data_cos[data_cos['Modality'] == mod]['Δcos'].values.astype(float)
        mean, lo, hi = _mean_t_ci(vals)
        y0 = y_positions[j]

        yjit = np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(vals, np.full_like(vals, y0) + yjit, s=30, alpha=0.6,
                   color=PALETTE[mod], edgecolors='w', linewidths=0.5)
        ax.errorbar(mean, y0, xerr=[[mean - lo], [hi - mean]], fmt='o', color='black',
                    capsize=4, markersize=7)

        frac = float((data_cos[data_cos['Modality'] == mod]['p'] < 0.05).mean() * 100)
        ax.text(mean, y0 + 0.03, f"{mean:+.3f}\n({frac:.0f}% p<0.05)",
                ha='center', va='bottom', fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(['Mass', 'RNA'])
    ax.set_ylim(-0.08, 1.08)
    ax.axvline(0, ls='--', color='0.6', lw=1)
    ax.set_xlabel("Δ Cosine similarity (matched − permuted)")
    ax.set_title("Within-pair similarity", pad=18)

    # ============================
    # ============================
    ax = axes[1]
    data_rho = pd.concat([
        pd.DataFrame({'Modality': 'Mass', 'rho': df_rho['spearman_cost_zs_mass']}),
        pd.DataFrame({'Modality': 'RNA',  'rho': df_rho['spearman_cost_zs_rna']})
    ], ignore_index=True)

    if not df_rho_neg.empty:
        neg = pd.concat([
            pd.DataFrame({'Modality': 'Mass', 'rho': df_rho_neg['spearman_cost_zs_mass']}),
            pd.DataFrame({'Modality': 'RNA',  'rho': df_rho_neg['spearman_cost_zs_rna']})
        ], ignore_index=True)

        for j, mod in enumerate(['Mass', 'RNA']):
            rhos_neg = neg[neg['Modality'] == mod]['rho'].values.astype(float)
            y0 = y_positions[j]
            ax.scatter(
                rhos_neg,
                np.full_like(rhos_neg, y0) - 0.10 + np.random.normal(0, 0.025, len(rhos_neg)),
                color='0.65',
                marker='x',
                s=34,
                alpha=0.8,
                label='Shuffled-cost neg. ctrl.' if j == 0 else None
            )

    for j, mod in enumerate(['Mass', 'RNA']):
        vals = data_rho[data_rho['Modality'] == mod]['rho'].values.astype(float)
        mean, lo, hi = _mean_t_ci(vals)
        y0 = y_positions[j]

        ax.scatter(
            vals,
            np.full_like(vals, y0) + np.random.normal(0, 0.035, len(vals)),
            s=30, alpha=0.6, color=PALETTE[mod], edgecolors='w', linewidths=0.5
        )
        ax.errorbar(mean, y0, xerr=[[mean - lo], [hi - mean]], fmt='o', color='black',
                    capsize=4, markersize=7)
        ax.text(mean, y0 + 0.03, f"{mean:+.3f}", ha='center', va='bottom', fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(['Mass', 'RNA'])
    ax.set_ylim(-0.08, 1.08)
    ax.axvline(0, ls='--', color='0.6', lw=1)
    ax.set_xlabel(r"Spearman $\rho$ (PS-cost vs. $z_s$ distance)")
    ax.set_title("Cost–latent consistency", pad=18)

    ax.legend(frameon=False, loc='upper right', fontsize=FONT_SIZES['tick'])

    plt.tight_layout()
    _save(fig, name)

def plot_coupling_locality(
    df_m: pd.DataFrame,
    df_r: pd.DataFrame,
    df_hit: pd.DataFrame,
    name: str,
    df_ot_repair: pd.DataFrame | None = None,
    show_hit: bool = True,
):
    if df_m.empty or df_r.empty:
        print("Locality/Hit plot skipped.")
        return

    if show_hit and (df_hit is None or df_hit.empty):
        print("Hit panel requested but hit_at_k is empty; will plot locality only.")
        show_hit = False

    fig, axes = plt.subplots(1, 2 if show_hit else 1, figsize=(9, 4) if show_hit else (4.6, 4))
    ax_loc = axes[0] if show_hit else axes

    # --- Locality violin ---
    data_loc = pd.concat([
        pd.DataFrame({'Modality': 'Mass', 'Entropy': df_m['row_entropy']}),
        pd.DataFrame({'Modality': 'RNA',  'Entropy': df_r['row_entropy']})
    ], ignore_index=True)
    sns.violinplot(
        data=data_loc, x='Modality', y='Entropy', hue='Modality',
        palette=[PALETTE['Mass'], PALETTE['RNA']],
        cut=0, inner='quartile', ax=ax_loc,
        linewidth=1.2, legend=False
    )
    ax_loc.set_title("Coupling locality", pad=10)
    ax_loc.set_xlabel('')
    ax_loc.set_ylabel('Row entropy')
    ax_loc.grid(axis="y", ls="--", lw=0.8, alpha=0.30)

    # --- Max entropy line: log(n_source) ---
    if df_ot_repair is not None and not df_ot_repair.empty and "n_source" in df_ot_repair.columns:
        tmp = df_ot_repair.copy()
        tmp["n_source"] = pd.to_numeric(tmp["n_source"], errors="coerce")
        max_lines = {}
        for mod in ["Mass", "RNA"]:
            ns = tmp[tmp["modality"] == mod]["n_source"].dropna().values
            if len(ns) == 0:
                continue
            m_med = float(np.median(ns))
            max_lines[mod] = float(np.log(max(m_med, 1.0)))

        x_map = {"Mass": 0, "RNA": 1}
        for mod, y_max in max_lines.items():
            x0 = x_map.get(mod, None)
            if x0 is None:
                continue
            ax_loc.hlines(
                y=y_max,
                xmin=x0 - 0.38,
                xmax=x0 + 0.38,
                colors="0.35",
                linestyles="--",
                linewidth=1.2,
                zorder=5,
            )
            ax_loc.text(
                x0,
                y_max + 0.05,
                r"$\log(n_{\mathrm{source}})$",
                ha="center",
                va="bottom",
                fontsize=10,
                color="0.35",
            )

    # --- Optional hit@k panel ---
    if show_hit:
        ax = axes[1]

        hit_cols = [c for c in df_hit.columns if c.startswith('hit@') or c.startswith('hit_at_')]
        if not hit_cols:
            hit_cols = [c for c in df_hit.columns if 'hit' in c.lower() and c not in ['modality', 'seed', 'fold']]

        if not hit_cols:
            ax.axis('off')
        else:
            df_hit_agg = df_hit.groupby('modality')[hit_cols].mean().reset_index()
            df_melt = df_hit_agg.melt(id_vars='modality', var_name='k', value_name='hit_rate')
            df_melt['k_display'] = df_melt['k'].astype(str).str.replace('hit@', 'k=').str.replace('hit_at_', 'k=')

            sns.barplot(
                data=df_melt, x='k_display', y='hit_rate', hue='modality',
                palette=PALETTE, ax=ax, edgecolor='black', linewidth=1.2, width=0.7
            )
            ax.set_ylim(0, max(1.0, df_melt['hit_rate'].max() * 1.15))
            ax.set_title("Retrieval performance", pad=10)
            ax.set_xlabel('')
            ax.set_ylabel('Hit rate')
            ax.legend(title=None, frameon=False)

    plt.tight_layout()
    _save(fig, name)

def plot_representation_agreement(df_cka: pd.DataFrame, name: str):
    """Plot the CKA gain over random row-stochastic couplings."""
    if df_cka is None or df_cka.empty:
        print("CKA gain skipped (no data).")
        return

    if "delta_vs_random" in df_cka.columns:
        y_col = "delta_vs_random"
        y_label = r"$\Delta$ Linear CKA (OT - Random)"
        title = "Representation alignment gain"
    elif "delta_vs_colperm" in df_cka.columns:
        y_col = "delta_vs_colperm"
        y_label = r"$\Delta$ Linear CKA (OT - ColPerm)"
        title = "Representation alignment gain"
    else:
        if "cka_ot" not in df_cka.columns:
            print("CKA gain skipped (no compatible columns).")
            return
        y_col = "cka_ot"
        y_label = "Linear CKA (OT matched)"
        title = "Representation alignment"

    plot = df_cka.copy()
    plot = plot[plot["modality"].isin(["Mass", "RNA"])].copy()
    plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
    plot.dropna(subset=[y_col], inplace=True)
    if plot.empty:
        print("CKA gain skipped (empty after cleaning).")
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))
    ax.set_title(title, fontsize=FONT_SIZES["title"] + 1, pad=10)

    order = ["Mass", "RNA"]
    x_map = {"Mass": 0.3, "RNA": 0.7}

    rng = np.random.default_rng(7)
    for m in order:
        sub = plot[plot["modality"] == m][y_col].values.astype(float)
        if len(sub) == 0:
            continue

        x0 = x_map[m]
        jitter = rng.normal(0.0, 0.06, size=len(sub))
        ax.scatter(
            np.full_like(sub, x0, dtype=float) + jitter,
            sub,
            s=26,
            alpha=0.35,
            color=PALETTE[m],
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        mean, lo, hi = _mean_t_ci(sub)
        ax.errorbar(
            [x0], [mean],
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color="black",
            mfc="black",
            mec="black",
            ms=6.5,
            capsize=3.5,
            elinewidth=1.6,
            zorder=4,
        )
        ax.text(x0, hi + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    if y_col.startswith("delta_"):
        ax.axhline(0.0, color="0.6", lw=1.0, ls="--", zorder=1)

    ax.set_xticks([x_map["Mass"], x_map["RNA"]])
    ax.set_xticklabels(["Mass", "RNA"])
    ax.set_xlim(0.0, 1.0)

    ax.set_ylabel(y_label)
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.35)

    plt.tight_layout()
    _save(fig, name)

def plot_class_flow_heatmap(flow_files: list, name: str, cmap: str):
    if not flow_files:
        print("Flow heatmap skipped (no files).")
        return

    mats = []
    labels = None
    for fp in flow_files:
        try:
            df = pd.read_csv(fp, index_col=0)
            mats.append(df.values.astype(float))
            labels = df.columns.tolist()
        except Exception:
            continue

    if not mats or labels is None:
        print("Flow heatmap skipped (not readable).")
        return

    M = np.mean(mats, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))
    ax.set_title(name.replace("_", " "), fontsize=FONT_SIZES["title"] + 1, pad=10)

    sns.heatmap(
        M,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=1.0,
        linecolor="white",
        square=True,
    )
    ax.set_xlabel("Donor label")
    ax.set_ylabel("Glycan label")
    plt.tight_layout()
    _save(fig, name)

def plot_alignment_summary(data: dict, name: str = "alignment_summary"):
    """
    Compose the reported alignment diagnostics using a consistent layout.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import t
    import matplotlib.lines as mlines
    import matplotlib as mpl

    rc_backup = mpl.rcParams.copy()

    try:
        sns.set_style("ticks")
        plt.style.use("default")

        # Global style
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,

            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",

            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
        })

        PALETTE = {
            "Mass": "#B4AED3",
            "RNA": "#E8B2A6",
            "NegCtrl": "0.65",
        }

        HEADER_Y = 1.03
        LETTER_X = -0.18

        ANNOT_FS = 7.0
        ANNOT_FS_SMALL = 6.8

        def _set_header(ax, letter: str, title: str):
            ax.text(
                LETTER_X, HEADER_Y, letter,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=10, fontweight="bold",
                clip_on=False,
            )
            ax.text(
                0.5, HEADER_Y, title,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9, fontweight="normal",
                clip_on=False,
            )

        def _mean_ci(vals: np.ndarray):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            if n == 0:
                return np.nan, np.nan, np.nan, 0
            m = float(np.mean(vals))
            if n < 2:
                return m, m, m, n
            se = float(np.std(vals, ddof=1) / np.sqrt(n))
            ci = float(se * t.ppf(0.975, n - 1))
            return m, m - ci, m + ci, n

        def _style_axis(ax):
            ax.grid(axis="y", ls="--", lw=0.6, alpha=0.25)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.tick_params(pad=1.3)

        def _annot_box(ax, text: str, x: float, y: float):
            ax.text(
                x, y, text,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=ANNOT_FS,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.80", alpha=0.92),
            )

        def _select_true_coupling_rows(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            if "control" not in df.columns:
                return df.copy()
            control = df["control"].astype(str).str.lower()
            return df[control.eq("none")].copy()

        # ---------------------------
        # Pull dfs
        # ---------------------------
        df_smd = data.get("smd", pd.DataFrame())
        df_global = data.get("global", pd.DataFrame())
        df_within = data.get("within_pair", pd.DataFrame())
        df_rho = data.get("cost_latent", pd.DataFrame())
        df_rho_neg = data.get("cost_latent_neg", pd.DataFrame())
        df_cka = data.get("cka", pd.DataFrame())
        df_loc_m = data.get("locality_mass", pd.DataFrame())
        df_loc_r = data.get("locality_rna", pd.DataFrame())
        df_ot_repair = data.get("ot_repair", pd.DataFrame())

        # ---------------------------
        # A4 width × half A4 height
        # ---------------------------
        fig = plt.figure(figsize=(8.27, 5.845))
        gs = fig.add_gridspec(
            2, 3,
            left=0.050, right=0.995,
            top=0.975, bottom=0.110,
            wspace=0.22, hspace=0.44
        )

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axCoup = fig.add_subplot(gs[0, 2])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])
        axE = fig.add_subplot(gs[1, 2])

        # =========================================================
        # a: Balance
        # =========================================================
        _set_header(axA, "a", "Balance in PS space")

        if df_smd.empty:
            axA.text(0.5, 0.5, "No data", ha="center", va="center", transform=axA.transAxes, fontsize=ANNOT_FS)
        else:
            order = ["Stable", "Active", "Control"]
            ymap = {g: i for i, g in enumerate(order)}
            y_offset = 0.16

            rows = []
            for modality in ["Mass", "RNA"]:
                for group in order:
                    sub = df_smd[(df_smd["modality"] == modality) & (df_smd["group"] == group)]
                    if sub.empty:
                        continue
                    pre_m, pre_lo, pre_hi, _ = _mean_ci(sub["pre_match_smd"].values)
                    post_m, post_lo, post_hi, _ = _mean_ci(sub["post_match_smd"].values)
                    rows.append({
                        "modality": modality,
                        "group": group,
                        "y": ymap[group],
                        "pre_m": pre_m, "pre_lo": pre_lo, "pre_hi": pre_hi,
                        "post_m": post_m, "post_lo": post_lo, "post_hi": post_hi,
                    })
            plot_df = pd.DataFrame(rows)

            for modality in ["Mass", "RNA"]:
                sub = plot_df[plot_df["modality"] == modality].copy()
                if sub.empty:
                    continue
                color = PALETTE[modality]
                off = +y_offset if modality == "Mass" else -y_offset

                for _, r in sub.iterrows():
                    y = r["y"] + off
                    axA.plot([r["pre_m"], r["post_m"]], [y, y], color="0.70", lw=1.0, alpha=0.55, zorder=1)
                    axA.errorbar(
                        r["pre_m"], y,
                        xerr=[[r["pre_m"] - r["pre_lo"]], [r["pre_hi"] - r["pre_m"]]],
                        fmt="o", ms=5.6, mfc="white", mec=color, ecolor=color,
                        elinewidth=1.1, capsize=2.5, zorder=3
                    )
                    axA.errorbar(
                        r["post_m"], y,
                        xerr=[[r["post_m"] - r["post_lo"]], [r["post_hi"] - r["post_m"]]],
                        fmt="s", ms=5.2, mfc=color, mec=color, ecolor=color,
                        elinewidth=1.1, capsize=2.5, zorder=4
                    )

            axA.axvline(0.10, ls="--", color="0.55", lw=1.0)
            axA.axvline(0.05, ls=":", color="0.55", lw=1.0)

            axA.set_yticks([ymap[g] for g in order])
            axA.set_yticklabels([f"PS\n({g})" for g in order])

            axA.invert_yaxis()
            axA.set_xlabel(r"|SMD| (mean ± 95% CI)")
            _style_axis(axA)

            if "delta_smd" in df_smd.columns:
                dm = float(pd.to_numeric(df_smd[df_smd["modality"] == "Mass"]["delta_smd"], errors="coerce").dropna().mean())
                dr = float(pd.to_numeric(df_smd[df_smd["modality"] == "RNA"]["delta_smd"], errors="coerce").dropna().mean())
                _annot_box(axA, f"Δ|SMD| (Post−Pre)\nMass: {dm:+.3f}\nRNA:  {dr:+.3f}", x=0.98, y=0.96)

        # =========================================================
        # b: Global distribution distance
        # =========================================================
        _set_header(axB, "b", "Global PS-space distance")

        def _global_metric_rows(df, modality: str):
            metrics = [("emd", "EMD"), ("mmd2", r"MMD$^2$"), ("sink", "Sinkhorn div.")]
            sub = df[df["modality"] == modality].copy()
            rows = []
            for key, label in metrics:
                pre = f"{key}_pre"
                post = f"{key}_post"
                if pre not in sub.columns or post not in sub.columns:
                    continue
                pre_m, pre_lo, pre_hi, _ = _mean_ci(sub[pre].values)
                post_m, post_lo, post_hi, _ = _mean_ci(sub[post].values)
                rows.append({
                    "metric_key": key,
                    "metric": label,
                    "pre_m": pre_m, "pre_lo": pre_lo, "pre_hi": pre_hi,
                    "post_m": post_m, "post_lo": post_lo, "post_hi": post_hi,
                })
            return pd.DataFrame(rows)

        def _pct_drop(pre: float, post: float) -> float:
            if pre is None or not np.isfinite(pre) or abs(pre) < 1e-12:
                return np.nan
            return (post - pre) / pre * 100.0

        if df_global.empty:
            axB.text(0.5, 0.5, "No data", ha="center", va="center", transform=axB.transAxes, fontsize=ANNOT_FS)
        else:
            mass_df = _global_metric_rows(df_global, "Mass")
            rna_df = _global_metric_rows(df_global, "RNA")

            metrics_order = ["emd", "mmd2", "sink"]
            metric_labels = {"emd": "EMD", "mmd2": r"MMD$^2$", "sink": "Sinkhorn\ndiv."}

            ymap = {k: i for i, k in enumerate(metrics_order)}
            y_offset = 0.16

            axB.set_yticks([ymap[k] for k in metrics_order])
            axB.set_yticklabels([metric_labels[k] for k in metrics_order])
            axB.invert_yaxis()

            def _plot_mod(modality: str, dfm: pd.DataFrame, off: float):
                if dfm.empty:
                    return
                color = PALETTE[modality]
                for _, r in dfm.iterrows():
                    if r["metric_key"] not in ymap:
                        continue
                    y = ymap[r["metric_key"]] + off
                    axB.plot([r["pre_m"], r["post_m"]], [y, y], color="0.70", lw=1.0, alpha=0.55, zorder=1)

                    axB.errorbar(
                        r["pre_m"], y,
                        xerr=[[r["pre_m"] - r["pre_lo"]], [r["pre_hi"] - r["pre_m"]]],
                        fmt="o", ms=5.6, mfc="white", mec=color, ecolor=color,
                        elinewidth=1.1, capsize=2.5, zorder=3
                    )
                    axB.errorbar(
                        r["post_m"], y,
                        xerr=[[r["post_m"] - r["post_lo"]], [r["post_hi"] - r["post_m"]]],
                        fmt="s", ms=5.2, mfc=color, mec=color, ecolor=color,
                        elinewidth=1.1, capsize=2.5, zorder=4
                    )

                    pct = _pct_drop(float(r["pre_m"]), float(r["post_m"]))
                    if np.isfinite(pct):
                        x_right = float(r["pre_hi"])
                        axB.annotate(
                            f"{pct:.0f}%",
                            xy=(x_right, y),
                            xytext=(4, 0),
                            textcoords="offset points",
                            ha="left",
                            va="center",
                            fontsize=ANNOT_FS_SMALL,
                            color=color,
                        )

            _plot_mod("Mass", mass_df, +y_offset)
            _plot_mod("RNA", rna_df, -y_offset)

            axB.set_xlabel(r"Distance (mean ± 95% CI)")
            axB.grid(axis="x", ls="--", lw=0.6, alpha=0.25)
            for spine in axB.spines.values():
                spine.set_linewidth(0.6)
            axB.tick_params(pad=1.3)

            xs = []
            for dfm in (mass_df, rna_df):
                if dfm is not None and not dfm.empty:
                    xs.extend(dfm["pre_lo"].tolist())
                    xs.extend(dfm["pre_hi"].tolist())
                    xs.extend(dfm["post_lo"].tolist())
                    xs.extend(dfm["post_hi"].tolist())
            if xs:
                xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
                span = max(xmax - xmin, 1e-6)
                axB.set_xlim(xmin - 0.04 * span, xmax + 0.22 * span)

        # =========================================================
        # c: Coupling locality
        # =========================================================
        _set_header(axCoup, "c", "Coupling locality")
        axCoup.set_xlabel("Modality")
        axCoup.set_ylabel("Row entropy")

        sub_m = _select_true_coupling_rows(df_loc_m)
        sub_r = _select_true_coupling_rows(df_loc_r)

        loc_rows = []
        if (sub_m is not None) and (not sub_m.empty) and ("row_entropy" in sub_m.columns):
            loc_rows.append(pd.DataFrame({"Modality": "Mass", "RowEntropy": pd.to_numeric(sub_m["row_entropy"], errors="coerce")}))
        if (sub_r is not None) and (not sub_r.empty) and ("row_entropy" in sub_r.columns):
            loc_rows.append(pd.DataFrame({"Modality": "RNA", "RowEntropy": pd.to_numeric(sub_r["row_entropy"], errors="coerce")}))

        loc_df = pd.concat(loc_rows, ignore_index=True) if loc_rows else pd.DataFrame()
        loc_df = loc_df.dropna(subset=["RowEntropy"]) if not loc_df.empty else loc_df

        if loc_df.empty:
            axCoup.text(0.5, 0.5, "No locality data", ha="center", va="center", transform=axCoup.transAxes, fontsize=ANNOT_FS)
        else:
            sns.violinplot(
                data=loc_df,
                x="Modality",
                y="RowEntropy",
                hue="Modality",
                order=["Mass", "RNA"],
                hue_order=["Mass", "RNA"],
                palette={"Mass": PALETTE["Mass"], "RNA": PALETTE["RNA"]},
                cut=0,
                inner="quartile",
                linewidth=1.0,
                legend=False,
                ax=axCoup
            )
            axCoup.grid(axis="y", ls="--", lw=0.6, alpha=0.25)

            if (df_ot_repair is not None) and (not df_ot_repair.empty) and ("n_source" in df_ot_repair.columns):
                tmp = df_ot_repair.copy()
                tmp["n_source"] = pd.to_numeric(tmp["n_source"], errors="coerce")
                for i, mod in enumerate(["Mass", "RNA"]):
                    ns = tmp[tmp["modality"] == mod]["n_source"].dropna().values
                    if len(ns) == 0:
                        continue
                    m_med = float(np.median(ns))
                    y_max = float(np.log(max(m_med, 2.0)))
                    axCoup.hlines(y=y_max, xmin=i - 0.35, xmax=i + 0.35, colors="0.35",
                                 linestyles="--", linewidth=1.0)
                    axCoup.text(i, y_max + 0.05, r"$\log(n_{\mathrm{source}})$",
                                ha="center", va="bottom", fontsize=ANNOT_FS, color="0.35")

        for spine in axCoup.spines.values():
            spine.set_linewidth(0.6)
        axCoup.tick_params(pad=1.3)

        # =========================================================
        # d: Within-pair similarity
        # =========================================================
        _set_header(axC, "d", "Within-pair similarity")

        if df_within.empty:
            axC.text(0.5, 0.5, "No data", ha="center", va="center", transform=axC.transAxes, fontsize=ANNOT_FS)
        else:
            tmp = pd.concat([
                pd.DataFrame({"Modality": "Mass", "delta": pd.to_numeric(df_within["delta_mean_mass"], errors="coerce"),
                              "p": pd.to_numeric(df_within["perm_p_mass"], errors="coerce")}),
                pd.DataFrame({"Modality": "RNA", "delta": pd.to_numeric(df_within["delta_mean_rna"], errors="coerce"),
                              "p": pd.to_numeric(df_within["perm_p_rna"], errors="coerce")}),
            ], ignore_index=True).dropna(subset=["delta"])

            y_positions = {"Mass": 0.30, "RNA": 0.75}
            rng = np.random.default_rng(7)

            for mod in ["Mass", "RNA"]:
                vals = tmp[tmp["Modality"] == mod]["delta"].values.astype(float)
                if len(vals) == 0:
                    continue
                mean, lo, hi, _ = _mean_ci(vals)
                y0 = y_positions[mod]
                jitter = rng.normal(0.0, 0.03, size=len(vals))

                axC.scatter(vals, np.full_like(vals, y0) + jitter,
                            s=18, alpha=0.55, color=PALETTE[mod],
                            edgecolors="white", linewidths=0.4)
                axC.errorbar([mean], [y0], xerr=[[mean - lo], [hi - mean]],
                             fmt="o", color="black", mfc="black", mec="black",
                             ms=5.5, capsize=3.0, elinewidth=1.2, zorder=5)

                frac = float(np.mean(tmp[tmp["Modality"] == mod]["p"].values < 0.05) * 100.0)
                axC.text(mean, y0 + 0.04, f"{mean:+.3f}\n({frac:.0f}% p<0.05)",
                         ha="center", va="bottom", fontsize=ANNOT_FS)

            axC.axvline(0.0, ls="--", color="0.6", lw=1.0)
            axC.set_yticks([y_positions["Mass"], y_positions["RNA"]])
            axC.set_yticklabels(["Mass", "RNA"])
            axC.set_ylim(0.05, 1.00)
            axC.set_xlabel(r"$\Delta$ Cosine similarity (matched − permuted)")
            _style_axis(axC)

        # =========================================================
        # e: Cost–latent consistency
        # =========================================================
        _set_header(axD, "e", "Cost–latent consistency")

        if df_rho.empty:
            axD.text(0.5, 0.5, "No data", ha="center", va="center", transform=axD.transAxes, fontsize=ANNOT_FS)
        else:
            rho_mass = pd.to_numeric(df_rho.get("spearman_cost_zs_mass"), errors="coerce")
            rho_rna = pd.to_numeric(df_rho.get("spearman_cost_zs_rna"), errors="coerce")

            neg_mass = pd.Series([], dtype=float)
            neg_rna = pd.Series([], dtype=float)
            if df_rho_neg is not None and not df_rho_neg.empty:
                neg_mass = pd.to_numeric(df_rho_neg.get("spearman_cost_zs_mass"), errors="coerce")
                neg_rna = pd.to_numeric(df_rho_neg.get("spearman_cost_zs_rna"), errors="coerce")

            y_positions = {"Mass": 0.30, "RNA": 0.75}
            rng = np.random.default_rng(11)

            neg_y_shift = -0.07
            if len(neg_mass.dropna()) > 0:
                j = rng.normal(0.0, 0.020, size=len(neg_mass.dropna()))
                axD.scatter(
                    neg_mass.dropna().values,
                    np.full_like(neg_mass.dropna().values, y_positions["Mass"] + neg_y_shift) + j,
                    marker="x", s=28, color=PALETTE["NegCtrl"], alpha=0.8,
                    label="Shuffled-cost neg. ctrl."
                )
            if len(neg_rna.dropna()) > 0:
                j = rng.normal(0.0, 0.020, size=len(neg_rna.dropna()))
                axD.scatter(
                    neg_rna.dropna().values,
                    np.full_like(neg_rna.dropna().values, y_positions["RNA"] + neg_y_shift) + j,
                    marker="x", s=28, color=PALETTE["NegCtrl"], alpha=0.8
                )

            for mod, series in [("Mass", rho_mass), ("RNA", rho_rna)]:
                vals = series.dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                mean, lo, hi, _ = _mean_ci(vals)
                y0 = y_positions[mod]
                j = rng.normal(0.0, 0.03, size=len(vals))
                axD.scatter(vals, np.full_like(vals, y0) + j,
                            s=18, alpha=0.55, color=PALETTE[mod],
                            edgecolors="white", linewidths=0.4)
                axD.errorbar([mean], [y0], xerr=[[mean - lo], [hi - mean]],
                             fmt="o", color="black", mfc="black", mec="black",
                             ms=5.5, capsize=3.0, elinewidth=1.2, zorder=5)
                axD.text(mean, y0 + 0.04, f"{mean:+.3f}", ha="center", va="bottom", fontsize=ANNOT_FS)

            axD.axvline(0.0, ls="--", color="0.6", lw=1.0)
            axD.set_yticks([y_positions["Mass"], y_positions["RNA"]])
            axD.set_yticklabels(["Mass", "RNA"])
            axD.set_ylim(0.05, 1.00)
            axD.set_xlabel(r"Spearman $\rho$ (PS-cost vs. $z_s$ distance)")
            _style_axis(axD)

            if (df_rho_neg is not None) and (not df_rho_neg.empty):
                axD.legend(frameon=False, loc="upper right", fontsize=7)

        # =========================================================
        # f: Representation alignment gain
        # =========================================================
        _set_header(axE, "f", "Representation alignment gain")

        if df_cka.empty:
            axE.text(0.5, 0.5, "No data", ha="center", va="center", transform=axE.transAxes, fontsize=ANNOT_FS)
        else:
            ycol = "delta_vs_random" if "delta_vs_random" in df_cka.columns else None
            if ycol is None:
                axE.text(0.5, 0.5, "No delta_vs_random", ha="center", va="center",
                         transform=axE.transAxes, fontsize=ANNOT_FS)
            else:
                plot = df_cka[df_cka["modality"].isin(["Mass", "RNA"])].copy()
                plot[ycol] = pd.to_numeric(plot[ycol], errors="coerce")
                plot = plot.dropna(subset=[ycol])

                if plot.empty:
                    axE.text(0.5, 0.5, "No usable rows", ha="center", va="center",
                             transform=axE.transAxes, fontsize=ANNOT_FS)
                else:
                    x_map = {"Mass": 0.33, "RNA": 0.67}
                    rng = np.random.default_rng(17)

                    for mod in ["Mass", "RNA"]:
                        vals = plot[plot["modality"] == mod][ycol].values.astype(float)
                        if len(vals) == 0:
                            continue
                        x0 = x_map[mod]
                        jitter = rng.normal(0.0, 0.050, size=len(vals))
                        axE.scatter(np.full_like(vals, x0, dtype=float) + jitter, vals,
                                   s=18, alpha=0.45, color=PALETTE[mod],
                                   edgecolors="white", linewidths=0.4)
                        mean, lo, hi, _ = _mean_ci(vals)
                        axE.errorbar([x0], [mean], yerr=[[mean - lo], [hi - mean]],
                                     fmt="o", color="black", mfc="black", mec="black",
                                     ms=5.8, capsize=3.0, elinewidth=1.2, zorder=5)
                        axE.text(x0, hi + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=ANNOT_FS)

                    axE.axhline(0.0, color="0.6", lw=1.0, ls="--")
                    axE.set_xlim(0.0, 1.0)
                    axE.set_xticks([x_map["Mass"], x_map["RNA"]])
                    axE.set_xticklabels(["Mass", "RNA"])
                    axE.set_ylabel(r"$\Delta$ Linear CKA (OT − Random)")
                    axE.grid(axis="y", ls="--", lw=0.6, alpha=0.25)
                    for spine in axE.spines.values():
                        spine.set_linewidth(0.6)
                    axE.tick_params(pad=1.3)

        # ---------------------------
        # Shared legend (bottom) — move slightly down to free space
        # ---------------------------
        pre_handle = mlines.Line2D([], [], color="0.25", marker="o", ms=5.6, mfc="white", mec="0.25",
                                  ls="none", label="Pre-match")
        post_handle = mlines.Line2D([], [], color="0.25", marker="s", ms=5.2, mfc="0.25", mec="0.25",
                                   ls="none", label="Post-match")
        mass_handle = mlines.Line2D([], [], color=PALETTE["Mass"], lw=2.2, label="Mass")
        rna_handle = mlines.Line2D([], [], color=PALETTE["RNA"], lw=2.2, label="RNA")

        fig.legend(
            handles=[pre_handle, post_handle, mass_handle, rna_handle],
            frameon=False,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            columnspacing=1.35,
            handletextpad=0.6
        )

        FINAL_FIGURES_DIR.mkdir(exist_ok=True, parents=True)
        png_path = FINAL_FIGURES_DIR / f"{name}.png"
        pdf_path = FINAL_FIGURES_DIR / f"{name}.pdf"
        fig.savefig(png_path, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)

        print(f"[OK] Alignment summary saved to: {png_path}")
        print(f"[OK] Alignment summary saved to: {pdf_path}")

    finally:
        mpl.rcParams.update(rc_backup)

def main():
    set_style()
    data = collect_all_results()

    plot_smd_balance(data['smd'], "smd_balance")
    plot_global_distances(data['global'], "global_distances")
    plot_pair_consistency(data['within_pair'], data['cost_latent'], data['cost_latent_neg'], "pair_consistency")

    plot_coupling_locality(
        data['locality_mass'],
        data['locality_rna'],
        data['hit_at_k'],
        "coupling_locality",
        df_ot_repair=data.get("ot_repair"),
        show_hit=False,
    )

    plot_representation_agreement(data['cka'], "representation_agreement")
    plot_class_flow_heatmap(data['class_flow_mass'], "class_flow_mass", cmap="Purples")
    plot_class_flow_heatmap(data['class_flow_rna'],  "class_flow_rna",  cmap="Reds")

    plot_alignment_summary(data, name="alignment_summary")

    print("All alignment plots have been generated.")

if __name__ == "__main__":
    main()
