# plot_signed_shap_summary.py
"""Plot class-specific signed SHAP summaries for the three clinical groups."""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

import textwrap

from slemodel import config as cfg


# -----------------------------
# Paths
# -----------------------------
SHAP_RESULTS_PATH = Path("clinical_explain_results") / "shap_outputs" / "mor_shap_outputs.npz"
OUT_DIR = Path("clinical_explain_results") / "signed_shap_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "signed_shap_summary.png"
OUT_PDF = OUT_DIR / "signed_shap_summary.pdf"


# -----------------------------
# Publication style
# -----------------------------
def set_publication_style(font: str = "Helvetica"):
    matplotlib.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",

        "font.family": "sans-serif",
        "font.sans-serif": [font, "Arial", "DejaVu Sans"],

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Compact typography
        "axes.titlesize": 10.2,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.0,

        "axes.linewidth": 0.9,
        "axes.grid": False,
    })

# -----------------------------
# Constants / Colors
# -----------------------------
LABEL_NAMES = cfg.VisualizationConfig.LABEL_NAMES
CLASS_ORDER = [1, 0, 2]  # Active, Stable, Control
CLASS_COLORS = {
    1: cfg.VisualizationConfig.LABEL_COLORS.get(1, "#E68B81"),
    0: cfg.VisualizationConfig.LABEL_COLORS.get(0, "#8aab82"),
    2: cfg.VisualizationConfig.LABEL_COLORS.get(2, "#7DA6C6"),
}
MODALITY_COLORS = cfg.VisualizationConfig.MODALITY_COLORS
MODALITY_PRETTY = {
    "gly": cfg.VisualizationConfig.MODALITY_NAMES.get("gly", "Glycan"),
    "mass": cfg.VisualizationConfig.MODALITY_NAMES.get("mass", "Mass"),
    "rna": cfg.VisualizationConfig.MODALITY_NAMES.get("rna", "RNA"),
}


def panel_label(ax, letter: str):
    ax.text(
        -0.14, 1.04, letter,
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def _modality_of(feature_name: str) -> str:
    if isinstance(feature_name, str) and "::" in feature_name:
        return feature_name.split("::", 1)[0]
    return "?"


def load_shap_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SHAP NPZ: {path}. Please run shap_analysis.py first."
        )
    with np.load(path, allow_pickle=True) as data:
        d = {k: data[k] for k in data.files}

    required = ["phi_macro", "phi_stack", "X_total", "y_true",
                "feature_names", "display_feature_names", "allowed_mask"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}")
    return d


# -----------------------------
# Label handling (mass long names)
# -----------------------------
def shorten_label(label: str, max_chars: int = 22, max_lines: int = 2) -> str:
    """
    Publication-friendly wrapping/truncation:
    - Wrap on spaces/hyphens
    - Limit lines
    - Truncate with ellipsis if still too long
    """
    if label is None:
        return ""
    s = str(label).strip()

    # Normalize multiple spaces
    s = " ".join(s.split())

    # Prefer splitting on space; keep hyphen as break opportunity
    s2 = s.replace("-", "- ")

    lines = textwrap.wrap(s2, width=max_chars, break_long_words=False, break_on_hyphens=True)
    lines = [ln.replace("- ", "-") for ln in lines]  # restore

    if len(lines) <= max_lines:
        return "\n".join(lines)

    # truncate to max_lines with ellipsis
    kept = lines[:max_lines]
    # ensure last line not too long
    last = kept[-1]
    if len(last) > max_chars - 1:
        last = last[: max_chars - 1].rstrip()
    kept[-1] = last + "…"
    return "\n".join(kept)


# -----------------------------
# Bootstrap utilities
# -----------------------------
def bootstrap_ci_mean(X: np.ndarray, n_boot: int = 600, seed: int = 7):
    """
    X: (N, K) values
    Returns mean, ci_low, ci_high for each column.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= 2:
        m = X.mean(axis=0)
        return m, m, m

    mean = X.mean(axis=0)
    boots = np.empty((n_boot, X.shape[1]), dtype=float)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        boots[b] = X[sel].mean(axis=0)

    ci_low = np.quantile(boots, 0.025, axis=0)
    ci_high = np.quantile(boots, 0.975, axis=0)
    return mean, ci_low, ci_high

def _robust_scale01(values: np.ndarray,
                    ref_values: np.ndarray,
                    q_low: float = 5.0,
                    q_high: float = 95.0) -> np.ndarray:
    """
    Robustly scale values to [0,1] using ref_values percentiles.
    This makes color meaning consistent across panels and resistant to outliers.
    """
    v = np.asarray(values, dtype=float)
    ref = np.asarray(ref_values, dtype=float)

    vmin, vmax = np.nanpercentile(ref, [q_low, q_high])
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-12):
        return np.full(v.shape, 0.5, dtype=float)

    z = (v - vmin) / (vmax - vmin)
    return np.clip(z, 0.0, 1.0)


def _beeswarm_offsets_binned(x: np.ndarray,
                             max_spread: float = 0.32,
                             nbins: int = 24,
                             seed: int = 7) -> np.ndarray:
    """
    Deterministic beeswarm-like offsets by binning x and stacking points vertically.
    Produces clean 'violin-like' point clouds without random messy jitter.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return np.zeros(n, dtype=float)

    lo, hi = np.nanpercentile(x, [1.0, 99.0])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < 1e-12):
        return np.zeros(n, dtype=float)

    bins = np.linspace(lo, hi, nbins + 1)
    bin_id = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)

    offsets = np.zeros(n, dtype=float)
    rng = np.random.default_rng(seed)

    for b in range(nbins):
        idx = np.where(bin_id == b)[0]
        if idx.size <= 1:
            continue

        # shuffle within bin (but deterministic via seed)
        rng.shuffle(idx)
        k = idx.size

        # step chosen to fit within max_spread
        step = min(0.11, (2.0 * max_spread) / max(k - 1, 1))

        # sequence: 0, +1, -1, +2, -2, ...
        seq = np.zeros(k, dtype=float)
        for i in range(1, k):
            t = (i + 1) // 2
            seq[i] = t if (i % 2 == 1) else -t

        off = np.clip(seq * step, -max_spread, max_spread)
        off += (rng.random(k) - 0.5) * step * 0.12  # tiny dither to avoid exact overlaps
        offsets[idx] = np.clip(off, -max_spread, max_spread)

    return offsets


def _add_modality_strip(ax,
                        y: float,
                        color: str,
                        x_axes: float = -0.055,
                        width_axes: float = 0.018,
                        height_data: float = 0.56):
    """
    Draw a small colored strip left of y-tick labels to indicate modality.
    Uses blended transform (x in axes coords, y in data coords) -> stable, journal-like.
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    rect = Rectangle(
        (x_axes, y - height_data / 2),
        width_axes,
        height_data,
        transform=trans,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
        zorder=3
    )
    ax.add_patch(rect)


# -----------------------------
# Modality importance
# -----------------------------
def compute_modality_vectors(phi_macro: np.ndarray, feature_names: list[str]) -> dict:
    """
    phi_macro: (N,F) mean(|SHAP|) across classes per sample-feature.
    For each sample, modality contribution = sum(phi_macro over features in that modality).
    Returns vectors (N,) for gly/mass/rna, plus mean proportions.
    """
    mods = np.array([_modality_of(fn) for fn in feature_names], dtype=object)
    idx_g = np.where(mods == "gly")[0]
    idx_m = np.where(mods == "mass")[0]
    idx_r = np.where(mods == "rna")[0]

    vg = phi_macro[:, idx_g].sum(axis=1) if idx_g.size else np.zeros(phi_macro.shape[0])
    vm = phi_macro[:, idx_m].sum(axis=1) if idx_m.size else np.zeros(phi_macro.shape[0])
    vr = phi_macro[:, idx_r].sum(axis=1) if idx_r.size else np.zeros(phi_macro.shape[0])

    # proportions computed on mean contributions
    mean_arr = np.array([vg.mean(), vm.mean(), vr.mean()], dtype=float)
    prop = mean_arr / (mean_arr.sum() + 1e-12)

    return {"gly": vg, "mass": vm, "rna": vr, "prop": prop}


def plot_panel_a_modality_importance(ax, phi_macro: np.ndarray, feature_names: list[str],
                                    n_boot: int = 600, seed: int = 7):
    pack = compute_modality_vectors(phi_macro, feature_names)
    vecs = np.vstack([pack["gly"], pack["mass"], pack["rna"]]).T  # (N,3)

    mean, lo, hi = bootstrap_ci_mean(vecs, n_boot=n_boot, seed=seed)
    prop = pack["prop"]

    ylabels = [MODALITY_PRETTY["gly"], MODALITY_PRETTY["mass"], MODALITY_PRETTY["rna"]]
    mods = ["gly", "mass", "rna"]
    colors = [MODALITY_COLORS.get(m, "#999999") for m in mods]

    y = np.arange(3)[::-1]

    ax.barh(y=y, width=mean, color=colors, edgecolor="none", height=0.62, alpha=0.95)

    xerr_left = mean - lo
    xerr_right = hi - mean
    ax.errorbar(
        mean, y,
        xerr=np.vstack([xerr_left, xerr_right]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.0,
        capthick=0.9
    )

    # % label at end of each bar
    for yy, mval, p in zip(y, mean, prop):
        ax.text(mval * 1.02, yy, f"{p*100:.1f}%", va="center", ha="left", fontsize=8.6)

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=8.8)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("Mean |SHAP| contribution", labelpad=3)
    ax.set_title("Modality importance", loc="left", pad=4)

    # subtle vertical grid like the reference
    ax.xaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# -----------------------------
# Panels b/c/d: Signed SHAP (mean ± 95% CI), quota 5/5/5
# -----------------------------
def pick_top_features_quota(mean_abs: np.ndarray,
                            feature_names: list[str],
                            allowed_mask: np.ndarray,
                            top_k_total: int = 15,
                            k_per_mod: int = 5) -> np.ndarray:
    """
    Pick top features with modality quota, then (optionally) fill remaining slots
    by global importance. Mass features are filtered by allowed_mask.
    """
    mean_abs = np.asarray(mean_abs, dtype=float)
    mods = np.array([_modality_of(fn) for fn in feature_names], dtype=object)
    allowed_mask = np.asarray(allowed_mask, dtype=bool)

    F = len(feature_names)
    top_k_total = int(top_k_total)

    def _is_valid(i: int) -> bool:
        if mods[i] != "mass":
            return True
        return bool(allowed_mask[i])

    picked: list[int] = []

    # 1) quota per modality
    for mod in ["gly", "mass", "rna"]:
        idx = np.where(mods == mod)[0]
        if idx.size == 0:
            continue
        if mod == "mass":
            idx = idx[allowed_mask[idx]]
        if idx.size == 0:
            continue

        idx_sorted = idx[np.argsort(mean_abs[idx])[::-1]]
        picked.extend([int(i) for i in idx_sorted[:k_per_mod]])

    # unique preserve order
    seen = set()
    picked = [i for i in picked if not (i in seen or seen.add(i))]

    # 2) fill remaining by global importance (respect mass allowed_mask)
    if len(picked) < top_k_total:
        valid_idx = np.array([i for i in range(F) if _is_valid(i)], dtype=int)
        if valid_idx.size > 0:
            idx_sorted = valid_idx[np.argsort(mean_abs[valid_idx])[::-1]]
            for i in idx_sorted:
                ii = int(i)
                if ii in seen:
                    continue
                picked.append(ii)
                seen.add(ii)
                if len(picked) >= top_k_total:
                    break

    # 3) if too many, keep top_k_total by importance
    if len(picked) > top_k_total:
        picked = sorted(picked, key=lambda i: mean_abs[i], reverse=True)[:top_k_total]

    # 4) final ordering: grouped by modality, each group sorted by importance
    out: list[int] = []
    for mod in ["gly", "mass", "rna"]:
        idx_mod = [i for i in picked if _modality_of(feature_names[i]) == mod]
        idx_mod = sorted(idx_mod, key=lambda i: mean_abs[i], reverse=True)
        out.extend(idx_mod)

    return np.array(out, dtype=int)

def plot_beeswarm_panel(ax,
                        class_idx: int,
                        phi_stack: np.ndarray,
                        X_total: np.ndarray,
                        y_true: np.ndarray,
                        feature_names: list[str],
                        display_names: list[str],
                        allowed_mask: np.ndarray,
                        top_k_total: int = 15,
                        k_per_mod: int = 5,
                        max_display_points: int = 220,
                        point_size: float = 8.0,
                        alpha: float = 0.62,
                        jitter_scale: float = 0.32,
                        seed: int = 7):
    """
    Cleaner, journal-style beeswarm:
    - Deterministic bin-based beeswarm offsets (less messy than random jitter)
    - Robust color scaling by feature 5–95% percentiles (consistent across panels)
    - Stable modality indicator as a left colored strip (axes-x + data-y transform)
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true, dtype=int)
    mask = (y_true == int(class_idx))
    if mask.sum() == 0:
        mask = np.ones_like(y_true, dtype=bool)

    phi_class = np.asarray(phi_stack[mask, :, int(class_idx)], dtype=float)  # (N_class, F)
    X_class = np.asarray(X_total[mask, :], dtype=float)                     # (N_class, F)

    mean_abs = np.nanmean(np.abs(phi_class), axis=0)
    feat_idx = pick_top_features_quota(
        mean_abs,
        feature_names=feature_names,
        allowed_mask=allowed_mask,
        top_k_total=top_k_total,
        k_per_mod=k_per_mod,
    )

    # sort by |mean signed SHAP| for visual ranking
    mean_signed = np.nanmean(phi_class[:, feat_idx], axis=0)
    order = np.argsort(np.abs(mean_signed))[::-1]
    feat_idx = feat_idx[order]

    n_features = int(len(feat_idx))
    n_samples = int(phi_class.shape[0])

    labels = [shorten_label(display_names[i], max_chars=24, max_lines=2) for i in feat_idx]
    mods = [_modality_of(feature_names[i]) for i in feat_idx]

    cmap = plt.cm.coolwarm
    y_positions = np.arange(n_features)[::-1]

    # collect shap values for axis limit (robust)
    all_vals = phi_class[:, feat_idx].reshape(-1)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        xmax = 0.01
    else:
        xmax = float(np.nanpercentile(np.abs(all_vals), 99.0) * 1.18)
        xmax = max(xmax, 0.01)

    # draw points feature-by-feature
    for rank, (fidx, ypos) in enumerate(zip(feat_idx, y_positions)):
        shap_vals_full = phi_class[:, int(fidx)]
        feat_vals_full_class = X_class[:, int(fidx)]      # for this class panel points
        feat_vals_full_ref = X_total[:, int(fidx)]        # for robust scaling across all samples

        # downsample deterministically but keep tails (quantile sampling on shap)
        finite_mask = np.isfinite(shap_vals_full) & np.isfinite(feat_vals_full_class)
        shap_vals = shap_vals_full[finite_mask]
        feat_vals = feat_vals_full_class[finite_mask]

        if shap_vals.size == 0:
            continue

        if shap_vals.size > max_display_points:
            # take evenly spaced points in sorted SHAP to preserve distribution
            idx_sort = np.argsort(shap_vals)
            take = np.linspace(0, shap_vals.size - 1, max_display_points).astype(int)
            sel = idx_sort[take]
            shap_vals = shap_vals[sel]
            feat_vals = feat_vals[sel]

        # robust feature-value -> color in [0,1]
        feat_norm = _robust_scale01(feat_vals, ref_values=feat_vals_full_ref, q_low=5.0, q_high=95.0)
        colors = cmap(feat_norm)

        # deterministic beeswarm offsets based on SHAP distribution
        y_off = _beeswarm_offsets_binned(
            shap_vals,
            max_spread=jitter_scale,
            nbins=24,
            seed=seed + 31 * rank + int(class_idx) * 997
        )

        ax.scatter(
            shap_vals,
            ypos + y_off,
            c=colors,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            zorder=4
        )

        # subtle median tick (helps readability, low visual weight)
        med = float(np.nanmedian(shap_vals_full))
        ax.plot(
            [med, med],
            [ypos - 0.18, ypos + 0.18],
            color="#222222",
            linewidth=0.8,
            alpha=0.75,
            zorder=5
        )

        # modality strip (stable)
        mod_color = MODALITY_COLORS.get(mods[rank], "#999999")
        _add_modality_strip(ax, y=float(ypos), color=mod_color)

    # axis styling
    ax.axvline(0, color="#333333", linewidth=0.9, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=8)

    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("SHAP value (impact on model output)", labelpad=3)

    cname = LABEL_NAMES.get(int(class_idx), str(class_idx))
    ax.set_title(
        f"Feature importance ({cname})",
        loc="left",
        pad=4,
        color="black",
        fontweight="normal",
        fontsize=9.6
    )

    ax.xaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.55)
        ax.spines[spine].set_color("#444444")

    return {"n_features": n_features, "n_samples": n_samples, "xmax": xmax}

# -----------------------------
# Plot assembly
# -----------------------------
def plot_signed_shap_summary():
    set_publication_style(font="Helvetica")

    d = load_shap_npz(SHAP_RESULTS_PATH)
    phi_stack = d["phi_stack"]                    # (N,F,C)
    X_total = d["X_total"]                        # (N,F)
    y_true = d["y_true"].astype(int)              # (N,)
    feature_names = d["feature_names"].tolist()
    display_names = d["display_feature_names"].tolist()
    allowed_mask = d["allowed_mask"].astype(bool)

    fig = plt.figure(figsize=(8.27, 6.2))

    # 2-row layout: top row = 3 panels, bottom row = legend
    gs = gridspec.GridSpec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.22],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.75,
        hspace=0.28
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Active
    ax2 = fig.add_subplot(gs[0, 1])  # Stable
    ax3 = fig.add_subplot(gs[0, 2])  # Control

    plot_beeswarm_panel(
        ax1, class_idx=1,
        phi_stack=phi_stack, X_total=X_total, y_true=y_true,
        feature_names=feature_names, display_names=display_names,
        allowed_mask=allowed_mask,
        top_k_total=15, k_per_mod=5,
        point_size=10, alpha=0.72, jitter_scale=0.32, seed=7
    )

    plot_beeswarm_panel(
        ax2, class_idx=0,
        phi_stack=phi_stack, X_total=X_total, y_true=y_true,
        feature_names=feature_names, display_names=display_names,
        allowed_mask=allowed_mask,
        top_k_total=15, k_per_mod=5,
        point_size=10, alpha=0.72, jitter_scale=0.32, seed=7
    )

    plot_beeswarm_panel(
        ax3, class_idx=2,
        phi_stack=phi_stack, X_total=X_total, y_true=y_true,
        feature_names=feature_names, display_names=display_names,
        allowed_mask=allowed_mask,
        top_k_total=15, k_per_mod=5,
        point_size=10, alpha=0.72, jitter_scale=0.32, seed=7
    )

    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")

    handles_mod = []
    labels_mod = []
    for mod in ["gly", "mass", "rna"]:
        h = plt.Line2D([0], [0], color=MODALITY_COLORS.get(mod, "#999999"), lw=5)
        handles_mod.append(h)
        labels_mod.append(MODALITY_PRETTY.get(mod, mod))

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cbar_ax = fig.add_axes([0.38, 0.08, 0.24, 0.025])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Feature value', fontsize=8.5, labelpad=2)
    cbar.outline.set_linewidth(0.5)

    ax_leg.legend(
        handles_mod, labels_mod,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        columnspacing=2.0,
        handlelength=1.8,
        fontsize=8.5
    )

    # Outer margins
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.14)

    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"[Done] Saved: {OUT_PNG}")
    print(f"[Done] Saved: {OUT_PDF}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_signed_shap_summary()
