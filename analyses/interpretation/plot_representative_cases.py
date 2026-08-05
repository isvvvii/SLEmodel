# plot_representative_cases.py
"""Plot representative correct and incorrect classifications with signed SHAP values."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from slemodel import config as cfg


def set_publication_style(font: str = "Helvetica"):
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.85,
            "axes.grid": False,
        }
    )


def _label_to_index_map() -> dict[str, int]:
    return {v: int(k) for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}


def _index_to_label_map() -> dict[int, str]:
    return {int(k): str(v) for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}


def _modality_of(feature_name: str) -> str:
    if isinstance(feature_name, str) and "::" in feature_name:
        return feature_name.split("::", 1)[0]
    return "?"


def shorten_label(label: str, max_chars: int = 26, max_lines: int = 2) -> str:
    if label is None:
        return ""
    s = str(label).strip()
    s = " ".join(s.split())
    s2 = s.replace("-", "- ")
    words = s2.split()

    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        w_clean = w.replace("- ", "-")
        add_len = len(w_clean) + (1 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(w_clean)
            cur_len += add_len
        else:
            lines.append(" ".join(cur).replace("- ", "-"))
            cur = [w_clean]
            cur_len = len(w_clean)
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur).replace("- ", "-"))

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    joined = "\n".join(lines)
    if len(joined.replace("\n", " ")) > max_chars * max_lines:
        if lines:
            last = lines[-1]
            if len(last) > max_chars - 1:
                last = last[: max_chars - 1].rstrip()
            lines[-1] = last + "…"
    return "\n".join(lines)


def load_npz_required(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with np.load(path, allow_pickle=True) as dat:
        return {k: dat[k] for k in dat.files}


def pick_representative_indices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ids: np.ndarray,
    active_idx: int,
    stable_idx: int,
    prefer: str = "most_confident",
) -> tuple[int, int]:
    y_pred = np.argmax(y_prob, axis=1)
    p_active = y_prob[:, active_idx].astype(float)

    mask_ac = (y_true == active_idx) & (y_pred == active_idx)
    if not mask_ac.any():
        raise RuntimeError("No correctly predicted Active sample found.")

    mask_sm = (y_true == stable_idx) & (y_pred == active_idx)
    if not mask_sm.any():
        raise RuntimeError("No Stable→Active misclassified sample found. "
                           "If this is expected, choose a different MoR/fold or relax criteria.")

    def _choose(mask: np.ndarray) -> int:
        idxs = np.where(mask)[0]
        if prefer == "most_confident":
            return int(idxs[np.argmax(p_active[idxs])])
        if prefer == "median_confidence":
            order = idxs[np.argsort(p_active[idxs])]
            return int(order[len(order) // 2])
        return int(idxs[0])

    return _choose(mask_ac), _choose(mask_sm)


def compute_modality_contributions(abs_shap: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    mods = np.array([_modality_of(n) for n in feature_names], dtype=object)
    out: dict[str, float] = {}
    for mod in ["gly", "mass", "rna"]:
        idx = np.where(mods == mod)[0]
        out[mod] = float(abs_shap[idx].sum()) if idx.size else 0.0
    total = sum(out.values()) + 1e-12
    out["total"] = float(total)
    return out


def top_features_for_active_case(
    shap_active: np.ndarray,
    feature_names: list[str],
    display_names: list[str],
    allowed_mask: np.ndarray,
    top_n: int = 12,
    per_modality_quota: int | None = 4,
) -> list[int]:
    shap_active = np.asarray(shap_active, dtype=float).reshape(-1)
    allowed_mask = np.asarray(allowed_mask, dtype=bool).reshape(-1)

    idx_allowed = np.where(allowed_mask)[0]
    if idx_allowed.size == 0:
        idx_allowed = np.arange(len(shap_active))

    abs_vals = np.abs(shap_active)

    if per_modality_quota is None:
        order = idx_allowed[np.argsort(abs_vals[idx_allowed])[::-1]]
        return [int(i) for i in order[:top_n]]

    mods = np.array([_modality_of(n) for n in feature_names], dtype=object)
    picked: list[int] = []
    for mod in ["gly", "mass", "rna"]:
        idx = idx_allowed[mods[idx_allowed] == mod]
        if idx.size == 0:
            continue
        ord_mod = idx[np.argsort(abs_vals[idx])[::-1]]
        picked.extend([int(i) for i in ord_mod[:per_modality_quota]])

    picked = list(dict.fromkeys(picked))
    if len(picked) < top_n:
        order = idx_allowed[np.argsort(abs_vals[idx_allowed])[::-1]]
        for i in order:
            ii = int(i)
            if ii not in picked:
                picked.append(ii)
            if len(picked) >= top_n:
                break

    picked = sorted(picked, key=lambda i: abs_vals[i], reverse=True)[:top_n]
    return picked

def plot_header(ax, sample_id: str, true_name: str, pred_name: str, p_active: float, is_misclassified: bool):
    ax.axis("off")
    title = f"Sample {sample_id}"
    subtitle = f"True: {true_name}   Pred: {pred_name}   p(Active)={p_active:.3f}"
    ax.text(
        0.0,
        0.88,
        title,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.12,
        subtitle,
        ha="left",
        va="center",
        fontsize=8.8,
        color=("#B00020" if is_misclassified else "#222222"),
        transform=ax.transAxes,
    )

def plot_probabilities(ax, probs: np.ndarray, idx2label: dict[int, str], class_colors: dict[int, str]):
    C = probs.shape[0]
    xs = np.arange(C)
    colors = [class_colors.get(i, "#999999") for i in range(C)]
    bars = ax.bar(xs, probs, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [idx2label.get(i, str(i)) for i in range(C)],
        rotation=0,
        ha="center",
        fontweight="bold",
    )

    ax.set_ylabel("Probability", labelpad=4)
    ax.set_title("Predicted probabilities", loc="left", pad=10)

    # Increase tick-label padding to avoid overlap with the adjacent panel.
    ax.tick_params(axis="x", pad=6)

    for b, p in zip(bars, probs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(1.02, p + 0.04),
            f"{p:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.axhline(0, color="#333333", linewidth=0.8)


def plot_signed_shap_active(
    ax,
    shap_active: np.ndarray,
    x_row: np.ndarray,
    feature_idx: list[int],
    feature_names: list[str],
    display_names: list[str],
    modality_colors: dict[str, str],
):
    shap_active = np.asarray(shap_active, dtype=float).reshape(-1)
    x_row = np.asarray(x_row, dtype=float).reshape(-1)

    idx = np.array(feature_idx, dtype=int)
    vals = shap_active[idx]
    zvals = x_row[idx]

    # order by |SHAP|
    order = np.argsort(np.abs(vals))[::-1]
    idx = idx[order]
    vals = vals[order]
    zvals = zvals[order]

    labels = [shorten_label(display_names[i], max_chars=26, max_lines=2) for i in idx]
    mods = [_modality_of(feature_names[i]) for i in idx]

    # fill encodes direction; edge encodes modality
    pos_color = "#C96963"  # muted red
    neg_color = "#4A6A91"  # muted blue
    facecolors = [pos_color if v >= 0 else neg_color for v in vals]
    edgecolors = [modality_colors.get(m, "#333333") for m in mods]

    y = np.arange(len(idx))[::-1]
    ax.barh(y, vals, color=facecolors, edgecolor=edgecolors, linewidth=1.05, alpha=0.92, height=0.62)
    ax.axvline(0.0, color="#111111", linewidth=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.tick_params(axis="y", length=0, pad=7)

    ax.set_xlabel("Signed SHAP to Active logit", labelpad=3)
    ax.set_title("Top drivers (signed SHAP → Active)", loc="left", pad=4)

    # Place z-score annotations in a separate column.
    # Using y-axis transform: x in axes coords, y in data coords
    ax.text(
        1.01,
        1.01,
        "Feature z",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
        fontweight="bold",
        clip_on=False,
    )
    for yy, z in zip(y, zvals):
        ax.text(
            1.01,
            yy,
            f"z={z:+.2f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.2,
            color="#555555",
            clip_on=False,
        )

    # subtle x grid
    ax.xaxis.grid(True, linestyle=":", linewidth=0.75, alpha=0.35)
    ax.set_axisbelow(True)

    # Full box spines (NC-like)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.65)
        ax.spines[spine].set_color("#333333")

def plot_modality_donut(ax, shap_active: np.ndarray, feature_names: list[str], modality_colors: dict[str, str]):
    """
    Modality contributions (donut) with leader lines.
    - Donut radius: 3.375
    - Shifted downward for improved visual balance
    - Labels with leader lines positioned proportionally
    - Caption anchored at bottom
    """
    abs_shap = np.abs(np.asarray(shap_active, dtype=float).reshape(-1))
    pack = compute_modality_contributions(abs_shap, feature_names)

    mods = ["gly", "mass", "rna"]
    vals = np.array([pack.get(m, 0.0) for m in mods], dtype=float)
    total = float(pack.get("total", vals.sum() + 1e-12))
    props = vals / (total + 1e-12)

    pretty = cfg.VisualizationConfig.MODALITY_NAMES
    colors = [modality_colors.get(m, "#999999") for m in mods]

    ax.set_title("Modality contributions", loc="left", pad=4)
    ax.axis("off")

    # ── Inset axes: occupy nearly full panel, shifted down ──
    w, h = 1.00, 0.94
    x0 = (1.0 - w) / 2.0
    y0 = -0.04
    ax_pie = ax.inset_axes([x0, y0, w, h])

    # ── Enlarge donut: 2.25 × 1.5 = 3.375 ──
    R = 3.375
    wedges, _ = ax_pie.pie(
        props,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=R,
        wedgeprops=dict(width=0.52 * R, edgecolor="white", linewidth=1.4),
    )
    ax_pie.set_aspect("equal")
    ax_pie.set_xticks([])
    ax_pie.set_yticks([])

    # ── Viewport: tight limits to maximize visual size ──
    xlim = R + 0.85
    ylim_top = R + 0.75
    ylim_bot = R + 0.45
    ax_pie.set_xlim(-xlim, xlim)
    ax_pie.set_ylim(-ylim_bot, ylim_top)

    # ── Labels with leader lines (scaled for larger R) ──
    for wdg, m, p in zip(wedges, mods, props):
        ang = 0.5 * (wdg.theta1 + wdg.theta2)
        ang_rad = np.deg2rad(ang)
        x = np.cos(ang_rad)
        y = np.sin(ang_rad)

        # Anchor on outer edge of wedge
        xy = (x * R, y * R)

        # Text position: scaled proportionally with R
        x_text = (R + 0.55) * (1.0 if x >= 0 else -1.0)
        y_text = (R + 0.15) * y
        ha = "left" if x >= 0 else "right"

        label = f"{pretty.get(m, m)}  {p * 100:.1f}%"

        fc = wdg.get_facecolor()
        line_color = (fc[0], fc[1], fc[2], 0.85)

        ax_pie.annotate(
            label,
            xy=xy,
            xytext=(x_text, y_text),
            ha=ha,
            va="center",
            fontsize=10.0,
            color="#222222",
            arrowprops=dict(
                arrowstyle="-",
                lw=1.2,
                color=line_color,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.08",
            ),
            annotation_clip=False,
        )

    # ── Caption anchored at panel bottom ──
    ax.text(
        0.5,
        -0.06,
        "Share of within-sample Σ|SHAP| (modality-level)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#666666",
    )

def plot_representative_cases(
    shap_npz: Path,
    infer_npz: Path,
    out_png: Path,
    out_pdf: Path,
    top_features: int = 12,
    per_modality_quota: int = 4,
    prefer: str = "most_confident",
    active_name: str = "Active",
    stable_name: str = "Stable",
):
    set_publication_style(font="Helvetica")

    d_shap = load_npz_required(shap_npz)
    d_inf = load_npz_required(infer_npz)

    phi_stack = d_shap["phi_stack"]  # (N,F,C)
    X_total = d_shap["X_total"]      # (N,F)
    y_true_shap = d_shap["y_true"].astype(int)
    feature_names = d_shap["feature_names"].tolist()
    display_names = d_shap["display_feature_names"].tolist()
    allowed_mask = d_shap.get("allowed_mask", np.ones(len(feature_names), dtype=bool)).astype(bool)

    y_true_inf = d_inf["y_true"].astype(int)
    y_prob = d_inf["y_prob"].astype(float)
    ids = d_inf.get("ids", np.arange(y_prob.shape[0])).astype(object)

    if phi_stack.shape[0] != y_prob.shape[0]:
        raise RuntimeError(
            f"SHAP N ({phi_stack.shape[0]}) != inference N ({y_prob.shape[0]}). "
            "Alignment cannot be recovered safely."
        )
    if not np.array_equal(y_true_shap, y_true_inf):
        warnings.warn("y_true mismatch between SHAP and inference NPZ; proceeding by index.")

    idx2label = _index_to_label_map()
    label2idx = _label_to_index_map()

    if active_name not in label2idx or stable_name not in label2idx:
        raise ValueError(f"active_name/stable_name not found in LABEL_NAMES: {cfg.VisualizationConfig.LABEL_NAMES}")

    active_idx = int(label2idx[active_name])
    stable_idx = int(label2idx[stable_name])

    idx_active_correct, idx_stable_misactive = pick_representative_indices(
        y_true_inf, y_prob, ids, active_idx=active_idx, stable_idx=stable_idx, prefer=prefer
    )

    chosen = [
        ("True Active, Pred Active", int(idx_active_correct)),
        ("True Stable, Pred Active", int(idx_stable_misactive)),
    ]

    class_colors = {int(k): cfg.VisualizationConfig.LABEL_COLORS.get(int(k), "#999999") for k in idx2label.keys()}
    modality_colors = cfg.VisualizationConfig.MODALITY_COLORS

    # Slightly taller figure to relieve vertical collisions
    fig = plt.figure(figsize=(10.6, 7.4))
    outer = gridspec.GridSpec(1, 2, wspace=0.40)

    for col, (_tag, idx) in enumerate(chosen):
        inner = gridspec.GridSpecFromSubplotSpec(
            4,
            1,
            subplot_spec=outer[col],
            height_ratios=[0.26, 0.82, 1.85, 1.20],
            hspace=0.82,
        )

        ax_h = fig.add_subplot(inner[0])
        ax_p = fig.add_subplot(inner[1])
        ax_s = fig.add_subplot(inner[2])
        ax_m = fig.add_subplot(inner[3])

        sid = str(ids[idx])
        true_idx = int(y_true_inf[idx])
        pred_idx = int(np.argmax(y_prob[idx]))
        true_name = idx2label.get(true_idx, str(true_idx))
        pred_name = idx2label.get(pred_idx, str(pred_idx))
        p_active = float(y_prob[idx, active_idx])
        is_mis = (true_idx != pred_idx)

        plot_header(ax_h, sid, true_name, pred_name, p_active, is_misclassified=is_mis)
        plot_probabilities(ax_p, y_prob[idx], idx2label, class_colors)

        shap_active = phi_stack[idx, :, active_idx]
        feat_idx = top_features_for_active_case(
            shap_active=shap_active,
            feature_names=feature_names,
            display_names=display_names,
            allowed_mask=allowed_mask,
            top_n=top_features,
            per_modality_quota=per_modality_quota,
        )

        # Draw top drivers with z-score annotations in the outer column.
        plot_signed_shap_active(
            ax_s,
            shap_active=shap_active,
            x_row=X_total[idx],
            feature_idx=feat_idx,
            feature_names=feature_names,
            display_names=display_names,
            modality_colors=modality_colors,
        )

        plot_modality_donut(ax_m, shap_active=shap_active, feature_names=feature_names, modality_colors=modality_colors)

        if is_mis:
            for ax in [ax_p, ax_s]:
                for sp in ax.spines.values():
                    sp.set_linewidth(1.15)
                    sp.set_color("#B00020")

    handles = [
        Line2D([0], [0], color="#C96963", lw=6, label="Pushes toward Active (SHAP > 0)"),
        Line2D([0], [0], color="#4A6A91", lw=6, label="Pushes away from Active (SHAP < 0)"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=modality_colors.get("gly", "#999999"),
               markeredgecolor="white", markeredgewidth=1.0, markersize=10,
               label=cfg.VisualizationConfig.MODALITY_NAMES.get("gly", "Glycan")),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=modality_colors.get("mass", "#999999"),
               markeredgecolor="white", markeredgewidth=1.0, markersize=10,
               label=cfg.VisualizationConfig.MODALITY_NAMES.get("mass", "Mass")),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=modality_colors.get("rna", "#999999"),
               markeredgecolor="white", markeredgewidth=1.0, markersize=10,
               label=cfg.VisualizationConfig.MODALITY_NAMES.get("rna", "RNA")),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )

    fig.suptitle(
        "Representative cases",
        y=0.985,
        fontsize=11.5,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.15)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("[OK] Saved:", out_png)
    print("[OK] Saved:", out_pdf)
    print("[Selected samples]")
    print("  Active→Active:", ids[idx_active_correct])
    print("  Stable→Active:", ids[idx_stable_misactive])

def main():
    ap = argparse.ArgumentParser(description="Plot representative classifications and their SHAP values.")
    ap.add_argument("--shap_npz", type=str, default="clinical_explain_results/shap_outputs/mor_shap_outputs.npz")
    ap.add_argument("--infer_npz", type=str, default="clinical_explain_results/shap_outputs/mor_inference_results.npz")
    ap.add_argument("--outdir", type=str, default="clinical_explain_results/representative_cases")
    ap.add_argument("--top_features", type=int, default=12)
    ap.add_argument("--per_modality_quota", type=int, default=4)
    ap.add_argument("--prefer", type=str, default="most_confident", choices=["most_confident", "median_confidence"])
    ap.add_argument("--active_name", type=str, default="Active")
    ap.add_argument("--stable_name", type=str, default="Stable")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    out_png = outdir / "representative_cases.png"
    out_pdf = outdir / "representative_cases.pdf"

    plot_representative_cases(
        shap_npz=Path(args.shap_npz),
        infer_npz=Path(args.infer_npz),
        out_png=out_png,
        out_pdf=out_pdf,
        top_features=args.top_features,
        per_modality_quota=args.per_modality_quota,
        prefer=args.prefer,
        active_name=args.active_name,
        stable_name=args.stable_name,
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
