# plot_ablation_results.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import t


def apply_publication_style_helvetica():
    """
    Matplotlib style used for the journal figure.
    Helvetica is widely accepted for Nature Portfolio figures.
    """
    plt.style.use("default")
    plt.rcParams.update(
        {
            # --- Fonts ---
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # --- Sizes (tuned for 7x6 inch figure) ---
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,

            # --- Lines ---
            "figure.dpi": 300,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
        }
    )


def add_panel_label_aligned_to_ylabel(fig, ax, label: str, *, y_pad: float = 0.006):
    """
    Place panel label (a/b/c) aligned to the LEFT EDGE of the y-axis label text,
    rather than to the y-axis spine or tick labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ylab = ax.yaxis.label
    if ylab is None or ylab.get_text() == "":
        x_fig = ax.get_position().x0
    else:
        bbox = ylab.get_window_extent(renderer=renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        x_fig = bbox_fig.x0

    y_fig = ax.get_position().y1 + float(y_pad)

    fig.text(
        x_fig,
        y_fig,
        label,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def auto_zoom_limits_near_origin(x, y, *, keep_frac: float = 0.75, pad_frac: float = 0.18):
    """
    Automatically choose zoom-in limits around the dense cluster near the origin.

    Strategy:
    - compute score = |x| + |y|
    - keep the smallest keep_frac points (closest to origin)
    - derive limits from those points (+ include 0) and pad a bit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 4:
        # fallback: just use min/max
        x0, x1 = np.min(x), np.max(x)
        y0, y1 = np.min(y), np.max(y)
    else:
        score = np.abs(x) + np.abs(y)
        k = max(4, int(np.ceil(x.size * float(keep_frac))))
        idx = np.argsort(score)[:k]
        xs, ys = x[idx], y[idx]
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))

    # always include 0 in inset
    x0, x1 = min(x0, 0.0), max(x1, 0.0)
    y0, y1 = min(y0, 0.0), max(y1, 0.0)

    # pad
    dx = (x1 - x0) if (x1 - x0) > 1e-12 else 1e-2
    dy = (y1 - y0) if (y1 - y0) > 1e-12 else 1e-2
    xlim = (x0 - dx * pad_frac, x1 + dx * pad_frac)
    ylim = (y0 - dy * pad_frac, y1 + dy * pad_frac)

    return xlim, ylim

def choose_inset_bbox_axes(
    ax,
    x,
    y,
    *,
    inset_w: float = 0.42,
    inset_h: float = 0.42,
    margin: float = 0.03,
):
    """
    Choose an inset location (in axes fraction coordinates) that minimally occludes the data.

    We test 4 candidate corners (UL/UR/LL/LR). For each, count how many points fall inside
    the candidate inset bbox in axes coordinates, and pick the smallest count.

    Returns:
        (x0, y0, w, h) in ax.transAxes coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Transform all data points into axes coordinates [0,1]x[0,1]
    pts_disp = ax.transData.transform(np.column_stack([x, y]))
    pts_axes = ax.transAxes.inverted().transform(pts_disp)
    xa, ya = pts_axes[:, 0], pts_axes[:, 1]

    candidates = {
        "UL": (margin, 1.0 - margin - inset_h, inset_w, inset_h),
        "UR": (1.0 - margin - inset_w, 1.0 - margin - inset_h, inset_w, inset_h),
        "LL": (margin, margin, inset_w, inset_h),
        "LR": (1.0 - margin - inset_w, margin, inset_w, inset_h),
    }

    def count_inside(b):
        x0, y0, w, h = b
        inside = (xa >= x0) & (xa <= x0 + w) & (ya >= y0) & (ya <= y0 + h)
        return int(np.sum(inside))

    # Pick the corner with minimum point overlap; tie-breaker prefers upper corners a bit
    scores = {k: count_inside(v) for k, v in candidates.items()}
    # deterministic tie-break order
    order = ["UL", "UR", "LL", "LR"]
    best_key = sorted(scores.keys(), key=lambda k: (scores[k], order.index(k)))[0]
    return candidates[best_key]

def add_zoom_inset(
    ax,
    x,
    y,
    colors,
    xlim,
    ylim,
    *,
    bbox_axes=None,
    inset_w: float = 0.40,
    inset_h: float = 0.40,
    background_alpha: float = 0.35,
    edge_color: str = "0.35",
    edge_lw: float = 0.7,
    show_main_region: bool = True,
    region_color: str = "0.35",
    region_lw: float = 0.9,
    region_ls: str = "--",
    connect: bool = True,
    connect_color: str = "0.55",
    connect_lw: float = 0.7,
    connect_alpha: float = 0.9,
    connector_spread: float = 0.35,
):
    """
    Circular zoom inset with an elliptical region marker and two connectors.
    Annotation text may extend beyond the circular clipping region.
    """

    def _unit(v):
        v = np.asarray(v, dtype=float)
        n = float(np.hypot(v[0], v[1]))
        if n < 1e-12:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    def _perp(v):
        v = np.asarray(v, dtype=float)
        return np.array([-v[1], v[0]], dtype=float)

    def _ellipse_boundary_point(cx, cy, rx, ry, direction):
        d = _unit(direction)
        dx, dy = float(d[0]), float(d[1])
        denom = (dx / rx) ** 2 + (dy / ry) ** 2
        if denom < 1e-12:
            return (cx, cy)
        t = 1.0 / np.sqrt(denom)
        return (cx + t * dx, cy + t * dy)

    def _circle_boundary_point(center_xy, radius, direction_xy):
        d = _unit(direction_xy)
        return (center_xy[0] + radius * d[0], center_xy[1] + radius * d[1])

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    colors = np.asarray(colors)[mask]

    # inset placement
    if bbox_axes is None:
        x0 = 0.5 - inset_w / 2.0
        y0 = 0.5 - inset_h / 2.0
        bbox_axes = (x0, y0, inset_w, inset_h)
    x0, y0, w, h = bbox_axes

    # inset axes
    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(x0, y0, w, h),
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0.0,
    )
    axins.set_zorder(10)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axins.set_frame_on(False)
    axins.patch.set_alpha(0.0)
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)

    # axins.set_clip_on(False)

    circle = patches.Circle(
        (0.5, 0.5),
        radius=0.5,
        transform=axins.transAxes,
        facecolor=(1, 1, 1, background_alpha),
        edgecolor=edge_color,
        linewidth=edge_lw,
        zorder=2,
    )
    axins.add_patch(circle)

    sc = axins.scatter(
        x, y,
        c=colors,
        s=18,
        alpha=0.95,
        edgecolors="white",
        linewidth=0.35,
        zorder=3,
    )
    sc.set_clip_path(circle)

    hline = axins.axhline(0, color="0.55", ls="--", lw=0.6, zorder=3)
    vline = axins.axvline(0, color="0.55", ls="--", lw=0.6, zorder=3)
    hline.set_clip_path(circle)
    vline.set_clip_path(circle)

    # ROI ellipse on main axes
    xA0, xA1 = float(xlim[0]), float(xlim[1])
    yA0, yA1 = float(ylim[0]), float(ylim[1])
    cx = (xA0 + xA1) / 2.0
    cy = (yA0 + yA1) / 2.0
    rx = max((xA1 - xA0) / 2.0, 1e-12)
    ry = max((yA1 - yA0) / 2.0, 1e-12)

    if show_main_region:
        region_patch = patches.Ellipse(
            (cx, cy),
            width=2 * rx,
            height=2 * ry,
            fill=False,
            ec=region_color,
            lw=region_lw,
            ls=region_ls,
            zorder=6,
        )
        ax.add_patch(region_patch)

    # connectors
    if connect:
        inset_center_axes = np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=float)
        inset_center_disp = ax.transAxes.transform(inset_center_axes)
        inset_center_data = ax.transData.inverted().transform(inset_center_disp)

        base_d = _unit(inset_center_data - np.array([cx, cy], dtype=float))
        base_p = _perp(base_d)

        d1 = _unit(base_d + connector_spread * base_p)
        d2 = _unit(base_d - connector_spread * base_p)

        p1 = _ellipse_boundary_point(cx, cy, rx, ry, d1)
        p2 = _ellipse_boundary_point(cx, cy, rx, ry, d2)

        roi_center_disp = ax.transData.transform(np.array([cx, cy], dtype=float))
        roi_center_inset_axes = axins.transAxes.inverted().transform(roi_center_disp)
        v_to_roi = _unit(roi_center_inset_axes - np.array([0.5, 0.5], dtype=float))
        v_perp = _perp(v_to_roi)

        w1 = _unit(v_to_roi + connector_spread * v_perp)
        w2 = _unit(v_to_roi - connector_spread * v_perp)

        q1 = _circle_boundary_point((0.5, 0.5), 0.5, w1)
        q2 = _circle_boundary_point((0.5, 0.5), 0.5, w2)

        con1 = patches.ConnectionPatch(
            xyA=p1, coordsA=ax.transData,
            xyB=q1, coordsB=axins.transAxes,
            axesA=ax, axesB=axins,
            color=connect_color, lw=connect_lw, ls="--", alpha=connect_alpha,
            zorder=9,
        )
        con2 = patches.ConnectionPatch(
            xyA=p2, coordsA=ax.transData,
            xyB=q2, coordsB=axins.transAxes,
            axesA=ax, axesB=axins,
            color=connect_color, lw=connect_lw, ls="--", alpha=connect_alpha,
            zorder=9,
        )
        ax.add_artist(con1)
        ax.add_artist(con2)

    return axins

def annotate_callouts_on_main(
    ax,
    points_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    *,
    offset_pts: int = 10,
    border_frac: float = 0.08,
    text_kwargs: dict | None = None,
    arrowprops: dict | None = None,
    avoid_axes: list | None = None,
    avoid_points_df: pd.DataFrame | None = None,
    avoid_x_col: str | None = None,
    avoid_y_col: str | None = None,
    avoid_pad_px: float = 3.0,
    custom_offsets: dict[str, tuple] | None = None,
):
    if points_df is None or points_df.empty:
        return

    if text_kwargs is None:
        text_kwargs = dict(fontsize=7, color="0.15", zorder=8)
    if arrowprops is None:
        arrowprops = dict(arrowstyle="-", color="0.55", lw=0.6, alpha=0.95, shrinkA=0, shrinkB=0)

    base_arrow = dict(arrowprops)
    base_arrow["shrinkA"] = 0
    base_arrow["shrinkB"] = 0

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    placed_bboxes = []

    avoid_bboxes = []
    if avoid_axes:
        for a in avoid_axes:
            try:
                bb = a.get_window_extent(renderer=renderer).padded(2.0)
                avoid_bboxes.append(bb)
            except Exception:
                pass

    avoid_pts = None
    if avoid_points_df is not None and avoid_x_col and avoid_y_col and not avoid_points_df.empty:
        av = avoid_points_df[[avoid_x_col, avoid_y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if not av.empty:
            avoid_pts = ax.transData.transform(av.to_numpy(dtype=float))

    def bbox_overlaps_points(bbox):
        if avoid_pts is None:
            return False
        bb = bbox.padded(avoid_pad_px)
        inside = (
            (avoid_pts[:, 0] >= bb.x0) & (avoid_pts[:, 0] <= bb.x1) &
            (avoid_pts[:, 1] >= bb.y0) & (avoid_pts[:, 1] <= bb.y1)
        )
        return bool(np.any(inside))

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = float(x1 - x0)
    yr = float(y1 - y0)
    mx = border_frac * xr
    my = border_frac * yr

    for _, row in points_df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        lab = str(row[label_col])

        if custom_offsets and lab in custom_offsets:
            dx, dy, ha, va = custom_offsets[lab]
            arrow_i = dict(base_arrow)
            arrow_i["connectionstyle"] = "arc3,rad=0.0"

            ann = ax.annotate(
                lab,
                xy=(x, y),
                xycoords="data",
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                clip_on=True,
                annotation_clip=True,
                arrowprops=arrow_i,
                **text_kwargs,
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).padded(2.0)
            placed_bboxes.append(bbox)
            continue
        # --------------------------------------------------------

        if x <= x0 + mx:
            dx0, ha0 = +offset_pts, "left"
        elif x >= x1 - mx:
            dx0, ha0 = -offset_pts, "right"
        else:
            dx0 = +offset_pts if x >= 0 else -offset_pts
            ha0 = "left" if dx0 > 0 else "right"

        if y <= y0 + my:
            dy0, va0 = +offset_pts, "bottom"
        elif y >= y1 - my:
            dy0, va0 = -offset_pts, "top"
        else:
            dy0 = +offset_pts if y >= 0 else -offset_pts
            va0 = "bottom" if dy0 > 0 else "top"

        candidates = [
            (dx0, dy0, ha0, va0),
            (dx0, int(dy0 * 0.7), ha0, va0),
            (int(dx0 * 0.7), dy0, ha0, va0),
            (dx0, -dy0, ha0, "top" if va0 == "bottom" else "bottom"),
            (-dx0, dy0, "right" if ha0 == "left" else "left", va0),
            (-dx0, -dy0, "right" if ha0 == "left" else "left", "top" if va0 == "bottom" else "bottom"),
        ]

        best_ann = None
        best_bbox = None

        for k, (dx, dy, ha, va) in enumerate(candidates):
            rad = 0.05 if (k % 2 == 0) else -0.05
            arrow_i = dict(base_arrow)
            arrow_i["connectionstyle"] = f"arc3,rad={rad}"

            ann = ax.annotate(
                lab,
                xy=(x, y),
                xycoords="data",
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                clip_on=True,
                annotation_clip=True,
                arrowprops=arrow_i,
                **text_kwargs,
            )

            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).padded(2.0)

            if (bbox.x0 < ax_bbox.x0) or (bbox.x1 > ax_bbox.x1) or (bbox.y0 < ax_bbox.y0) or (bbox.y1 > ax_bbox.y1):
                ann.remove()
                continue

            if any(bbox.overlaps(b2) for b2 in placed_bboxes):
                ann.remove()
                continue

            if any(bbox.overlaps(bb) for bb in avoid_bboxes):
                ann.remove()
                continue

            if bbox_overlaps_points(bbox):
                ann.remove()
                continue

            best_ann = ann
            best_bbox = bbox
            break

        if best_ann is None:
            dx, dy, ha, va = candidates[0]
            arrow_i = dict(base_arrow)
            arrow_i["connectionstyle"] = "arc3,rad=0.0"
            best_ann = ax.annotate(
                lab,
                xy=(x, y),
                xycoords="data",
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                clip_on=True,
                annotation_clip=True,
                arrowprops=arrow_i,
                **text_kwargs,
            )
            fig.canvas.draw()
            best_bbox = best_ann.get_window_extent(renderer=renderer).padded(2.0)

        placed_bboxes.append(best_bbox)

def _mask_in_limits(df, x_col, y_col, xlim, ylim):
    return (
        df[x_col].between(float(xlim[0]), float(xlim[1]))
        & df[y_col].between(float(ylim[0]), float(ylim[1]))
    )


def _pick_by_score(df, n, score_series, *, exclude_codes=None, ascending=True):
    if df is None or df.empty or n <= 0:
        return df.iloc[0:0].copy()

    out = df.copy()
    if exclude_codes:
        out = out[~out["Code"].isin(set(exclude_codes))].copy()
    if out.empty:
        return out

    out["_score_"] = score_series.loc[out.index].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["_score_"])
    out = out.sort_values("_score_", ascending=ascending)
    out = out.head(int(n)).drop(columns=["_score_"])
    return out


def _score_to_origin(df, x_col, y_col):
    return np.hypot(df[x_col].astype(float), df[y_col].astype(float))


def _score_to_corner(df, x_col, y_col, corner_x, corner_y):
    return np.hypot(df[x_col].astype(float) - float(corner_x), df[y_col].astype(float) - float(corner_y))


def _select_inset_points_panel_b(
    plot_data_b,
    x_col,
    y_col,
    xlim_b,
    ylim_b,
    *,
    n_total_inset: int = 6,
):
    """Select representative points for the panel-b inset."""
    mask = _mask_in_limits(plot_data_b, x_col, y_col, xlim_b, ylim_b)
    df_inset = plot_data_b[mask].copy()

    if df_inset.empty:
        return df_inset

    selected = []
    exclude = set()

    df_inset["_dist_"] = _score_to_origin(df_inset, x_col, y_col)

    better = df_inset[
        (df_inset[x_col] > 0) | (df_inset[y_col] > 0)
    ].sort_values("_dist_", ascending=False)

    n_better = int(np.ceil(n_total_inset / 2))
    if not better.empty:
        curr = better.head(n_better)
        selected.append(curr)
        exclude.update(curr["Code"].tolist())

    remain = df_inset[~df_inset["Code"].isin(exclude)].sort_values("_dist_", ascending=True)

    current_count = sum(len(x) for x in selected)
    n_remain = n_total_inset - current_count

    if n_remain > 0 and not remain.empty:
        selected.append(remain.head(n_remain))

    if not selected:
        return df_inset.iloc[0:0].copy()

    return pd.concat(selected, ignore_index=True).drop_duplicates(subset=["Code"])

def _select_main_outliers(plot_data, x_col, y_col, xlim, ylim, *, n=2, exclude_codes=None):
    """
    Select the points outside the inset limits that are farthest from the origin.

    Args:
        plot_data: Complete plotting data.
        x_col, y_col: Coordinate columns.
        xlim, ylim: Inset limits.
        n: Number of points to select.
        exclude_codes: Experiment codes to exclude.

    Returns:
        DataFrame with selected outlier points
    """
    mask_outside = ~_mask_in_limits(plot_data, x_col, y_col, xlim, ylim)
    df = plot_data[mask_outside].copy()

    if exclude_codes:
        df = df[~df["Code"].isin(set(exclude_codes))].copy()

    if df.empty:
        return df

    dist = _score_to_origin(df, x_col, y_col)
    return _pick_by_score(df, n, dist, ascending=False)

def _select_inset_points_panel_c(
    plot_data_c,
    x_col,
    y_col,
    xlim_c,
    ylim_c,
    *,
    n_total_inset: int = 6,
):
    """Select representative points for the panel-c inset."""
    mask = _mask_in_limits(plot_data_c, x_col, y_col, xlim_c, ylim_c)
    df_inset = plot_data_c[mask].copy()

    if df_inset.empty:
        return df_inset

    selected = []
    exclude = set()

    df_inset["_dist_"] = _score_to_origin(df_inset, x_col, y_col)

    better = df_inset[
        (df_inset[x_col] > 0) | (df_inset[y_col] > 0)
    ].sort_values("_dist_", ascending=False)

    n_better = int(np.ceil(n_total_inset / 2))
    if not better.empty:
        curr = better.head(n_better)
        selected.append(curr)
        exclude.update(curr["Code"].tolist())

    remain = df_inset[~df_inset["Code"].isin(exclude)].sort_values("_dist_", ascending=True)

    current_count = sum(len(x) for x in selected)
    n_remain = n_total_inset - current_count

    if n_remain > 0 and not remain.empty:
        selected.append(remain.head(n_remain))

    if not selected:
        return df_inset.iloc[0:0].copy()

    return pd.concat(selected, ignore_index=True).drop_duplicates(subset=["Code"])


def annotate_callouts_in_inset(
    axins,
    points_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    parent_ax,
    *,
    xlim,
    ylim,
    r: float = 0.65,
    axes_margin: float = 0.05,
    n_anchors: int | None = None,
    text_kwargs: dict | None = None,
    arrowprops: dict | None = None,
    bbox_pad_px: float = 2.0,
    max_trials_per_point: int = 100,
    custom_directions: dict[str, str] | None = None,
):
    if points_df is None or points_df.empty:
        return

    if text_kwargs is None:
        text_kwargs = dict(fontsize=7, color="0.15", zorder=20)

    final_text_kwargs = text_kwargs.copy()
    final_text_kwargs["annotation_clip"] = False
    final_text_kwargs["clip_on"] = False

    if arrowprops is None:
        arrowprops = dict(arrowstyle="-", color="0.55", lw=0.6, alpha=0.95, shrinkA=0, shrinkB=0)
    base_arrow = dict(arrowprops)
    base_arrow["shrinkA"] = 0
    base_arrow["shrinkB"] = 0

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    sub = points_df[[x_col, y_col, label_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    sub = sub[
        (sub[x_col] >= x0) & (sub[x_col] <= x1) &
        (sub[y_col] >= y0) & (sub[y_col] <= y1)
    ].copy()

    if sub.empty:
        return

    norm_x = (sub[x_col] - x0) / (x1 - x0)
    norm_y = (sub[y_col] - y0) / (y1 - y0)
    mask_circle = ((norm_x - 0.5)**2 + (norm_y - 0.5)**2) <= (0.51**2)
    sub = sub[mask_circle]

    if sub.empty:
        return

    xs = sub[x_col].to_numpy(dtype=float)
    ys = sub[y_col].to_numpy(dtype=float)
    labels = sub[label_col].astype(str).to_list()

    dir_map = {
        "right": 0.0,
        "top": np.pi/2,
        "left": np.pi,
        "bottom": -np.pi/2
    }

    angles = np.arctan2(ys - cy, xs - cx)

    if n_anchors is None:
        n_anchors = max(48, len(labels) * 15)
    anchor_thetas = np.linspace(-np.pi, np.pi, n_anchors, endpoint=False)

    def ang_dist(a, b):
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return float(abs(d))

    fig = axins.figure
    renderer = fig.canvas.get_renderer()
    parent_bbox = parent_ax.get_window_extent(renderer=renderer)

    placed_bboxes = []

    order = np.argsort(angles)
    angles_s = angles[order]
    xs_s = xs[order]
    ys_s = ys[order]
    labels_s = [labels[i] for i in order]

    r_list = [r, r + 0.15, r + 0.30, r - 0.1, r + 0.45]

    for i, (ang, px, py, lab) in enumerate(zip(angles_s, xs_s, ys_s, labels_s)):
        custom_dir = custom_directions.get(lab) if custom_directions else None

        if custom_dir and custom_dir in dir_map:
            target_ang = dir_map[custom_dir]
            valid_anchors = []
            for j, th in enumerate(anchor_thetas):
                if ang_dist(th, target_ang) < (np.pi / 4):
                    valid_anchors.append(j)
            anchor_idx = sorted(valid_anchors, key=lambda idx: ang_dist(anchor_thetas[idx], target_ang))
        else:
            anchor_idx = np.argsort([ang_dist(th, ang) for th in anchor_thetas])

        best_ann = None
        best_bbox = None

        for rr in r_list:
            for k_idx in range(min(len(anchor_idx), max_trials_per_point)):
                j = anchor_idx[k_idx]
                th = float(anchor_thetas[j])

                tx = 0.5 + rr * float(np.cos(th))
                ty = 0.5 + rr * float(np.sin(th))

                ha = "left" if tx >= 0.5 else "right"
                va = "bottom" if ty >= 0.5 else "top"

                if custom_dir:
                    rad = 0.0
                else:
                    rad = (0.1 if (i % 2 == 0) else -0.1)

                arrow_i = dict(base_arrow)
                arrow_i["connectionstyle"] = f"arc3,rad={rad}"

                ann = axins.annotate(
                    lab,
                    xy=(float(px), float(py)),
                    xycoords="data",
                    xytext=(float(tx), float(ty)),
                    textcoords="axes fraction",
                    ha=ha,
                    va=va,
                    arrowprops=arrow_i,
                    **final_text_kwargs
                )

                bbox = ann.get_window_extent(renderer=renderer).padded(bbox_pad_px)

                if (bbox.x0 < parent_bbox.x0) or (bbox.x1 > parent_bbox.x1) or \
                   (bbox.y0 < parent_bbox.y0) or (bbox.y1 > parent_bbox.y1):
                    ann.remove()
                    continue

                if any(bbox.overlaps(b2) for b2 in placed_bboxes):
                    ann.remove()
                    continue

                best_ann = ann
                best_bbox = bbox
                break

            if best_ann is not None:
                break

        if best_ann is None:
            pass
        else:
            placed_bboxes.append(best_bbox)

def plot_ablation_results(results_dir="ablation_analysis_results"):
    """
    Plot classification and reconstruction results from the ablation experiments:
    - Panel A: ΔAUC bar (short labels)
    - Panel B: ΔF1(Active) vs ΔF1(Stable) with auto zoom-in inset
    - Panel C: ΔAUC vs ΔMean R² with auto zoom-in inset
    Outputs: PNG and PDF.
    """
    results_dir = Path(results_dir)
    summary_path = results_dir / "ablation_comparison_summary_with_ranks.csv"
    recon_path = results_dir / "ablation_reconstruction_summary.csv"
    sig_path = results_dir / "ablation_significance_tests.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Classification summary not found: {summary_path}")
    if not recon_path.exists():
        raise FileNotFoundError(f"Reconstruction summary not found: {recon_path}")
    if not sig_path.exists():
        raise FileNotFoundError(f"Significance tests not found: {sig_path}")

    summary_df = pd.read_csv(summary_path)
    recon_df = pd.read_csv(recon_path)
    sig_df = pd.read_csv(sig_path)

    summary_df["Code"] = summary_df["Experiment_ID"]
    recon_df["Code"] = recon_df["Experiment_ID"]
    df = pd.merge(summary_df, recon_df, on="Code", how="left")

    # ----------------- Reference model -----------------
    ref = df[df["Code"] == "A0"].iloc[0]

    df["d_auc"] = df["AUC_Mean"] - ref["AUC_Mean"]
    df["d_acc"] = df["Accuracy_Mean"] - ref["Accuracy_Mean"]


    def welch_auc_difference_ci95(row, reference):
        """
        Calculate the two-sided 95% Welch confidence-interval half-width
        for the difference:

            mean AUC_variant - mean AUC_A0

        This includes uncertainty from both the ablation variant and A0.
        """
        s_variant = float(row["AUC_Std"])
        n_variant = float(row["N_Runs"])
        s_reference = float(reference["AUC_Std"])
        n_reference = float(reference["N_Runs"])

        values = [s_variant, n_variant, s_reference, n_reference]
        if not np.all(np.isfinite(values)):
            return np.nan

        if n_variant <= 1 or n_reference <= 1:
            return np.nan

        # Variance components of the two estimated means
        v_variant = s_variant**2 / n_variant
        v_reference = s_reference**2 / n_reference

        # Standard error of the mean difference
        se_difference = np.sqrt(v_variant + v_reference)

        if se_difference == 0:
            return 0.0

        # Welch–Satterthwaite degrees of freedom
        denominator = (
            v_variant**2 / (n_variant - 1)
            + v_reference**2 / (n_reference - 1)
        )

        if denominator <= 0:
            return np.nan

        welch_df = (v_variant + v_reference) ** 2 / denominator
        t_critical = t.ppf(0.975, welch_df)

        return t_critical * se_difference


    # Half-width of the 95% CI for ΔAUC
    df["auc_ci95"] = df.apply(
        lambda row: (
            0.0
            if row["Code"] == "A0"
            else welch_auc_difference_ci95(row, ref)
        ),
        axis=1,
    )
    df["d_f1_active"] = df["f1_active_Mean"] - ref["f1_active_Mean"]
    df["d_f1_stable"] = df["f1_stable_Mean"] - ref["f1_stable_Mean"]

    df["d_gly_r2"] = df.get("gly_r2_mean", np.nan) - ref.get("gly_r2_mean", np.nan)
    df["d_mass_r2"] = df.get("mass_r2_mean", np.nan) - ref.get("mass_r2_mean", np.nan)
    df["d_rna_r2"] = df.get("rna_r2_mean", np.nan) - ref.get("rna_r2_mean", np.nan)

    def calculate_d_r2_mean(row):
        if row["Code"] == "B1":
            return row["d_gly_r2"]
        if row["Code"] == "B2":
            return row["d_mass_r2"]
        if row["Code"] == "B3":
            return row["d_rna_r2"]
        return np.nanmean([row["d_gly_r2"], row["d_mass_r2"], row["d_rna_r2"]])

    df["d_r2_mean"] = df.apply(calculate_d_r2_mean, axis=1)

    # ----------------- Short names for Panel A (and labels) -----------------
    name_map = {
        "A0": "SLEmodel",
        "A1": "No Cross-Attn",
        "A2": "Pair Recon",
        "A3": "Semi Recon",
        "A4": "No Disentangle",
        "A5": "Static OT",
        "A6": "Random Match",
        "B1": "Gly",
        "B2": "Mass",
        "B3": "RNA",
        "C1_1": "MH=1",
        "C1_2": "MH=2",
        "C2_05": "Drop=0.5",
        "C2_09": "Drop=0.9",
        "C3": "Concat-MLP",
        "C4_32": "LD=32",
        "C4_128": "LD=128",
    }

    cat_map = {
        "A0": "Reference",
        "A1": "Fusion",
        "A2": "Objective",
        "A3": "Objective",
        "A4": "Objective",
        "A5": "Alignment",
        "A6": "Alignment",
        "B1": "Single-Modality",
        "B2": "Single-Modality",
        "B3": "Single-Modality",
        "C1_1": "Fusion",
        "C1_2": "Fusion",
        "C2_05": "Regularisation",
        "C2_09": "Regularisation",
        "C3": "Fusion",
        "C4_32": "Capacity",
        "C4_128": "Capacity",
    }

    df["Name"] = df["Code"].map(name_map)
    df["Category"] = df["Code"].map(cat_map)

    # ----------------- Significance stars (macro_roc_auc, FDR q-values) -----------------
    sig_pivot = sig_df.pivot_table(index="Experiment_ID", columns="Metric", values="Q_Value_FDR").to_dict()

    def get_stars(code, metric):
        try:
            q_val = sig_pivot[metric][code]
        except (KeyError, TypeError):
            return ""
        if q_val < 0.001:
            return "***"
        if q_val < 0.01:
            return "**"
        if q_val < 0.05:
            return "*"
        return ""

    df["auc_sig"] = df.apply(lambda row: get_stars(row["Code"], "macro_roc_auc"), axis=1)

    # ----------------- Style -----------------
    apply_publication_style_helvetica()

    category_colors = {
        "Alignment": "#d55e00",
        "Objective": "#0072b2",
        "Fusion": "#009e73",
        "Capacity": "#e69f00",
        "Regularisation": "#56b4e9",
        "Single-Modality": "#cc79a7",
    }

    # ----------------- Layout -----------------
    fig = plt.figure(figsize=(7.2, 6.0))
    mosaic = [["A", "A"], ["B", "C"]]
    ax_dict = fig.subplot_mosaic(mosaic, gridspec_kw={"height_ratios": [1.05, 0.9]})

    # ====================================================
    # Macro-AUC change relative to SLEmodel
    # ====================================================
    ax = ax_dict["A"]
    plot_data_a = df[df["Code"] != "A0"].sort_values("d_auc", ascending=False)
    colors_a = plot_data_a["Category"].map(category_colors)

    ax.bar(
        plot_data_a["Name"],
        plot_data_a["d_auc"],
        color=colors_a,
        width=0.72,
        zorder=3,
    )
    ax.errorbar(
        plot_data_a["Name"],
        plot_data_a["d_auc"],
        yerr=plot_data_a["auc_ci95"],
        fmt="none",
        ecolor="black",
        capsize=2,
        elinewidth=0.6,
        capthick=0.6,
        zorder=4,
    )

    # give extra room so significance markers never hit axes
    y_min = float(np.nanmin(plot_data_a["d_auc"] - plot_data_a["auc_ci95"]))
    y_max = float(np.nanmax(plot_data_a["d_auc"] + plot_data_a["auc_ci95"]))
    ax.set_ylim(bottom=y_min - 0.035, top=y_max + 0.035)

    y_low, y_high = ax.get_ylim()
    pad = 0.02 * (y_high - y_low)  # data-space padding

    for _, row in plot_data_a.iterrows():
        sig = row["auc_sig"]
        if not sig:
            continue

        ci = float(row["auc_ci95"])
        if not np.isfinite(ci):
            continue

        if row["d_auc"] >= 0:
            y_star = row["d_auc"] + ci + pad
            va = "bottom"
        else:
            y_star = row["d_auc"] - ci - pad
            va = "top"
            if y_star < (y_low + 1.5 * pad):
                y_star = row["d_auc"] + ci + pad
                va = "bottom"

        ax.annotate(
            sig,
            xy=(row["Name"], y_star),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            zorder=5,
        )

    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_ylabel("ΔAUC vs. SLEmodel")
    ax.grid(axis="y", linestyle=":", linewidth=0.45, color="0.85", zorder=1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ====================================================
    # Panel B: ΔF1(Active) vs ΔF1(Stable) + inset zoom
    # ====================================================
    ax = ax_dict["B"]
    plot_data_b = df[df["Code"] != "A0"].dropna(subset=["d_f1_active", "d_f1_stable"]).copy()
    colors_b = plot_data_b["Category"].map(category_colors).values

    ax.scatter(
        plot_data_b["d_f1_active"],
        plot_data_b["d_f1_stable"],
        c=colors_b,
        s=28,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=2)
    ax.axvline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=2)
    ax.set_xlabel("ΔF1 (Active)")
    ax.set_ylabel("ΔF1 (Stable)")
    ax.grid(True, linestyle=":", linewidth=0.45, color="0.85", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # inset zoom near origin (auto)
    x_b = plot_data_b["d_f1_active"].to_numpy()
    y_b = plot_data_b["d_f1_stable"].to_numpy()
    xlim_b, ylim_b = auto_zoom_limits_near_origin(x_b, y_b, keep_frac=0.75, pad_frac=0.18)
    axins_b = add_zoom_inset(
        ax, x_b, y_b, colors_b, xlim_b, ylim_b,
        bbox_axes=(0.5 - 0.38/2, 0.5 - 0.38/2, 0.38, 0.38),
        background_alpha=0.35,
        show_main_region=True,
        connect=True,
    )

    inset_points_b = _select_inset_points_panel_b(
        plot_data_b,
        "d_f1_active",
        "d_f1_stable",
        xlim_b,
        ylim_b,
        n_total_inset=6,
    )


    exclude_b = set(inset_points_b["Code"].tolist()) if not inset_points_b.empty else set()
    outliers_b = _select_main_outliers(
        plot_data_b,
        "d_f1_active",
        "d_f1_stable",
        xlim_b,
        ylim_b,
        n=3,
        exclude_codes=exclude_b,
    )


    # ====================================================
    # Panel C: ΔAUC vs ΔMean R² + inset zoom
    # ====================================================
    ax = ax_dict["C"]
    plot_data_c = df[df["Code"] != "A0"].dropna(subset=["d_auc", "d_r2_mean"]).copy()
    colors_c = plot_data_c["Category"].map(category_colors).values

    ax.scatter(
        plot_data_c["d_auc"],
        plot_data_c["d_r2_mean"],
        c=colors_c,
        s=28,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=2)
    ax.axvline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=2)
    ax.set_xlabel("ΔAUC (Classification)")
    ax.set_ylabel("ΔMean R² (Reconstruction)")
    ax.grid(True, linestyle=":", linewidth=0.45, color="0.85", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # inset zoom near origin (auto)
    x_c = plot_data_c["d_auc"].to_numpy()
    y_c = plot_data_c["d_r2_mean"].to_numpy()
    xlim_c, ylim_c = auto_zoom_limits_near_origin(x_c, y_c, keep_frac=0.55, pad_frac=0.22)
    axins_c = add_zoom_inset(
        ax, x_c, y_c, colors_c, xlim_c, ylim_c,
        bbox_axes=(0.5 - 0.38/2, 0.5 - 0.38/2, 0.38, 0.38),
        background_alpha=0.35,
        show_main_region=True,
        connect=True,
    )

    inset_points_c = _select_inset_points_panel_c(
        plot_data_c,
        "d_auc",
        "d_r2_mean",
        xlim_c,
        ylim_c,
        n_total_inset=6,
    )


    exclude_c = set(inset_points_c["Code"].tolist()) if not inset_points_c.empty else set()
    outliers_c = _select_main_outliers(
        plot_data_c,
        "d_auc",
        "d_r2_mean",
        xlim_c,
        ylim_c,
        n=2,
        exclude_codes=exclude_c,
    )


    # ----------------- Legend -----------------
    legend_patches = [
        mpatches.Patch(color=color, label=cat)
        for cat, color in category_colors.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(category_colors),
        frameon=False,
    )

    fig.suptitle("Ablation study of SLEmodel", fontsize=10, fontweight="bold", y=0.965)
    plt.tight_layout(rect=[0, 0.07, 1, 0.955])

    # ----------------- Callouts (MUST be after tight_layout) -----------------
    fig.canvas.draw()

    # --- Panel B: Inset Annotations ---
    b_inset_dirs = {"Gly": "bottom"}

    annotate_callouts_in_inset(
        axins_b,
        inset_points_b,
        "d_f1_active",
        "d_f1_stable",
        "Name",
        parent_ax=ax_dict["B"],
        xlim=xlim_b,
        ylim=ylim_b,
        r=0.65,
        axes_margin=0.05,
        text_kwargs=dict(fontsize=7, color="0.15", zorder=20),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6, shrinkA=0, shrinkB=0, alpha=0.95),
        custom_directions=b_inset_dirs,
    )

    # --- Panel B: Main Outliers ---
    b_main_offsets = {
        "Random Match": (0, -25, "center", "top")
    }

    annotate_callouts_on_main(
        ax_dict["B"],
        outliers_b,
        "d_f1_active",
        "d_f1_stable",
        "Name",
        offset_pts=10,
        text_kwargs=dict(fontsize=7, color="0.15", zorder=8),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6, alpha=0.95, shrinkA=0, shrinkB=0),
        avoid_axes=[axins_b],
        avoid_points_df=plot_data_b,
        avoid_x_col="d_f1_active",
        avoid_y_col="d_f1_stable",
        custom_offsets=b_main_offsets,
    )

    # --- Panel C: Inset Annotations ---
    c_inset_dirs = {"LD=128": "left"}

    annotate_callouts_in_inset(
        axins_c,
        inset_points_c,
        "d_auc",
        "d_r2_mean",
        "Name",
        parent_ax=ax_dict["C"],
        xlim=xlim_c,
        ylim=ylim_c,
        r=0.65,
        axes_margin=0.05,
        text_kwargs=dict(fontsize=7, color="0.15", zorder=20),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6, shrinkA=0, shrinkB=0, alpha=0.95),
        custom_directions=c_inset_dirs,
    )

    # --- Panel C: Main Outliers ---
    c_main_offsets = {
        "Random Match": (5, -20, "left", "top")
    }

    annotate_callouts_on_main(
        ax_dict["C"],
        outliers_c,
        "d_auc",
        "d_r2_mean",
        "Name",
        offset_pts=10,
        text_kwargs=dict(fontsize=7, color="0.15", zorder=8),
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6, alpha=0.95, shrinkA=0, shrinkB=0),
        avoid_axes=[axins_c],
        avoid_points_df=plot_data_c,
        avoid_x_col="d_auc",
        avoid_y_col="d_r2_mean",
        custom_offsets=c_main_offsets,
    )

    # Panel labels aligned to ylabel left edge
    add_panel_label_aligned_to_ylabel(fig, ax_dict["A"], "a")
    add_panel_label_aligned_to_ylabel(fig, ax_dict["B"], "b")
    add_panel_label_aligned_to_ylabel(fig, ax_dict["C"], "c")

    # ----------------- Save -----------------
    png_path = results_dir / "ablation_summary.png"
    pdf_path = results_dir / "ablation_summary.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="ablation_analysis_results",
        help="Directory containing ablation summary CSVs",
    )
    args = parser.parse_args()
    plot_ablation_results(args.results_dir)
