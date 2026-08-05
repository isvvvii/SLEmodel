# gly_traits_analysis.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from slemodel import config as cfg
from .enrichment_plot_utils import set_publication_style

# SHAP/MoR helpers are imported lazily inside the functions that use
# pseudo-paired RNA data. Glycan-trait calculations do not require SHAP.

from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw, ImageFont


def _safe_sum(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)

def _safe_ratio(num: pd.Series, den: pd.Series, scale: float = 100.0) -> pd.Series:
    den = den.replace(0, np.nan)
    out = (num / den) * scale
    return out.fillna(0.0)

def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 202510) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    mean = values.mean()
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(mean), float(lo), float(hi)

def _map_labels(y_str: np.ndarray) -> np.ndarray:
    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    return pd.Series(y_str).map(label_map).values

def _load_ensg_map(csv_path: Path = Path("ref/ensembl_to_hgnc_GRCh38.csv")) -> Dict[str,str]:
    if not csv_path.exists(): return {}
    mp = {}
    df = pd.read_csv(csv_path)
    id_col = next((c for c in df.columns if "ensembl" in c.lower()), None)
    sym_col = next((c for c in df.columns if "hgnc" in c.lower() or "symbol" in c.lower()), None)
    if id_col and sym_col:
        for gid, sym in df[[id_col, sym_col]].itertuples(index=False):
            gid = str(gid).split(".")[0]
            mp[str(gid)] = str(sym)
    return mp

def get_pseudo_rna_matrix_for_gly_val(seed: Optional[int]=None, fold_idx: Optional[int]=None) -> pd.DataFrame:
    """
    Return a data frame containing the sample ID and RNA features.
    """
    from .shap_analysis import pick_model_of_record, get_val_loader_for_fold

    if seed is None or fold_idx is None:
        seed, fold_idx, _ = pick_model_of_record()
    device = cfg.DEVICE
    val_loader, meta = get_val_loader_for_fold(seed, fold_idx, device)
    genes = meta.get("rna_featnames", [])
    ensg2sym = _load_ensg_map()
    out_cols = []
    for g in genes:
        if g.startswith("ENSG"):
            sym = ensg2sym.get(g.split(".")[0], g)
            out_cols.append(sym.upper())
        else:
            out_cols.append(str(g).upper())
    recs = []
    for batch in val_loader:
        if "rna_x" not in batch or "id" not in batch: continue
        X = batch["rna_x"].numpy()
        ids = list(batch["id"])
        for row, sid in zip(X, ids):
            recs.append(pd.Series([sid] + list(row), index=["id"] + out_cols))
    if not recs:
        return pd.DataFrame(columns=["id"])
    df = pd.DataFrame(recs)
    df = df.groupby("id", as_index=False).mean(numeric_only=True)
    return df



TRAIT_FORMULAE = {
    "S_total_calc": ["GP16a","GP16b","GP17","GP18a","GP18b","GP19","GP21","GP22","GP23","GP24"],
    "S1_total_calc": ["GP16a","GP16b","GP17","GP18a","GP18b","GP19"],
    "FG1S1_over_FG1_plus_FG1S1_calc_num": ["GP16a","GP16b"],
    "FG1S1_over_FG1_plus_FG1S1_calc_den": ["GP8b","GP9","GP16a","GP16b"],
    "FG2S1_over_FG2_plus_FG2S1_plus_FG2S2_calc_num": ["GP18b"],
    "FG2S1_over_FG2_plus_FG2S1_plus_FG2S2_calc_den": ["GP14","GP18b","GP23"],
}

def compute_gly_traits(gly_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gly_csv)
    id_col, group_col = df.columns[0], df.columns[1]

    keep_cols = [id_col, group_col]
    preferred = ["STotal","S1Total","S2Total","G0n","G1n","G2n","bisecting",
                 "FnTotal","GI","G2FS2","Sia","Gal","Sia/Gal",
                 "FG0vsG0","FG1vsG1","FG2vsG2","FBn","FBG0vsG0","FBG1vsG1","FBG2vsG2",
                 "FBS1vsFS1plusFBS1","FBS2vsFS2plusFBS2"]
    for c in preferred:
        if c in df.columns: keep_cols.append(c)

    def _safe_sum(cols):
        exist = [c for c in cols if c in df.columns]
        return df[exist].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) if exist else pd.Series(0.0, index=df.index)
    if "STotal" not in df.columns:
        stotal = _safe_sum(["GP16a","GP16b","GP17","GP18a","GP18b","GP19","GP21","GP22","GP23","GP24"])
        df["S_total_calc"] = stotal; keep_cols.append("S_total_calc")
    if "S1Total" not in df.columns:
        s1total = _safe_sum(["GP16a","GP16b","GP17","GP18a","GP18b","GP19"])
        df["S1_total_calc"] = s1total; keep_cols.append("S1_total_calc")

    def _safe_ratio(num, den, scale=100.0):
        den = den.replace(0, np.nan)
        return ((num/den)*scale).fillna(0.0)
    if ("FG1S1_ratio_pct_calc" not in df.columns) and not set(["GP16a","GP16b","GP8b","GP9"]).isdisjoint(df.columns):
        num = _safe_sum(["GP16a","GP16b"])
        den = _safe_sum(["GP8b","GP9","GP16a","GP16b"])
        df["FG1S1_ratio_pct_calc"] = _safe_ratio(num, den); keep_cols.append("FG1S1_ratio_pct_calc")
    if ("FG2S1_ratio_pct_calc" not in df.columns) and not set(["GP18b","GP14","GP23"]).isdisjoint(df.columns):
        num = _safe_sum(["GP18b"]); den = _safe_sum(["GP14","GP18b","GP23"])
        df["FG2S1_ratio_pct_calc"] = _safe_ratio(num, den); keep_cols.append("FG2S1_ratio_pct_calc")

    return df[keep_cols].copy()


def plot_gly_trait_forest(traits_df: pd.DataFrame,
                          trait_cols: List[str],
                          save_path: Path,
                          ref_group: str = "Control",
                          comp_group: str = "Active") -> None:
    set_publication_style()
    group_col = traits_df.columns[1]
    sub = traits_df[traits_df[group_col].isin([ref_group, comp_group])].copy()
    if sub.empty:
        warnings.warn("No data for requested groups in forest plot.")
        return

    rows = []
    pvals = []
    for t in trait_cols:
        if t not in sub.columns:
            continue
        a = pd.to_numeric(sub.loc[sub[group_col]==comp_group, t], errors='coerce').dropna().values
        c = pd.to_numeric(sub.loc[sub[group_col]==ref_group, t], errors='coerce').dropna().values
        if a.size==0 or c.size==0:
            continue
        diff = a.mean() - c.mean()
        rng = np.random.default_rng(202510)
        boots = []
        for _ in range(3000):
            aa = rng.choice(a, size=a.size, replace=True)
            cc = rng.choice(c, size=c.size, replace=True)
            boots.append(aa.mean() - cc.mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        try:
            p = mannwhitneyu(a, c, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        rows.append({"trait": t, "diff": diff, "lo": lo, "hi": hi, "p": p})
        pvals.append(p)

    if not rows:
        warnings.warn("No traits available for forest plot.")
        return

    # BH-FDR
    P = np.array([r["p"] for r in rows], dtype=float)
    mask = np.isfinite(P)
    m = mask.sum()
    q = np.full_like(P, np.nan)
    if m > 0:
        order = np.argsort(np.where(mask, P, np.inf))
        ranks = np.arange(1, len(P)+1)
        ranked_p = P[order]
        ranked_q = ranked_p * m / ranks
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        q[order] = ranked_q
    for i, qi in enumerate(q):
        rows[i]["q"] = float(qi) if np.isfinite(qi) else np.nan

    D = pd.DataFrame(rows).sort_values("diff")
    labels = []
    for row in D.itertuples():
        star = ""
        if np.isfinite(row.q):
            if row.q < 0.05: star = " *"
            elif row.q < 0.10: star = " ·"
        labels.append(f"{row.trait}{star}")

    fig, ax = plt.subplots(figsize=(7.9, max(4.8, 0.36*len(D)+1.3)))
    y = np.arange(len(D))
    ax.hlines(y, D["lo"], D["hi"], color="#6b6b6b", lw=2)
    ax.plot(D["diff"], y, "o", color="#2f6f80")
    for yi, row in zip(y, D.itertuples()):
        ax.text(row.hi + 0.02*np.nanmax(np.abs(D[["lo","hi"]].values)), yi,
                f"{row.diff:.2f} [{row.lo:.2f},{row.hi:.2f}]  q={row.q:.3f}" if np.isfinite(row.q) else
                f"{row.diff:.2f} [{row.lo:.2f},{row.hi:.2f}]",
                va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel(f"Mean difference ({comp_group} - {ref_group})")
    ax.set_title("Glycan traits: group differences (BH-FDR)", pad=8, fontweight="bold", fontsize=16)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _load_gmt(gmt_path: Path) -> Dict[str, List[str]]:
    """Parse GMT rows formatted as term, description and genes."""
    mp = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            mp[term] = genes
    return mp

DEFAULT_ISG5 = ["MX1","IRF7","OAS1","IFIT1","IFI44"]

def compute_ifn_scores_for_gly_val_by_ot(
    hallmark_gmt: Optional[Path] = None,
    hallmark_term: str = "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    seed: Optional[int] = None,
    fold_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate interferon scores from the standardized pseudo-RNA profiles
    assigned to validation glycomics samples by propensity-score-guided OT.
    Ensembl identifiers are mapped with ``ref/ensembl_to_hgnc_GRCh38.csv``.
    """
    from .shap_analysis import pick_model_of_record, get_val_loader_for_fold

    if seed is None or fold_idx is None:
        seed, fold_idx, _ = pick_model_of_record()

    device = cfg.DEVICE
    val_loader, meta = get_val_loader_for_fold(seed, fold_idx, device)

    genes = []
    if hallmark_gmt and Path(hallmark_gmt).exists():
        with open(hallmark_gmt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0].strip() == hallmark_term:
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    break
    if not genes:
        genes = [g.upper() for g in ["MX1","IRF7","OAS1","IFIT1","IFI44"]]

    rna_feats = meta.get("rna_featnames", [])
    ensg2sym = _load_ensg_map(Path("ref/ensembl_to_hgnc_GRCh38.csv"))
    col_symbols = []
    for g in rna_feats:
        if isinstance(g, str) and g.upper().startswith("ENSG"):
            sym = ensg2sym.get(g.split(".")[0], None)
            col_symbols.append(sym.upper() if sym else g.upper())
        else:
            col_symbols.append(str(g).upper())

    gene_set = set(genes)
    idx_use = [i for i, sym in enumerate(col_symbols) if sym in gene_set]

    if not idx_use:
        warnings.warn("No IFN genes matched after ENSG→HGNC mapping. IFN scatter will be skipped.")
        return pd.DataFrame(columns=["id","ifn_score"])

    recs = []
    for batch in val_loader:
        if "rna_x" not in batch or "id" not in batch:
            continue
        Xr = batch["rna_x"].numpy()
        ids = list(batch["id"])
        sc = Xr[:, idx_use].mean(axis=1)
        for sid, v in zip(ids, sc):
            recs.append({"id": sid, "ifn_score": float(v)})
    return pd.DataFrame(recs)


def plot_ifn_scatter(gly_traits_df: pd.DataFrame,
                     ifn_df: pd.DataFrame,
                     trait: str,
                     save_path: Path) -> None:
    set_publication_style()
    id_col = gly_traits_df.columns[0]
    group_col = gly_traits_df.columns[1]
    df = gly_traits_df[[id_col, group_col, trait]].merge(ifn_df, left_on=id_col, right_on="id", how="inner")
    if df.empty:
        warnings.warn("No overlap between gly traits and IFN scores; skip scatter.")
        return

    df[trait] = pd.to_numeric(df[trait], errors="coerce")
    df["ifn_score"] = pd.to_numeric(df["ifn_score"], errors="coerce")
    df = df.dropna(subset=[trait, "ifn_score"])
    if df.empty:
        warnings.warn("Trait or IFN score has no valid numeric values after cleaning.")
        return

    # Spearman
    rho = df[[trait, "ifn_score"]].corr(method="spearman").iloc[0, 1]

    plt.figure(figsize=(6.8, 5.6))
    palette = {"Active": "#E68B81", "Stable": "#8aab82", "Control": "#7DA6C6"}

    sns.regplot(
        data=df, x=trait, y="ifn_score",
        scatter=False,
        line_kws={"color": "#555", "lw": 2.0}
    )

    sns.scatterplot(
        data=df, x=trait, y="ifn_score",
        hue=group_col, palette=palette, s=55,
        edgecolor="k"
    )

    plt.title("Glycan trait versus IFN score (pseudo-paired)", fontsize=16, fontweight="bold", pad=8)
    plt.xlabel(trait)
    plt.ylabel("IFN-I score (ss-average)")
    plt.legend(frameon=True)
    plt.text(0.02, 0.98, f"Spearman \u03C1 = {rho:.3f}\nNote: pseudo‑paired via OT",
             transform=plt.gca().transAxes,
             ha="left", va="top", fontsize=10,
             bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.25"))
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def plot_ifn_scatters_grid(gly_traits_df: pd.DataFrame,
                           ifn_alpha_df: pd.DataFrame,
                           ifn_gamma_df: pd.DataFrame,
                           trait: str,
                           save_path: Path) -> None:
    set_publication_style()
    id_col = gly_traits_df.columns[0]
    group_col = gly_traits_df.columns[1]

    def _prep(df_ifn):
        df = gly_traits_df[[id_col, group_col, trait]].merge(df_ifn, left_on=id_col, right_on="id", how="inner")
        if df.empty:
            return df, np.nan, np.nan
        df[trait] = pd.to_numeric(df[trait], errors="coerce")
        df["ifn_score"] = pd.to_numeric(df["ifn_score"], errors="coerce")
        df = df.dropna(subset=[trait, "ifn_score"])
        if df.empty:
            return df, np.nan, np.nan
        r, p = spearmanr(df[trait], df["ifn_score"])
        return df, r, p

    df_a, r_a, p_a = _prep(ifn_alpha_df)
    df_g, r_g, p_g = _prep(ifn_gamma_df)

    # BH-FDR across two tests
    P = np.array([p_a, p_g], dtype=float)
    mask = np.isfinite(P)
    q = np.full_like(P, np.nan, dtype=float)
    m = mask.sum()
    if m > 0:
        order = np.argsort(np.where(mask, P, np.inf))
        ranks = np.arange(1, len(P)+1)
        ranked_p = P[order]
        ranked_q = ranked_p * m / ranks
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        q[order] = ranked_q
    qa, qg = q[0], q[1]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True)
    palette = {"Active":"#E68B81","Stable":"#8aab82","Control":"#7DA6C6"}

    def _panel(ax, df, r, q, title):
        if df is None or df.empty:
            ax.axis("off")
            ax.set_title(f"{title}\n(no overlap)", fontsize=12)
            return
        sns.regplot(data=df, x=trait, y="ifn_score", scatter=False, line_kws={"color":"#555","lw":2.0}, ax=ax)
        sns.scatterplot(data=df, x=trait, y="ifn_score", hue=group_col, palette=palette, s=55, edgecolor="black", ax=ax)
        star = ""
        if np.isfinite(q):
            if q < 0.05: star = " *"
            elif q < 0.10: star = " ·"
        ax.set_title(f"{title}{star}", fontsize=14, fontweight="bold")
        ax.set_xlabel(trait); ax.set_ylabel("IFN score")
        if np.isfinite(r):
            txt = f"Spearman r = {r:.3f}" + (f"\nBH-FDR q = {q:.3f}" if np.isfinite(q) else "")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.25"))
        ax.legend(frameon=True)

    _panel(axes[0], df_a, r_a, qa, "IFN-alpha")
    _panel(axes[1], df_g, r_g, qg, "IFN-gamma")

    fig.suptitle("Glycan traits versus IFN scores (pseudo-paired; BH-FDR)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_trait_roc_multi(gly_traits_df: pd.DataFrame,
                         trait_cols: List[str],
                         tasks: List[Tuple[str,str,str]],
                         save_path: Path) -> None:
    set_publication_style()
    group_col = gly_traits_df.columns[1]
    plt.figure(figsize=(7.6, 6.4))
    colors = ["#C96963","#4C8C8A","#6A5ACD","#2F6F80"]

    for i,(pos,neg,name) in enumerate(tasks):
        sub = gly_traits_df[gly_traits_df[group_col].isin([pos,neg])].copy()
        if sub.empty:
            continue
        X = sub[trait_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        y = (sub[group_col]==pos).astype(int).values
        if X.shape[0] < 4 or len(np.unique(y))<2:
            continue

        clf = LogisticRegression(solver="lbfgs", max_iter=400).fit(X,y)
        prob = clf.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, prob)
        base_auc = auc(fpr, tpr)

        rng = np.random.default_rng(202510)
        aucs = []
        n = len(y)
        for _ in range(2000):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[idx], prob[idx]))
        if len(aucs) >= 50:
            lo, hi = np.quantile(aucs, [0.025, 0.975])
        else:
            lo, hi = np.nan, np.nan

        plt.plot(fpr, tpr, lw=2.0, color=colors[i%len(colors)],
                 label=f"{name} (AUC={base_auc:.3f}"
                       f"{f' [{lo:.3f},{hi:.3f}]' if np.isfinite(lo) else ''})")

    plt.plot([0,1],[0,1],'--', color='grey', lw=1.2)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Glycan traits: multi-task ROC (95% CI)", fontsize=16, fontweight="bold", pad=8)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight'); plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()


def compose_panel_c_2up(img_top_or_left: Path,
                        img_bottom_or_right: Path,
                        out_path: Path,
                        orientation: str = "vertical",
                        gap: int = 32,
                        bg_color: str = "white",
                        add_labels: bool = True,
                        labels: Tuple[str, str] = ("C2", "C3")) -> None:
    """
    Combine two images vertically or horizontally.
    """
    def _open_rgb(p: Path) -> Image.Image:
        im = Image.open(p)
        return im.convert("RGB")

    if not Path(img_top_or_left).exists() or not Path(img_bottom_or_right).exists():
        warnings.warn("One of the input images does not exist; skip composing C panel.")
        return

    im1 = _open_rgb(img_top_or_left)
    im2 = _open_rgb(img_bottom_or_right)

    if orientation.lower().startswith("v"):
        target_w = max(im1.width, im2.width)
        def _resize_to_w(img: Image.Image, w: int) -> Image.Image:
            if img.width == w:
                return img
            h = int(round(img.height * (w / img.width)))
            return img.resize((w, h), Image.LANCZOS)
        im1r = _resize_to_w(im1, target_w)
        im2r = _resize_to_w(im2, target_w)
        W = target_w
        H = im1r.height + gap + im2r.height
        canvas = Image.new("RGB", (W, H), color=bg_color)
        canvas.paste(im1r, (0, 0))
        canvas.paste(im2r, (0, im1r.height + gap // 2))
        if add_labels:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
            draw.text((12, 8), labels[0], fill=(0, 0, 0), font=font)
            draw.text((12, im1r.height + gap // 2 + 8), labels[1], fill=(0, 0, 0), font=font)
    else:
        target_h = max(im1.height, im2.height)
        def _resize_to_h(img: Image.Image, h: int) -> Image.Image:
            if img.height == h:
                return img
            w = int(round(img.width * (h / img.height)))
            return img.resize((w, h), Image.LANCZOS)
        im1r = _resize_to_h(im1, target_h)
        im2r = _resize_to_h(im2, target_h)
        W = im1r.width + gap + im2r.width
        H = target_h
        canvas = Image.new("RGB", (W, H), color=bg_color)
        canvas.paste(im1r, (0, 0))
        canvas.paste(im2r, (im1r.width + gap, 0))
        if add_labels:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
            draw.text((12, 8), labels[0], fill=(0, 0, 0), font=font)
            draw.text((im1r.width + gap + 12, 8), labels[1], fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    try:
        canvas.save(out_path.with_suffix(".pdf"))
    except Exception:
        pass

# ---------------- 6）supplication --------------------------

def _curated_sets_for_axes() -> Dict[str, List[str]]:
    return {
        "SYK_AXIS": ["SYK","BLNK","LCP2","PLCG2","BTK","LYN","HCK","PIK3CD","CARD11","VAV1","LAT2"],
        "TLR_PDC_AXIS": ["TLR7","TLR9","TLR8","MYD88","IRAK4","UNC93B1","IRF7","TBK1","IKBKE"],
        "FCGR_AXIS": ["FCGR2A","FCGR2B","FCGR3A","FCGR3B","FCGR1A","FCGR1B","FCGR1C"]
    }

def _glycosylation_enzymes() -> List[str]:
    return [
        # sialylation
        "ST6GAL1","ST6GAL2","ST3GAL1","ST3GAL2","ST3GAL3","ST3GAL4","ST3GAL5","ST3GAL6",
        # galactosylation
        "B4GALT1",
        # core fucosylation
        "FUT8",
        # bisecting/branching
        "MGAT3","MGAT5"
    ]

def _score_by_mean(df_expr: pd.DataFrame, genes: List[str], prefix: str) -> pd.Series:
    cols = [c for c in genes if c in df_expr.columns]
    if not cols: return pd.Series(np.nan, index=df_expr.index, name=prefix)
    return df_expr[cols].mean(axis=1).rename(prefix)

def compute_and_plot_mechanism_heatmap(traits_df: pd.DataFrame,
                                       hallmark_gmt: Optional[Path],
                                       out_path: Path,
                                       trait_cols: Optional[List[str]] = None,
                                       min_genes_per_set: int = 2) -> None:
    """Plot Spearman correlations between glycan traits and RNA modules."""
    supp_dir = out_path.parent
    supp_dir.mkdir(parents=True, exist_ok=True)

    rna_df = get_pseudo_rna_matrix_for_gly_val()
    if rna_df.empty or "id" not in rna_df.columns:
        warnings.warn("Pseudo RNA not available; skip mechanism heatmap.")
        return

    # Traits
    id_col, group_col = traits_df.columns[0], traits_df.columns[1]
    if trait_cols is None:
        trait_cols = [c for c in ["STotal","S1Total","bisecting","G0n","G1n","G2n","GI","Sia","Gal","Sia/Gal",
                                  "S_total_calc","S1_total_calc","FG1S1_ratio_pct_calc","FG2S1_ratio_pct_calc"]
                      if c in traits_df.columns]
    df = traits_df[[id_col, group_col] + trait_cols].merge(rna_df, left_on=id_col, right_on="id", how="inner")
    if df.empty:
        warnings.warn("No overlap between gly traits and pseudo RNA; skip heatmap.")
        return
    df = df.drop(columns=["id"])

    expr_cols = [c for c in df.columns if c not in [id_col, group_col] + trait_cols]
    expr = df[expr_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    trait_mat = df[trait_cols].apply(pd.to_numeric, errors='coerce')

    axes_sets = _curated_sets_for_axes()
    match_report = {"axes": {}, "enzymes": {}, "ifn": {}}

    def _score_gene_set(name: str, genes: List[str]) -> Optional[pd.Series]:
        genes_u = [g for g in genes if g in expr.columns]
        match_report["axes"][name] = {"requested": len(genes), "matched": len(genes_u), "genes": genes_u}
        if len(genes_u) < min_genes_per_set:
            return None
        return expr[genes_u].mean(axis=1).rename(name)

    axis_scores = []
    for name, genes in axes_sets.items():
        s = _score_gene_set(name, genes)
        if s is not None:
            axis_scores.append(s)

    ifn_score = None
    if hallmark_gmt and Path(hallmark_gmt).exists():
        genes = []
        with open(hallmark_gmt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[0].strip() == "HALLMARK_INTERFERON_ALPHA_RESPONSE":
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    break
        matched = [g for g in genes if g in expr.columns]
        match_report["ifn"]["HALLMARK_INTERFERON_ALPHA_RESPONSE"] = {"requested": len(genes), "matched": len(matched)}
        if matched:
            ifn_score = expr[matched].mean(axis=1).rename("IFN_ALPHA_HALLMARK")
    if ifn_score is None:
        isg5 = [g for g in ["MX1","IRF7","OAS1","IFIT1","IFI44"] if g in expr.columns]
        match_report["ifn"]["ISG5"] = {"requested": 5, "matched": len(isg5)}
        if isg5:
            ifn_score = expr[isg5].mean(axis=1).rename("IFN_ISG5")

    enz_list = [g for g in _glycosylation_enzymes() if g in expr.columns]
    match_report["enzymes"]["GLYCO_ENZ"] = {"requested": len(_glycosylation_enzymes()), "matched": len(enz_list), "genes": enz_list}
    enz_df = expr[enz_list].copy()
    enz_df.columns = [f"ENZ_{c}" for c in enz_df.columns]

    score_df = pd.concat([s for s in axis_scores if s is not None] + ([ifn_score] if ifn_score is not None else []), axis=1)
    targets = pd.concat([enz_df, score_df], axis=1)
    targets = targets.loc[:, targets.notna().any(axis=0)]
    if targets.shape[1] == 0:
        warnings.warn("No valid enzyme/axis/IFN columns to plot.")
        return

    from scipy.stats import spearmanr
    cols_tgt = targets.columns.tolist()
    rho = np.zeros((len(trait_cols), len(cols_tgt)), dtype=float) * np.nan
    pval = np.ones_like(rho, dtype=float)
    for i, t in enumerate(trait_cols):
        x = trait_mat[t].values
        for j, c in enumerate(cols_tgt):
            y = targets[c].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 4:
                r, p = spearmanr(x[mask], y[mask])
                rho[i, j] = r
                pval[i, j] = p if np.isfinite(p) else 1.0

    m = np.isfinite(pval).sum()
    q = np.full_like(pval, np.nan, dtype=float)
    if m > 0:
        flat_p = pval.flatten()
        idx = np.where(np.isfinite(flat_p))[0]
        order = idx[np.argsort(flat_p[idx])]
        ranked = flat_p[order] * m / (np.arange(len(order)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        flat_q = q.flatten()
        flat_q[order] = ranked
        q = flat_q.reshape(pval.shape)

    set_publication_style()
    fig_h = max(6.5, 0.4*len(trait_cols)+2.0)
    fig_w = max(8.0, 0.35*len(cols_tgt)+3.0)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(rho, vmin=-0.6, vmax=0.6, cmap="RdBu_r",
                     xticklabels=cols_tgt, yticklabels=trait_cols,
                     cbar_kws={"label": "Spearman r"})
    for i in range(len(trait_cols)):
        for j in range(len(cols_tgt)):
            if not np.isfinite(q[i, j]):
                continue
            if q[i, j] < 0.05:
                ax.text(j+0.5, i+0.5, "*", ha="center", va="center", color="k", fontsize=12)
            elif q[i, j] < 0.10:
                ax.text(j+0.5, i+0.5, "·", ha="center", va="center", color="k", fontsize=12)
    plt.title("Glycan traits and enzyme/pathway scores", fontsize=16, fontweight="bold", pad=8)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight'); plt.savefig(out_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    try:
        with open(out_path.with_suffix(".match_summary.json"), "w", encoding="utf-8") as f:
            json.dump(match_report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Glycan trait analysis and mechanism heatmap.")
    ap.add_argument("--gly_csv", type=str, default=cfg.GLY_PATH)
    ap.add_argument("--hallmark_gmt", type=str, default="ref/msigdb/h.all.v2025.1.Hs.symbols.gmt")
    ap.add_argument("--hallmark_term", type=str, default="HALLMARK_INTERFERON_ALPHA_RESPONSE")
    ap.add_argument("--outdir", type=str, default="clinical_explain_results/enrichment_results/plots")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    detail_dir = outdir / "glycan_trait_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    traits = compute_gly_traits(Path(args.gly_csv))
    traits_path = Path("clinical_explain_results/enrichment_results/gly_traits")
    traits_path.mkdir(parents=True, exist_ok=True)
    traits.to_csv(traits_path / "gly_traits_per_sample.csv", index=False)

    candidate_traits = [t for t in ["STotal","S1Total","bisecting","G0n","G1n","G2n",
                                    "S_total_calc","S1_total_calc","FG1S1_ratio_pct_calc","FG2S1_ratio_pct_calc"] if t in traits.columns]
    forest_path = detail_dir / "glycan_trait_forest.png"
    plot_gly_trait_forest(traits, candidate_traits, forest_path, ref_group="Control", comp_group="Active")

    from pathlib import Path as _P
    ifn_alpha = compute_ifn_scores_for_gly_val_by_ot(
        hallmark_gmt=_P(args.hallmark_gmt) if args.hallmark_gmt else None,
        hallmark_term="HALLMARK_INTERFERON_ALPHA_RESPONSE"
    )
    ifn_gamma = compute_ifn_scores_for_gly_val_by_ot(
        hallmark_gmt=_P(args.hallmark_gmt) if args.hallmark_gmt else None,
        hallmark_term="HALLMARK_INTERFERON_GAMMA_RESPONSE"
    )
    ifn_path = outdir / "glycan_trait_vs_interferon.png"
    trait_for_ifn = "STotal" if "STotal" in traits.columns else ("S_total_calc" if "S_total_calc" in traits.columns else candidate_traits[0])
    plot_ifn_scatters_grid(traits, ifn_alpha, ifn_gamma, trait_for_ifn, ifn_path)

    roc_path = outdir / "glycan_trait_roc.png"
    roc_traits = [t for t in ["STotal","S1Total","S_total_calc","S1_total_calc"] if t in traits.columns]
    if len(roc_traits) == 0 and candidate_traits:
        roc_traits = [candidate_traits[0]]
    roc_tasks = [
        ("Stable","Control","Stable vs Control"),
        ("Active","Control","Active vs Control"),
        ("Active","Stable","Active vs Stable")
    ]
    plot_trait_roc_multi(traits, roc_traits, roc_tasks, roc_path)

    compose_panel_c_2up(
        ifn_path,
        roc_path,
        outdir / "glycan_trait_summary.png",
        orientation="vertical",
        labels=("a", "b")
    )

    mechanism_path = outdir / "glycan_enzyme_pathway_heatmap.png"
    compute_and_plot_mechanism_heatmap(traits, Path(args.hallmark_gmt), mechanism_path)

    print(f"[OK] Glycan trait plots saved under: {outdir}")

if __name__ == "__main__":
    main()
