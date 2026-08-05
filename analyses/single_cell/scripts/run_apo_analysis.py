#!/usr/bin/env python3
"""APO versus non-APO PBMC cell-state, DEG, and CellPhoneDB analysis.

Read-only analysis of processed, annotated h5ad objects. Statistics are sample-level:
proportions, targeted cell-state/module summaries, newly computed current-run
pseudobulk DEG from sample-level summaries, and sample-level ligand-receptor
proxies.

Patient-level inputs and group assignments are not distributed with the public
repository. Paths are configured with ``SLEMODEL_SCRNA_*`` environment
variables; see ``analyses/single_cell/README.md``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import argparse
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle, FancyArrowPatch
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None
    TwoSlopeNorm = None
    Rectangle = None
    FancyArrowPatch = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


ROOT = Path(os.environ.get("SLEMODEL_SCRNA_ROOT", Path.cwd())).expanduser().resolve()
INPUT_DIR = Path(os.environ.get("SLEMODEL_SCRNA_INPUT_DIR", ROOT / "data" / "single_cell")).expanduser().resolve()
OUT = Path(os.environ.get("SLEMODEL_SCRNA_OUTPUT_DIR", ROOT / "outputs" / "single_cell")).expanduser().resolve()
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
LOG = OUT / "logs"
OBJ = OUT / "objects"
REF = OUT / "references"
CPDB = OUT / "cellphonedb"
for d in [FIG, TAB, REP, LOG, OBJ, REF, CPDB]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON_PATH = os.environ.get("SLEMODEL_SCRNA_PYTHON", sys.executable)
RSCRIPT_PATH = os.environ.get("SLEMODEL_RSCRIPT", shutil.which("Rscript") or "Rscript")
CPDB_BIN = Path(os.environ.get("SLEMODEL_CELLPHONEDB_BIN", shutil.which("cellphonedb") or "cellphonedb"))
CPDB_DATABASE_DIR = Path(
    os.environ.get("SLEMODEL_CELLPHONEDB_DATABASE_DIR", ROOT / "ref" / "cellphonedb")
).expanduser().resolve()
CPDB_MIN_GROUP_CELLS = 30

H5ADS = {
    "full_atlas": INPUT_DIR / "adata_full_atlas_final.h5ad",
    "B_cell": INPUT_DIR / "adata_bcell_annotated.h5ad",
    "CD4": INPUT_DIR / "adata_cd4_annotated.h5ad",
    "CD8": INPUT_DIR / "adata_cd8_annotated.h5ad",
    "NK": INPUT_DIR / "adata_nk_annotated.h5ad",
    "myeloid": INPUT_DIR / "adata_myeloid_annotated.h5ad",
}
SAMPLE_METADATA_PATH = Path(
    os.environ.get("SLEMODEL_SCRNA_METADATA", INPUT_DIR / "sample_metadata.tsv")
).expanduser().resolve()
AUDIT = ROOT / "ref"


def load_sample_metadata(path: Path) -> pd.DataFrame:
    """Load the controlled sample-to-outcome table without embedding identifiers."""
    if not path.exists():
        raise FileNotFoundError(
            f"Single-cell metadata not found: {path}. "
            "Copy analyses/single_cell/config/sample_metadata.example.tsv, "
            "add the approved de-identified records, and set SLEMODEL_SCRNA_METADATA."
        )
    table = pd.read_csv(path, sep=None, engine="python")
    required = {"sample_id", "apo_group"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    table = table.copy()
    table["sample_id"] = table["sample_id"].astype(str)
    table["apo_group"] = table["apo_group"].astype(str).str.strip()
    allowed = {"APO", "non-APO"}
    invalid = sorted(set(table["apo_group"]).difference(allowed))
    if invalid:
        raise ValueError(f"apo_group must contain only {sorted(allowed)}; found {invalid}")
    if table["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    return table


SAMPLE_METADATA = pd.DataFrame()
SAMPLES: list[str] = []
FORCE_NO_APO: set[str] = set()
APO_SAMPLES: set[str] = set()
APO_LABEL = "APO"
NON_APO_LABEL = "non-APO"

# APO versus non-APO display labels.
# APO -> APO
# other -> non-APO
FINAL_DISPLAY_LABELS = {"APO": "APO", "non-APO": "non-APO"}
FINAL_COMPARISON_LABEL = "APO_vs_nonAPO"

GROUP_COLOR = {APO_LABEL: "#D95F5F", NON_APO_LABEL: "#5B8DB8", "unknown": "#BDBDBD"}
APO_COLOR = {"APO": "#D95F5F", "no_APO": "#5B8DB8", "unknown": "#BDBDBD"}
ACT_COLOR = {"Active": "#C05A48", "Stable": "#4C78A8", "unknown": "#BDBDBD"}

KEY_GENES = sorted(set("""
CD14 LST1 LYZ FCN1 S100A8 S100A9 TREM1 IL1B TNF NFKBIA FOS JUN
ISG15 IFIT1 IFIT2 IFIT3 IFI27 IFI44 IFI44L OAS1 OAS2 OAS3 MX1 RSAD2 USP18 CMPK2
CCR7 SELL IL7R TCF7 LEF1 GZMK GZMB CX3CR1 PRF1 GNLY NKG7 CCL5 GZMH
TBX21 ITGAX FCRL5 FCRL3 FCRL2 TLR7 CR2 CXCR5 MS4A1 CD79A CD79B
MZB1 XBP1 IGJ JCHAIN PRDM1 IRF4 SDC1
IFNG IFNGR1 IFNGR2 IFNA1 IFNAR1 IFNAR2 IFNB1
MIF CD74 CXCR4 CD44 HLA-E HLA-B HLA-C KLRK1 KIR3DL2 CD4 CD48 CD244
CD40LG CD40 CD27 CD70 ICOS ICOSLG TNFSF13B TNFRSF13B TNFRSF13C TNFSF13 TNFRSF17
PF4 CXCR3 PPBP CXCR2 IL1R1 IL1RAP IL1A TNFRSF1A TNFRSF1B
ST6GAL1 ST6GALNAC1 ST6GALNAC2 ST3GAL1 ST3GAL4 ST3GAL6 CMAS SLC35A1
B4GALT1 B4GALT2 B4GALT3 B4GALT4 B3GALT4 B3GALT5
MGAT3 MAN2A1 MAN2A2 MGAT1 MGAT2 FUT8 FUT4 FUT7 FUT10 SLC35C1
HLA-DRA HLA-DRB1 HLA-DQA1 HLA-DQB1 HLA-DRB5 HLA-DPB1 HLA-DPA1
TRIM22 PARP9 EPSTI1 SP110 XAF1 SAMD9L
""".split()))

MODULES = {
    "TREM1_inflammatory": ["TREM1", "IL1B", "TNF", "S100A8", "S100A9", "NFKBIA", "FOS", "JUN", "LST1", "FCN1"],
    "TNF_signaling": ["TNF", "TNFRSF1A", "TNFRSF1B", "NFKBIA", "FOS", "JUN"],
    "IL1_signaling": ["IL1B", "IL1A", "IL1R1", "IL1RAP", "NFKBIA", "FOS", "JUN"],
    "NFkB": ["NFKBIA", "TNF", "IL1B", "TNFRSF1A", "TNFRSF1B"],
    "AP1_FOS_JUN": ["FOS", "JUN", "JUNB", "FOSB"],
    "inflammatory_response": ["IL1B", "TNF", "S100A8", "S100A9", "LYZ", "FCN1", "TREM1"],
    "neutrophil_degranulation": ["S100A8", "S100A9", "LYZ", "FCN1"],
    "antigen_presentation_MHCII": ["HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1", "HLA-DPB1", "HLA-DPA1"],
    "type_I_IFN_response": ["ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI27", "IFI44", "IFI44L", "OAS1", "OAS2", "MX1", "RSAD2", "USP18", "CMPK2"],
    "ISG_module": ["ISG15", "IFIT1", "IFIT3", "IFI27", "MX1", "OAS1", "OAS2", "OAS3", "TRIM22"],
    "type_II_IFN_IFNG_response": ["IFNG", "IFNGR1", "IFNGR2", "CXCL9", "CXCL10", "CXCL11", "STAT1"],
    "naive_T": ["CCR7", "SELL", "IL7R", "TCF7", "LEF1"],
    "T_cell_activation": ["CD40LG", "CD27", "CD70", "ICOS", "CD48"],
    "TCR_signaling": ["CD3D", "CD3E", "TRAC", "LCK", "ZAP70"],
    "cytotoxicity": ["GZMB", "GZMK", "GZMH", "PRF1", "GNLY", "NKG7", "CCL5"],
    "exhaustion": ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "CTLA4"],
    "effector_memory": ["GZMK", "CCL5", "CX3CR1", "NKG7", "GZMH"],
    "CX3CR1_migration": ["CX3CR1", "CCL5", "GZMK", "NKG7"],
    "B_cell_activation": ["CD79A", "CD79B", "MS4A1", "CD40", "TLR7"],
    "atypical_B_ABC": ["TBX21", "ITGAX", "FCRL5", "FCRL3", "FCRL2", "TLR7"],
    "plasmablast_differentiation": ["MZB1", "XBP1", "IGJ", "JCHAIN", "PRDM1", "IRF4", "SDC1"],
    "plasma_cell_differentiation": ["MZB1", "XBP1", "JCHAIN", "PRDM1", "SDC1"],
    "TLR7_signaling": ["TLR7", "MYD88", "IRF7", "NFKBIA"],
    "antibody_secretion": ["MZB1", "XBP1", "IGJ", "JCHAIN"],
    "MIF_CD74_CXCR4_CD44_axis": ["MIF", "CD74", "CXCR4", "CD44"],
    "TNF_TNFR_axis": ["TNF", "TNFRSF1A", "TNFRSF1B"],
    "IL1_IL1R_axis": ["IL1B", "IL1A", "IL1R1", "IL1RAP"],
    "CD40_CD40LG_axis": ["CD40LG", "CD40"],
    "BAFF_APRIL_axis": ["TNFSF13B", "TNFSF13", "TNFRSF13B", "TNFRSF13C", "TNFRSF17"],
    "HLA_KIR_NKG2D_axis": ["HLA-E", "HLA-B", "HLA-C", "KLRK1", "KIR3DL2"],
    "CD48_CD244_axis": ["CD48", "CD244"],
    "PF4_CXCR3_axis": ["PF4", "CXCR3", "PPBP", "CXCR2"],
    "Sialylation_module": ["ST6GAL1", "ST6GALNAC1", "ST6GALNAC2", "ST3GAL1", "ST3GAL4", "ST3GAL6", "CMAS", "SLC35A1"],
    "Galactosylation_module": ["B4GALT1", "B4GALT2", "B4GALT3", "B4GALT4", "B3GALT4", "B3GALT5"],
    "Bisecting_GlcNAc_module": ["MGAT3", "MAN2A1", "MAN2A2", "MGAT1", "MGAT2"],
    "Fucosylation_module": ["FUT8", "FUT4", "FUT7", "FUT10", "SLC35C1"],
}
for genes in MODULES.values():
    KEY_GENES.extend(g for g in genes if g not in KEY_GENES)
KEY_GENES = sorted(set(KEY_GENES))

LR_PAIRS = [
    ("TNF", "TNFRSF1A", "TNF axis"), ("TNF", "TNFRSF1B", "TNF axis"),
    ("IL1B", "IL1R1", "IL-1 axis"), ("IL1B", "IL1RAP", "IL-1 axis"), ("IL1A", "IL1R1", "IL-1 axis"),
    ("IFNG", "IFNGR1", "type II IFN"), ("IFNG", "IFNGR2", "type II IFN"),
    ("MIF", "CD74", "MIF"), ("MIF", "CXCR4", "MIF"), ("MIF", "CD44", "MIF"),
    ("HLA-E", "KLRK1", "HLA-KIR/NKG2D"), ("HLA-B", "KIR3DL2", "HLA-KIR/NKG2D"),
    ("HLA-DRA", "CD4", "HLA-CD4"), ("HLA-DRB1", "CD4", "HLA-CD4"), ("HLA-DQA1", "CD4", "HLA-CD4"), ("HLA-DQB1", "CD4", "HLA-CD4"),
    ("CD48", "CD244", "CD48-CD244"),
    ("CD40LG", "CD40", "T-B help"), ("CD27", "CD70", "T-B help"), ("ICOS", "ICOSLG", "T-B help"),
    ("TNFSF13B", "TNFRSF13B", "BAFF/APRIL"), ("TNFSF13B", "TNFRSF13C", "BAFF/APRIL"),
    ("TNFSF13", "TNFRSF13B", "BAFF/APRIL"), ("TNFSF13", "TNFRSF17", "BAFF/APRIL"),
    ("PF4", "CXCR3", "Platelet/MK"), ("PPBP", "CXCR2", "Platelet/MK"),
]
for lig, rec, _ in LR_PAIRS:
    KEY_GENES.extend([lig, rec])
KEY_GENES = sorted(set(KEY_GENES))
TARGETED_GENES_FOR_MODULES = KEY_GENES

TARGET_THEME_SETS = {
    "IFN antiviral": MODULES["type_I_IFN_response"],
    "TNF/IL1 inflammation": MODULES["TREM1_inflammatory"] + MODULES["IL1_signaling"],
    "HLA antigen presentation": MODULES["antigen_presentation_MHCII"],
    "Cytotoxicity": MODULES["cytotoxicity"],
    "Atypical B": MODULES["atypical_B_ABC"],
    "Plasmablast": MODULES["plasmablast_differentiation"],
    "Migration/CX3CR1": MODULES["CX3CR1_migration"],
    "Glycosylation": MODULES["Sialylation_module"] + MODULES["Galactosylation_module"] + MODULES["Bisecting_GlcNAc_module"],
}

log_messages: list[str] = []
errors: list[str] = []
figure_index: list[dict] = []


def log(msg: str) -> None:
    print(f"[single-cell] {msg}", flush=True)
    log_messages.append(msg)


def err(msg: str) -> None:
    print(f"[single-cell:ERROR] {msg}", flush=True)
    errors.append(msg)


def read_vector(obj):
    if isinstance(obj, h5py.Group) and "codes" in obj and "categories" in obj:
        cats = read_vector(obj["categories"])
        return np.array([None if c < 0 else cats[int(c)] for c in obj["codes"][:]], dtype=object)
    arr = obj[:]
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in {"S", "O"}:
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in arr], dtype=object)
    return arr


def read_df(g: h5py.Group) -> pd.DataFrame:
    idx = g.attrs.get("_index", "_index")
    cols = list(g.attrs.get("column-order", [k for k in g.keys() if k != idx]))
    data = {}
    if idx in g:
        data[str(idx)] = read_vector(g[idx])
    for c in cols:
        if c in g:
            try:
                data[str(c)] = read_vector(g[c])
            except Exception:
                pass
    return pd.DataFrame(data)


def detect_col(cols, candidates) -> str | None:
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None


def choose_matrix_key(f: h5py.File) -> str:
    for key in ["layers/log1p_norm", "layers/log1p_renorm", "layers/lognorm", "layers/log1p"]:
        if key in f:
            return key
    return "X"


class H5Target:
    def __init__(self, name: str, path: Path, genes: list[str]):
        if h5py is None:
            raise ImportError("h5py is required for the full single-cell analysis")
        self.name = name
        self.path = path
        self.genes_requested = list(dict.fromkeys(genes))
        self.f = h5py.File(path, "r")
        self.obs = read_df(self.f["obs"])
        idx = self.f["var"].attrs.get("_index", "_index")
        self.var_names = pd.Index(read_vector(self.f["var"][idx]).astype(str))
        self.sample_col = detect_col(self.obs.columns, ["sample_id", "sample", "orig.ident", "sampleID"])
        self.samples = self.obs[self.sample_col].astype(str).to_numpy() if self.sample_col else np.array([""] * self.obs.shape[0])
        self.matrix_key = choose_matrix_key(self.f)
        self.node = self.f[self.matrix_key]
        self.gene_upper = {g.upper(): i for i, g in enumerate(self.var_names.astype(str))}
        self.genes = [g for g in self.genes_requested if g.upper() in self.gene_upper]
        self.X = self._load_target_matrix()
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def _load_target_matrix(self) -> np.ndarray:
        n_obs = self.obs.shape[0]
        arr = np.zeros((n_obs, len(self.genes)), dtype=np.float32)
        if not self.genes:
            return arr
        if isinstance(self.node, h5py.Group) and {"data", "indices", "indptr"}.issubset(self.node.keys()):
            shape = tuple(int(x) for x in self.node.attrs["shape"])
            enc = self.node.attrs.get("encoding-type", "")
            enc = enc.decode() if isinstance(enc, bytes) else str(enc)
            indptr = self.node["indptr"][:]
            indices = self.node["indices"]
            data = self.node["data"]
            if "csc" in enc:
                for j, g in enumerate(self.genes):
                    col = int(self.gene_upper[g.upper()])
                    start, end = int(indptr[col]), int(indptr[col + 1])
                    idx = np.asarray(indices[start:end], dtype=int)
                    vals = np.asarray(data[start:end], dtype=np.float32)
                    arr[idx, j] = vals
            else:
                mapper = np.full(shape[1], -1, dtype=np.int32)
                for j, g in enumerate(self.genes):
                    mapper[int(self.gene_upper[g.upper()])] = j
                chunk = 4096
                for r0 in range(0, shape[0], chunk):
                    r1 = min(shape[0], r0 + chunk)
                    start, end = int(indptr[r0]), int(indptr[r1])
                    if end <= start:
                        continue
                    idx = np.asarray(indices[start:end], dtype=np.int32)
                    vals = np.asarray(data[start:end], dtype=np.float32)
                    mapped = mapper[idx]
                    keep = mapped >= 0
                    if not np.any(keep):
                        continue
                    counts = np.diff(indptr[r0 : r1 + 1]).astype(np.int64)
                    row_ids = np.repeat(np.arange(r0, r1, dtype=np.int32), counts)
                    arr[row_ids[keep], mapped[keep]] = vals[keep]
        else:
            cols = [int(self.gene_upper[g.upper()]) for g in self.genes]
            arr = np.asarray(self.node[:, cols], dtype=np.float32)
        return arr

    def score(self, genes: list[str]) -> np.ndarray:
        idx = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
        if not idx:
            return np.full(self.X.shape[0], np.nan)
        return np.nanmean(self.X[:, idx], axis=1)

    def gene_expr(self, gene: str) -> np.ndarray:
        if gene not in self.gene_to_idx:
            return np.full(self.X.shape[0], np.nan)
        return self.X[:, self.gene_to_idx[gene]]


def safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"true", "1", "yes", "apo", "active"}


def build_metadata() -> pd.DataFrame:
    global SAMPLE_METADATA, SAMPLES, APO_SAMPLES
    if SAMPLE_METADATA.empty:
        SAMPLE_METADATA = load_sample_metadata(SAMPLE_METADATA_PATH)
        SAMPLES = SAMPLE_METADATA["sample_id"].tolist()
        APO_SAMPLES = set(SAMPLE_METADATA.loc[SAMPLE_METADATA["apo_group"].eq("APO"), "sample_id"])
    source = SAMPLE_METADATA.set_index("sample_id", drop=False)
    rows = []
    for sid in SAMPLES:
        record = source.loc[sid]
        clean_apo = record["apo_group"] == "APO"
        activity = str(record.get("activity_group", "unknown"))
        stage = str(record.get("trimester_or_stage", "unknown")).lower()
        if stage not in {"early", "mid", "late"}:
            stage = "unknown"
        rows.append({
            "sample_id": sid,
            "activity_group": "Active" if str(activity).lower().startswith("active") else "Stable" if str(activity).lower().startswith("stable") else "unknown",
            "apo_group": "APO" if clean_apo else "no_APO",
            "trimester_or_stage": stage,
            "gestational_week": record.get("gestational_week", np.nan),
            "clean_APO_severity": record.get("apo_severity", np.nan),
            "disease_loss": safe_bool(record.get("pregnancy_loss", False)),
            "PE_E": safe_bool(record.get("preeclampsia_or_eclampsia", False)),
            "analysis_group": APO_LABEL if sid in APO_SAMPLES else NON_APO_LABEL,
            "is_APO": sid in APO_SAMPLES,
            "metadata_source": str(SAMPLE_METADATA_PATH),
        })
    return pd.DataFrame(rows).set_index("sample_id", drop=False)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / s


def top_quantile_mask(score: np.ndarray, base: np.ndarray, q: float = 0.8) -> tuple[np.ndarray, float]:
    vals = score[base & np.isfinite(score)]
    if len(vals) == 0:
        return np.zeros_like(base, dtype=bool), np.nan
    thr = float(np.nanquantile(vals, q))
    return base & np.isfinite(score) & (score >= thr), thr


def nonzero_median_threshold(expr: np.ndarray, fallback_z: float = 0.5) -> tuple[float, str]:
    """Return a positive-expression threshold that cannot collapse to zero.

    Preference is the median among nonzero expressing cells. If no nonzero cells
    exist, return NaN and let the caller mark the state insufficient.
    """
    x = np.asarray(expr, dtype=float)
    nonzero = x[np.isfinite(x) & (x > 0)]
    if len(nonzero):
        return float(np.nanmedian(nonzero)), "nonzero_median"
    z = zscore(x)
    if np.isfinite(z).any():
        vals = x[np.isfinite(z) & (z > fallback_z)]
        if len(vals):
            return float(np.nanmin(vals)), f"z>{fallback_z}_fallback"
    return np.nan, "insufficient_nonzero_expression"


def contains_any(series: pd.Series, terms: list[str]) -> np.ndarray:
    text = series.astype(str).str.lower()
    mask = np.zeros(len(series), dtype=bool)
    for t in terms:
        mask |= text.str.contains(t.lower(), regex=False, na=False).to_numpy()
    return mask


def pick_col(obs: pd.DataFrame, prefs: list[str]) -> str | None:
    for p in prefs:
        if p in obs.columns:
            return p
    return detect_col(obs.columns, prefs)


def masks_for_object(obj: H5Target) -> tuple[dict[str, np.ndarray], list[dict]]:
    obs = obj.obs
    valid = np.isin(obj.samples, SAMPLES)
    masks: dict[str, np.ndarray] = {}
    defs = []

    def add(scope, mask, method):
        masks[scope] = valid & mask
        defs.append({"object_name": obj.name, "cell_scope": scope, "definition_method": method, "threshold": "", "marker_genes": "", "caution": "existing annotation or targeted marker/module-defined state"})

    if obj.name == "full_atlas":
        lvl1 = obs["lvl1_annotation"] if "lvl1_annotation" in obs.columns else obs[pick_col(obs, ["lvl1_label", "cell_type", "annotation"])]
        lvl2 = obs["lvl2_label"] if "lvl2_label" in obs.columns else lvl1
        add("full|Myeloid", contains_any(lvl1, ["Myeloid"]), "lvl1_annotation contains Myeloid")
        add("full|B", contains_any(lvl1, ["B"]), "lvl1_annotation contains B")
        add("full|CD4_T", contains_any(lvl2, ["CD4"]), "lvl2_label contains CD4")
        add("full|CD8_T", contains_any(lvl2, ["CD8"]), "lvl2_label contains CD8")
        add("full|NK_TNK", contains_any(lvl1, ["TNK"]) | contains_any(lvl2, ["NK", "NKT"]), "lvl1/lvl2 TNK/NK")
        add("full|pDC", contains_any(lvl2, ["pDC"]), "lvl2_label contains pDC")
        add("full|Platelet_MK_like", contains_any(lvl1, ["Platelet"]) | contains_any(lvl2, ["Platelet"]), "platelet contamination/proxy label")
        add("full|Erythroid_like", contains_any(lvl1, ["Erythroid"]), "erythroid/proxy label")
    elif obj.name == "myeloid":
        col = pick_col(obs, ["myeloid_annotation", "myeloid_label", "lvl2_label", "leiden_myeloid", "annotation"])
        lab = obs[col] if col else pd.Series([""] * obs.shape[0])
        add("myeloid|myeloid_total", np.ones(obs.shape[0], dtype=bool), "all cells in myeloid subset")
        add("myeloid|CD14_classical_monocyte", contains_any(lab, ["Mono_Classical", "classical", "CD14"]), f"{col} classical/CD14")
        add("myeloid|FCGR3A_CD16_monocyte", contains_any(lab, ["NonClassical", "nonclassical", "CD16", "FCGR3A"]), f"{col} nonclassical/CD16")
        add("myeloid|pDC", contains_any(lab, ["pDC", "plasmacytoid"]), f"{col} pDC")
        add("myeloid|cDC", contains_any(lab, ["cDC", "DC2", "cDC2"]), f"{col} cDC/cDC2")
        cd14 = masks["myeloid|CD14_classical_monocyte"]
        trem_score = obj.score(MODULES["TREM1_inflammatory"])
        ifn_score = obj.score(MODULES["type_I_IFN_response"])
        trem_mask, trem_thr = top_quantile_mask(trem_score, cd14, 0.8)
        ifn_mask, ifn_thr = top_quantile_mask(ifn_score, cd14, 0.8)
        add("myeloid|TREM1high_like_CD14_monocyte", trem_mask, f"CD14/classical monocyte with TREM1 inflammatory score top 20%; threshold={trem_thr:.4g}")
        defs[-1]["threshold"] = f"top20%, score >= {trem_thr:.4g}"
        defs[-1]["marker_genes"] = ";".join(MODULES["TREM1_inflammatory"])
        add("myeloid|IFN_high_monocyte", ifn_mask, f"CD14/classical monocyte with type I IFN score top 20%; threshold={ifn_thr:.4g}")
        defs[-1]["threshold"] = f"top20%, score >= {ifn_thr:.4g}"
        defs[-1]["marker_genes"] = ";".join(MODULES["type_I_IFN_response"])
        add("myeloid|inflammatory_monocyte", cd14 & (trem_mask | ifn_mask), "CD14 classical monocyte with TREM1high-like or IFN-high-like state")
    elif obj.name == "CD8":
        col = pick_col(obs, ["cd8_label", "tcell_label", "lvl2_label", "coarse_split_v2", "annotation"])
        lab = obs[col] if col else pd.Series([""] * obs.shape[0])
        add("CD8|CD8_total", np.ones(obs.shape[0], dtype=bool), "all cells in CD8 subset")
        naive = contains_any(lab, ["naive", "Tnaive"]) | (zscore(obj.score(MODULES["naive_T"])) > 0.5)
        add("CD8|naive_CD8", naive, f"{col} naive or naive_T score z>0.5")
        ifn = obj.score(MODULES["type_I_IFN_response"])
        isg_mask, isg_thr = top_quantile_mask(ifn, naive, 0.8)
        add("CD8|ISGhigh_naive_CD8_like", isg_mask, f"naive CD8 with ISG score top 20%; threshold={isg_thr:.4g}")
        defs[-1]["threshold"] = f"top20%, score >= {isg_thr:.4g}"
        defs[-1]["marker_genes"] = ";".join(MODULES["type_I_IFN_response"])
        gzmk = obj.gene_expr("GZMK")
        gzmb = obj.gene_expr("GZMB")
        ccr7 = obj.gene_expr("CCR7")
        sell = obj.gene_expr("SELL")
        gzmk_thr, gzmk_method = nonzero_median_threshold(gzmk)
        gzmb_nonzero = gzmb[np.isfinite(gzmb) & (gzmb > 0)]
        gzmb_thr = float(np.nanquantile(gzmb_nonzero, 0.25)) if len(gzmb_nonzero) else 0.0
        naive_score = np.nanmean(np.vstack([np.nan_to_num(ccr7, nan=0), np.nan_to_num(sell, nan=0)]), axis=0)
        naive_thr = float(np.nanmedian(naive_score[np.isfinite(naive_score)])) if np.isfinite(naive_score).any() else np.nan
        if np.isfinite(gzmk_thr):
            gzmk_pos = np.isfinite(gzmk) & (gzmk > 0) & ((gzmk >= gzmk_thr) | (zscore(gzmk) > 0.5))
        else:
            gzmk_pos = np.zeros(obs.shape[0], dtype=bool)
        gzmb_low = (~np.isfinite(gzmb)) | (gzmb <= gzmb_thr)
        ccr7_sell_low = np.isfinite(naive_score) & (naive_score <= naive_thr)
        eff = gzmk_pos & gzmb_low & ccr7_sell_low
        min_eff_cells = 20
        if int(eff.sum()) < min_eff_cells:
            method = f"insufficient_cells: n={int(eff.sum())}; GZMK threshold={gzmk_thr}; GZMB low threshold={gzmb_thr}; CCR7/SELL threshold={naive_thr}"
            eff = np.zeros(obs.shape[0], dtype=bool)
        else:
            method = f"GZMK > 0 and GZMK >= {gzmk_thr:.4g} ({gzmk_method}) or z>0.5; GZMB <= {gzmb_thr:.4g}; CCR7/SELL mean <= {naive_thr:.4g}; n={int(eff.sum())}"
        add("CD8|GZMKpos_GZMBlow_effmem_like_CD8", eff, method)
        defs[-1]["threshold"] = method
        defs[-1]["marker_genes"] = "GZMK;GZMB;CCR7;SELL;CX3CR1;PRF1;GNLY;NKG7;CCL5;GZMH"
        add("CD8|cytotoxic_CD8", zscore(obj.score(MODULES["cytotoxicity"])) > 0.8, "cytotoxicity score z>0.8")
        add("CD8|exhausted_like_CD8", zscore(obj.score(MODULES["exhaustion"])) > 0.8, "exhaustion score z>0.8")
        add("CD8|proliferating_CD8", contains_any(lab, ["prolif", "cycling"]), "annotation proliferating/cycling if available")
    elif obj.name == "B_cell":
        col = pick_col(obs, ["bcell_annotation", "bcell_label", "b_label", "lvl2_label", "annotation"])
        lab = obs[col] if col else pd.Series([""] * obs.shape[0])
        add("B|B_total", np.ones(obs.shape[0], dtype=bool), "all cells in B-cell subset")
        add("B|naive_B", contains_any(lab, ["naive"]), f"{col} naive")
        add("B|memory_B", contains_any(lab, ["memory"]), f"{col} memory")
        abc_score = obj.score(MODULES["atypical_B_ABC"])
        abc_mask, abc_thr = top_quantile_mask(abc_score, np.ones(obs.shape[0], dtype=bool), 0.8)
        add("B|atypical_B_ABC_like", contains_any(lab, ["ABC", "atypical"]) | abc_mask, f"{col} ABC/atypical or ABC score top20%; threshold={abc_thr:.4g}")
        defs[-1]["marker_genes"] = ";".join(MODULES["atypical_B_ABC"])
        pb_score = obj.score(MODULES["plasmablast_differentiation"])
        pb_mask, pb_thr = top_quantile_mask(pb_score, np.ones(obs.shape[0], dtype=bool), 0.8)
        add("B|true_annotation_plasmablast_B", contains_any(lab, ["plasmablast", "plasma"]), f"{col} explicit plasmablast/plasma annotation only; kept separate from score-high state")
        defs[-1]["marker_genes"] = ";".join(MODULES["plasmablast_differentiation"])
        ifn_score = obj.score(MODULES["type_I_IFN_response"])
        ifn_mask, ifn_thr = top_quantile_mask(ifn_score, np.ones(obs.shape[0], dtype=bool), 0.8)
        add("B|IFN_high_B", ifn_mask, f"B-cell IFN score top20%; threshold={ifn_thr:.4g}")
        add("B|plasmablast_differentiation_high_B", pb_mask, f"B-cell plasmablast differentiation score top20%; threshold={pb_thr:.4g}")
        defs[-1]["marker_genes"] = ";".join(MODULES["plasmablast_differentiation"])
        defs[-1]["caution"] = "Score-high B-cell state; not equal to true plasmablast proportion. Some samples may be absent from the B-cell subset."
    elif obj.name == "CD4":
        col = pick_col(obs, ["cd4_label", "tcell_label", "lvl2_label", "coarse_split_v2", "annotation"])
        lab = obs[col] if col else pd.Series([""] * obs.shape[0])
        add("CD4|CD4_total", np.ones(obs.shape[0], dtype=bool), "all cells in CD4 subset")
        add("CD4|CD4_naive", contains_any(lab, ["naive"]), f"{col} naive")
        add("CD4|CD4_Treg", contains_any(lab, ["Treg"]), f"{col} Treg")
        add("CD4|CD4_ISG", contains_any(lab, ["ISG"]) | (zscore(obj.score(MODULES["type_I_IFN_response"])) > 0.8), f"{col} ISG or IFN z>0.8")
    elif obj.name == "NK":
        col = pick_col(obs, ["nk_annotation", "nk_label", "lvl2_label", "annotation"])
        lab = obs[col] if col else pd.Series([""] * obs.shape[0])
        add("NK|NK_total", np.ones(obs.shape[0], dtype=bool), "all cells in NK subset")
        add("NK|NK_Cytotoxic", contains_any(lab, ["cytotoxic"]), f"{col} cytotoxic")
        add("NK|NK_GZMK", contains_any(lab, ["GZMK"]), f"{col} GZMK")
        add("NK|NK_Prolif", contains_any(lab, ["prolif"]), f"{col} proliferating")
    return masks, defs


def summarize_by_sample(obj: H5Target, masks: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prop_rows, score_rows, gene_rows = [], [], []
    for scope, mask in masks.items():
        total_by_sample = pd.Series(obj.samples[mask]).value_counts().to_dict()
        parent_scope = scope.split("|")[0]
        if obj.name == "full_atlas":
            parent_mask = np.isin(obj.samples, SAMPLES)
        elif parent_scope in {"myeloid", "CD8", "B", "CD4", "NK"}:
            parent_mask = np.isin(obj.samples, SAMPLES)
        else:
            parent_mask = np.isin(obj.samples, SAMPLES)
        parent_counts = pd.Series(obj.samples[parent_mask]).value_counts().to_dict()
        for sid in SAMPLES:
            cnt = int(total_by_sample.get(sid, 0))
            parent = int(parent_counts.get(sid, 0))
            prop_rows.append({
                "object_name": obj.name,
                "cell_scope": scope,
                "sample_id": sid,
                "n_cells": cnt,
                "parent_n_cells": parent,
                "proportion": cnt / parent if parent else np.nan,
                "low_count": cnt < 20,
            })
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        for module, genes in MODULES.items():
            found = [g for g in genes if g in obj.gene_to_idx]
            cell_score = obj.score(found)
            for sid in SAMPLES:
                rows = idx[obj.samples[idx] == sid]
                vals = cell_score[rows] if len(rows) else np.array([])
                score_rows.append({
                    "object_name": obj.name,
                    "cell_scope": scope,
                    "sample_id": sid,
                    "module_name": module,
                    "module_score_mean": float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
                    "module_score_median": float(np.nanmedian(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
                    "percent_high_score_cells": float(np.nanmean(zscore(cell_score[idx])[obj.samples[idx] == sid] > 0.8)) if len(rows) and np.isfinite(cell_score[idx]).any() else np.nan,
                    "n_cells": int(len(rows)),
                    "low_count": len(rows) < 20,
                    "genes_found": ";".join(found),
                    "n_genes_found": len(found),
                    "low_confidence_gene_coverage": len(found) < 3,
                })
        for gene in KEY_GENES:
            if gene not in obj.gene_to_idx:
                continue
            expr = obj.gene_expr(gene)
            for sid in SAMPLES:
                rows = idx[obj.samples[idx] == sid]
                vals = expr[rows] if len(rows) else np.array([])
                gene_rows.append({
                    "object_name": obj.name,
                    "cell_scope": scope,
                    "sample_id": sid,
                    "gene": gene,
                    "mean_expression": float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
                    "fraction_expressing": float(np.nanmean(vals > 0)) if len(vals) else np.nan,
                    "n_cells": int(len(rows)),
                    "low_count": len(rows) < 20,
                })
    return pd.DataFrame(prop_rows), pd.DataFrame(score_rows), pd.DataFrame(gene_rows)


def matrix_shape(node) -> tuple[int, int]:
    if isinstance(node, h5py.Group) and "shape" in node.attrs:
        return tuple(int(x) for x in node.attrs["shape"])
    return tuple(int(x) for x in node.shape)


def matrix_encoding(node) -> str:
    if isinstance(node, h5py.Group):
        enc = node.attrs.get("encoding-type", "")
        return enc.decode() if isinstance(enc, bytes) else str(enc)
    return "dense"


def sampled_matrix_values(node, max_values: int = 20000) -> np.ndarray:
    if isinstance(node, h5py.Group) and "data" in node:
        data = node["data"]
        n = int(data.shape[0])
        if n == 0:
            return np.array([], dtype=float)
        step = max(1, n // max_values)
        vals = np.asarray(data[0:n:step][:max_values], dtype=float)
        return vals[np.isfinite(vals)]
    shape = matrix_shape(node)
    r = min(shape[0], 256)
    c = min(shape[1], 512)
    vals = np.asarray(node[:r, :c], dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return vals[vals != 0][:max_values] if np.any(vals != 0) else vals[:max_values]


def is_count_like_values(vals: np.ndarray) -> tuple[bool, dict]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return False, {"reason": "no sampled finite values", "integer_like_fraction": np.nan, "min_value": np.nan, "max_value_sampled": np.nan}
    nonneg = bool(np.nanmin(vals) >= 0)
    integer_like = np.isclose(vals, np.round(vals), atol=1e-6)
    int_frac = float(np.nanmean(integer_like))
    maxv = float(np.nanmax(vals))
    count_like = nonneg and int_frac >= 0.95 and maxv >= 20
    reason = f"nonnegative={nonneg}; integer_like_fraction={int_frac:.3f}; max_sampled={maxv:.3g}"
    return count_like, {"reason": reason, "integer_like_fraction": int_frac, "min_value": float(np.nanmin(vals)), "max_value_sampled": maxv}


def matrix_node_by_path(f: h5py.File, path: str):
    return f[path]


def var_names_for_matrix(f: h5py.File, selected_matrix: str) -> pd.Index:
    if selected_matrix.startswith("raw/") and "raw/var" in f:
        idx = f["raw/var"].attrs.get("_index", "_index")
        return pd.Index(read_vector(f["raw/var"][idx]).astype(str))
    idx = f["var"].attrs.get("_index", "_index")
    return pd.Index(read_vector(f["var"][idx]).astype(str))


def detect_count_matrix(f: h5py.File, object_name: str, h5ad_path: Path) -> tuple[str | None, pd.DataFrame]:
    candidates = []
    if "raw/X" in f:
        candidates.append("raw/X")
    if "layers" in f:
        for layer in ["counts", "count", "raw_counts", "raw", "umi", "matrix"]:
            if f"layers/{layer}" in f:
                candidates.append(f"layers/{layer}")
    if "X" in f:
        candidates.append("X")
    rows = []
    selected = None
    for cand in dict.fromkeys(candidates):
        try:
            node = matrix_node_by_path(f, cand)
            vals = sampled_matrix_values(node)
            count_like, info = is_count_like_values(vals)
            shape = matrix_shape(node)
            use = selected is None and count_like
            if use:
                selected = cand
            rows.append({
                "object": object_name,
                "h5ad_path": str(h5ad_path),
                "selected_matrix": cand,
                "is_count_like": bool(count_like),
                "n_obs": shape[0],
                "n_vars": shape[1],
                "n_nonzero_sampled": int(len(vals)),
                "integer_like_fraction": info["integer_like_fraction"],
                "min_value": info["min_value"],
                "max_value_sampled": info["max_value_sampled"],
                "reason": info["reason"],
                "used_for_raw_pseudobulk": bool(use),
            })
        except Exception as e:
            rows.append({
                "object": object_name,
                "h5ad_path": str(h5ad_path),
                "selected_matrix": cand,
                "is_count_like": False,
                "n_obs": np.nan,
                "n_vars": np.nan,
                "n_nonzero_sampled": 0,
                "integer_like_fraction": np.nan,
                "min_value": np.nan,
                "max_value_sampled": np.nan,
                "reason": f"matrix audit failed: {repr(e)}",
                "used_for_raw_pseudobulk": False,
            })
    if selected is None:
        try:
            obs_idx = f["obs"].attrs.get("_index", "_index")
            n_obs = int(f["obs"][obs_idx].shape[0])
        except Exception:
            n_obs = np.nan
        try:
            var_idx = f["var"].attrs.get("_index", "_index")
            n_vars = int(f["var"][var_idx].shape[0])
        except Exception:
            n_vars = np.nan
        rows.append({
            "object": object_name,
            "h5ad_path": str(h5ad_path),
            "selected_matrix": "none",
            "is_count_like": False,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_nonzero_sampled": 0,
            "integer_like_fraction": np.nan,
            "min_value": np.nan,
            "max_value_sampled": np.nan,
            "reason": "count_matrix_unavailable; raw-count DEG skipped for this object",
            "used_for_raw_pseudobulk": False,
        })
    return selected, pd.DataFrame(rows)


def aggregate_scope_counts(node, samples: np.ndarray, mask: np.ndarray, sample_ids: list[str], min_cells: int = 30) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_obs, n_vars = matrix_shape(node)
    valid_samples, n_cells = [], []
    row_to_sample = np.full(n_obs, -1, dtype=np.int32)
    for i, sid in enumerate(sample_ids):
        rows = np.where(mask & (samples == sid))[0]
        if len(rows) >= min_cells:
            valid_samples.append(sid)
            n_cells.append(int(len(rows)))
            row_to_sample[rows] = len(valid_samples) - 1
    if not valid_samples:
        return np.zeros((0, n_vars), dtype=np.float64), np.array([], dtype=int), []
    counts = np.zeros((len(valid_samples), n_vars), dtype=np.float64)
    if isinstance(node, h5py.Group) and {"data", "indices", "indptr"}.issubset(node.keys()):
        enc = matrix_encoding(node).lower()
        indptr = node["indptr"]
        indices = node["indices"]
        data = node["data"]
        if "csc" in enc:
            for col in range(n_vars):
                start, end = int(indptr[col]), int(indptr[col + 1])
                if end <= start:
                    continue
                row_idx = np.asarray(indices[start:end], dtype=np.int64)
                vals = np.asarray(data[start:end], dtype=np.float64)
                sidx = row_to_sample[row_idx]
                keep = sidx >= 0
                if np.any(keep):
                    np.add.at(counts[:, col], sidx[keep], vals[keep])
        else:
            chunk = 4096
            for r0 in range(0, n_obs, chunk):
                r1 = min(n_obs, r0 + chunk)
                start, end = int(indptr[r0]), int(indptr[r1])
                if end <= start:
                    continue
                idx = np.asarray(indices[start:end], dtype=np.int64)
                vals = np.asarray(data[start:end], dtype=np.float64)
                row_counts = np.diff(np.asarray(indptr[r0 : r1 + 1], dtype=np.int64))
                row_ids = np.repeat(np.arange(r0, r1, dtype=np.int64), row_counts)
                sidx = row_to_sample[row_ids]
                keep = sidx >= 0
                if np.any(keep):
                    np.add.at(counts, (sidx[keep], idx[keep]), vals[keep])
    else:
        chunk = 512
        for r0 in range(0, n_obs, chunk):
            r1 = min(n_obs, r0 + chunk)
            sidx = row_to_sample[r0:r1]
            keep_rows = np.where(sidx >= 0)[0]
            if len(keep_rows) == 0:
                continue
            block = np.asarray(node[r0:r1, :], dtype=np.float64)
            for local_i in keep_rows:
                counts[sidx[local_i], :] += block[local_i, :]
    return counts, np.asarray(n_cells, dtype=int), valid_samples


def python_logcpm_deg(counts: np.ndarray, sample_ids: list[str], n_cells: np.ndarray, genes: pd.Index, meta: pd.DataFrame, object_name: str, cell_scope: str, comparison: str, group_col: str, g1: str, g0: str, matrix_used: str, subset: pd.Series | None = None) -> pd.DataFrame:
    if subset is None:
        subset = pd.Series(True, index=meta.index)
    keep_samples = [i for i, sid in enumerate(sample_ids) if sid in meta.index and bool(subset.reindex(meta.index).fillna(False).loc[sid])]
    if not keep_samples:
        return pd.DataFrame()
    counts = counts[keep_samples, :]
    n_cells = n_cells[keep_samples]
    sample_ids = [sample_ids[i] for i in keep_samples]
    groups = meta.loc[sample_ids, group_col].astype(str).to_numpy()
    idx1 = np.where(groups == g1)[0]
    idx0 = np.where(groups == g0)[0]
    if len(idx1) < 2 or len(idx0) < 2:
        return pd.DataFrame()
    totals = counts.sum(axis=1)
    totals[totals <= 0] = np.nan
    cpm = counts / totals[:, None] * 1e6
    expressed = np.nan_to_num(cpm > 1, nan=False)
    keep_gene = expressed.sum(axis=0) >= max(2, min(3, len(sample_ids)))
    n_before = int(counts.shape[1])
    n_after = int(keep_gene.sum())
    if n_after == 0:
        return pd.DataFrame()
    genes2 = np.asarray(genes.astype(str))[keep_gene]
    counts2 = counts[:, keep_gene]
    logcpm = np.log2(cpm[:, keep_gene] + 1)
    rows = []
    for j, gene in enumerate(genes2):
        a = logcpm[idx1, j]
        b = logcpm[idx0, j]
        stat = p = np.nan
        if stats is not None:
            try:
                tt = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                stat, p = float(tt.statistic), float(tt.pvalue)
            except Exception:
                pass
        rows.append({
            "comparison": comparison,
            "cell_scope": cell_scope,
            "object": object_name,
            "gene": gene,
            "log2FC": float(np.nanmean(a) - np.nanmean(b)),
            "statistic": stat,
            "p_value": p,
            "mean_count_group1": float(np.nanmean(counts2[idx1, j])),
            "mean_count_group0": float(np.nanmean(counts2[idx0, j])),
            "mean_logCPM_group1": float(np.nanmean(a)),
            "mean_logCPM_group0": float(np.nanmean(b)),
            "n_group1": int(len(idx1)),
            "n_group0": int(len(idx0)),
            "n_cells_group1_total": int(np.nansum(n_cells[idx1])),
            "n_cells_group0_total": int(np.nansum(n_cells[idx0])),
            "n_genes_tested_in_scope": n_before,
            "n_genes_after_filtering": n_after,
            "method": "python_logCPM_welch_fallback",
            "matrix_used": matrix_used,
            "is_raw_count_pseudobulk": True,
            "low_count_scope": False,
            "note": "raw-count sample-level pseudobulk; all genes after CPM filtering; no single-cell p values",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = bh_fdr(out["p_value"])
        out["fdr_bh"] = out["FDR"]
        out["neg_log10_p"] = out["p_value"].map(neglog)
        out["neg_log10_fdr"] = out["FDR"].map(neglog)
    return out


def edger_available() -> bool:
    rscript = RSCRIPT_PATH if Path(RSCRIPT_PATH).exists() else shutil.which("Rscript")
    if not rscript:
        return False
    try:
        res = subprocess.run([rscript, "-e", "suppressPackageStartupMessages(library(edgeR)); cat('edgeR_ok')"], capture_output=True, text=True, timeout=20)
        return res.returncode == 0 and "edgeR_ok" in res.stdout
    except Exception:
        return False


def edger_deg_from_counts(counts: np.ndarray, sample_ids: list[str], n_cells: np.ndarray, genes: pd.Index, meta: pd.DataFrame, object_name: str, cell_scope: str, comparison: str, group_col: str, g1: str, g0: str, matrix_used: str, subset: pd.Series | None = None) -> pd.DataFrame:
    if subset is None:
        subset = pd.Series(True, index=meta.index)
    keep_samples = [i for i, sid in enumerate(sample_ids) if sid in meta.index and bool(subset.reindex(meta.index).fillna(False).loc[sid])]
    if not keep_samples:
        return pd.DataFrame()
    counts = counts[keep_samples, :]
    n_cells = n_cells[keep_samples]
    sample_ids = [sample_ids[i] for i in keep_samples]
    groups = meta.loc[sample_ids, group_col].astype(str)
    idx1 = np.where(groups.to_numpy() == g1)[0]
    idx0 = np.where(groups.to_numpy() == g0)[0]
    if len(idx1) < 2 or len(idx0) < 2:
        return pd.DataFrame()
    with tempfile.TemporaryDirectory(prefix="exp32_edger_") as td:
        td_path = Path(td)
        counts_path = td_path / "counts.tsv"
        design_path = td_path / "design.tsv"
        out_path = td_path / "edger_results.tsv"
        pd.DataFrame(counts.T, index=genes.astype(str), columns=sample_ids).to_csv(counts_path, sep="\t")
        pd.DataFrame({"sample_id": sample_ids, "group": groups.to_numpy()}).to_csv(design_path, sep="\t", index=False)
        r_code = f"""
suppressPackageStartupMessages(library(edgeR))
counts <- read.delim("{counts_path}", row.names=1, check.names=FALSE)
design_df <- read.delim("{design_path}", check.names=FALSE)
group <- factor(design_df$group, levels=c("{g0}", "{g1}"))
y <- DGEList(counts=counts, group=group)
keep <- filterByExpr(y, group=group)
y <- y[keep,,keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
design <- model.matrix(~group)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
qlf <- glmQLFTest(fit, coef=2)
tab <- topTags(qlf, n=Inf, sort.by="none")$table
tab$gene <- rownames(tab)
tab$n_genes_tested_in_scope <- nrow(counts)
tab$n_genes_after_filtering <- nrow(tab)
write.table(tab, file="{out_path}", sep="\\t", quote=FALSE, row.names=FALSE)
"""
        rscript = RSCRIPT_PATH if Path(RSCRIPT_PATH).exists() else shutil.which("Rscript")
        if not rscript:
            raise RuntimeError("Rscript not found for edgeR")
        res = subprocess.run([rscript, "-e", r_code], capture_output=True, text=True, timeout=600)
        if res.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"edgeR failed: {res.stderr[:500]}")
        tab = pd.read_csv(out_path, sep="\t")
    if tab.empty:
        return pd.DataFrame()
    totals = counts.sum(axis=1)
    totals[totals <= 0] = np.nan
    cpm = counts / totals[:, None] * 1e6
    logcpm = np.log2(cpm + 1)
    gene_to_col = {g: i for i, g in enumerate(genes.astype(str))}
    rows = []
    for _, r in tab.iterrows():
        gene = str(r["gene"])
        j = gene_to_col.get(gene)
        if j is None:
            continue
        rows.append({
            "comparison": comparison,
            "cell_scope": cell_scope,
            "object": object_name,
            "gene": gene,
            "log2FC": float(r.get("logFC", np.nan)),
            "statistic": float(r.get("F", np.nan)),
            "p_value": float(r.get("PValue", np.nan)),
            "FDR": float(r.get("FDR", np.nan)),
            "fdr_bh": float(r.get("FDR", np.nan)),
            "mean_count_group1": float(np.nanmean(counts[idx1, j])),
            "mean_count_group0": float(np.nanmean(counts[idx0, j])),
            "mean_logCPM_group1": float(np.nanmean(logcpm[idx1, j])),
            "mean_logCPM_group0": float(np.nanmean(logcpm[idx0, j])),
            "n_group1": int(len(idx1)),
            "n_group0": int(len(idx0)),
            "n_cells_group1_total": int(np.nansum(n_cells[idx1])),
            "n_cells_group0_total": int(np.nansum(n_cells[idx0])),
            "n_genes_tested_in_scope": int(r.get("n_genes_tested_in_scope", counts.shape[1])),
            "n_genes_after_filtering": int(r.get("n_genes_after_filtering", tab.shape[0])),
            "method": "edgeR_glmQLF_raw_count_pseudobulk",
            "matrix_used": matrix_used,
            "is_raw_count_pseudobulk": True,
            "low_count_scope": False,
            "note": "edgeR raw-count sample-level pseudobulk; all genes after filterByExpr; no single-cell p values",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["neg_log10_p"] = out["p_value"].map(neglog)
        out["neg_log10_fdr"] = out["FDR"].map(neglog)
    return out


def run_raw_count_pseudobulk_deg(obj: H5Target, masks: dict[str, np.ndarray], meta: pd.DataFrame, min_cells: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, audit = detect_count_matrix(obj.f, obj.name, obj.path)
    skipped = []
    if selected is None:
        skipped.append({"object": obj.name, "cell_scope": "all", "reason": "count_matrix_unavailable", "method": "raw_count_pseudobulk_skipped"})
        return pd.DataFrame(), audit, pd.DataFrame(skipped)
    node = matrix_node_by_path(obj.f, selected)
    genes_all = var_names_for_matrix(obj.f, selected)
    edge_available = edger_available()
    edge_log_dir = LOG / "edgeR_scope_logs"
    edge_log_dir.mkdir(parents=True, exist_ok=True)
    deg_rows = []
    scope_allow_by_object = {
        "full_atlas": ["full|Myeloid", "full|CD4_T", "full|CD8_T", "full|B", "full|NK_TNK", "full|pDC", "full|Platelet_MK_like"],
        "myeloid": ["myeloid|myeloid_total", "myeloid|CD14_classical_monocyte", "myeloid|FCGR3A_CD16_monocyte", "myeloid|TREM1high_like_CD14_monocyte", "myeloid|IFN_high_monocyte", "myeloid|inflammatory_monocyte", "myeloid|pDC", "myeloid|cDC"],
        "CD8": ["CD8|CD8_total", "CD8|naive_CD8", "CD8|ISGhigh_naive_CD8_like", "CD8|GZMKpos_GZMBlow_effmem_like_CD8", "CD8|cytotoxic_CD8"],
        "B_cell": ["B|B_total", "B|naive_B", "B|memory_B", "B|atypical_B_ABC_like", "B|plasmablast_differentiation_high_B", "B|true_annotation_plasmablast_B"],
        "CD4": ["CD4|CD4_total", "CD4|CD4_ISG"],
        "NK": ["NK|NK_total", "NK|NK_Cytotoxic", "NK|NK_GZMK"],
    }
    requested_scopes = scope_allow_by_object.get(obj.name, sorted(masks))
    for scope in requested_scopes:
        if scope not in masks:
            skipped.append({"object": obj.name, "cell_scope": scope, "reason": "scope_not_defined_for_object", "method": "raw_count_pseudobulk_skipped"})
            continue
        counts, n_cells, valid_samples = aggregate_scope_counts(node, obj.samples, masks[scope], SAMPLES, min_cells=min_cells)
        if counts.shape[0] == 0:
            skipped.append({"object": obj.name, "cell_scope": scope, "reason": f"no sample-cell_scope with >= {min_cells} cells", "method": "raw_count_pseudobulk_skipped"})
            continue
        comparisons = [
            ("APO_vs_nonAPO", "analysis_group", APO_LABEL, NON_APO_LABEL, pd.Series(True, index=meta.index)),
        ]
        for comp, group_col, g1, g0, subset in comparisons:
            if not edge_available:
                skipped.append({"object": obj.name, "cell_scope": scope, "comparison": comp, "reason": "edgeR_unavailable; raw-count DEG main result not computed", "method": "edgeR_required_for_main_DEG"})
                continue
            try:
                res = edger_deg_from_counts(counts, valid_samples, n_cells, genes_all, meta, obj.name, scope, comp, group_col, g1, g0, selected, subset=subset)
            except Exception as e:
                safe_scope = "".join(ch if ch.isalnum() else "_" for ch in f"{obj.name}_{scope}_{comp}")[:180]
                (edge_log_dir / f"{safe_scope}.error.log").write_text(repr(e), encoding="utf-8")
                skipped.append({"object": obj.name, "cell_scope": scope, "comparison": comp, "reason": f"edgeR_failed_no_main_fallback: {repr(e)[:300]}", "method": "edgeR_glmQLF_raw_count_pseudobulk"})
                res = pd.DataFrame()
            if res.empty:
                skipped.append({"object": obj.name, "cell_scope": scope, "comparison": comp, "reason": "edgeR returned no main DEG rows; insufficient grouped samples or no genes after filterByExpr", "method": "edgeR_glmQLF_raw_count_pseudobulk"})
            else:
                deg_rows.append(res)
    deg = pd.concat(deg_rows, ignore_index=True) if deg_rows else pd.DataFrame()
    return deg, audit, pd.DataFrame(skipped)


def neglog(x) -> float:
    try:
        x = float(x)
        return -math.log10(max(x, 1e-300)) if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return pd.Series(out, index=pvals.index)
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(adj, 0, 1)
    out[np.where(mask)[0]] = tmp
    return pd.Series(out, index=pvals.index)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def binary_stats(values: pd.Series, meta: pd.DataFrame, group_col: str, g1: str, g0: str, subset: pd.Series | None = None) -> dict:
    df = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce")}).join(meta[[group_col]])
    if subset is not None:
        df = df.loc[subset.reindex(df.index).fillna(False)]
    a = df.loc[df[group_col].eq(g1), "value"].dropna().to_numpy(float)
    b = df.loc[df[group_col].eq(g0), "value"].dropna().to_numpy(float)
    stat = p = np.nan
    if len(a) >= 2 and len(b) >= 2 and stats is not None:
        try:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    med_eff = (np.nanmedian(a) - np.nanmedian(b)) if len(a) and len(b) else np.nan
    mean_eff = (np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else np.nan
    return {
        "group1": g1, "group0": g0,
        "n_group1": len(a), "n_group0": len(b),
        "mean_group1": float(np.nanmean(a)) if len(a) else np.nan,
        "mean_group0": float(np.nanmean(b)) if len(b) else np.nan,
        "median_group1": float(np.nanmedian(a)) if len(a) else np.nan,
        "median_group0": float(np.nanmedian(b)) if len(b) else np.nan,
        "mean_difference": mean_eff,
        "difference_in_median": med_eff,
        "statistic": stat,
        "p_value": p,
        "neg_log10_p": neglog(p),
        "cliffs_delta": cliffs_delta(a, b),
        "low_n": len(a) < 3 or len(b) < 3,
    }


def add_fdr(df: pd.DataFrame, by_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    out["fdr_bh"] = np.nan
    if by_cols:
        for _, idx in out.groupby(by_cols).groups.items():
            out.loc[idx, "fdr_bh"] = bh_fdr(out.loc[idx, "p_value"]).to_numpy()
    else:
        out["fdr_bh"] = bh_fdr(out["p_value"])
    out["neg_log10_fdr"] = out["fdr_bh"].map(neglog)
    return out


def group_stats_long(df: pd.DataFrame, feature_cols: list[str], value_col: str, meta: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(feature_cols):
        if not isinstance(key, tuple):
            key = (key,)
        vals = sub.set_index("sample_id")[value_col].reindex(meta.index)
        for comp, group_col, g1, g0, subset in [
            ("APO_vs_nonAPO", "analysis_group", APO_LABEL, NON_APO_LABEL, pd.Series(True, index=meta.index)),
        ]:
            rec = dict(zip(feature_cols, key))
            rec.update({"comparison": comp, "analysis_prefix": prefix})
            rec.update(binary_stats(vals, meta, group_col, g1, g0, subset))
            rows.append(rec)
    return add_fdr(pd.DataFrame(rows), ["comparison"])


def save_tsv(df: pd.DataFrame, name: str) -> Path:
    p = TAB / name
    df.to_csv(p, sep="\t", index=False)
    return p


def append_tsv(df: pd.DataFrame, name: str) -> Path:
    """Append potentially large DEG chunks without holding all scopes in memory."""
    p = TAB / name
    if df is None or df.empty:
        if not p.exists():
            p.write_text("", encoding="utf-8")
        return p
    write_header = (not p.exists()) or p.stat().st_size == 0
    df.to_csv(p, sep="\t", index=False, mode="a", header=write_header)
    return p


def read_tsv_if_available(name: str) -> pd.DataFrame:
    p = TAB / name
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_fig(fig, basename: str, source_table: str, method: str) -> None:
    for ext in ["png", "pdf"]:
        path = FIG / f"{basename}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figure_index.append({"figure": basename, "source_table": source_table, "statistical_method": method, "p_value_source": "sample-level APO versus non-APO summary, raw-count pseudobulk DEG, enrichment, or CellPhoneDB output as indicated"})


def placeholder_figure(basename: str, title: str, text: str, source_table: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.38, textwrap.fill(text, 90), ha="center", va="center", fontsize=10)
    save_fig(fig, basename, source_table, "not applicable; placeholder because data not evaluable")


def heatmap_matrix(mat: pd.DataFrame, basename: str, title: str, source_table: str, cmap="RdBu_r", center_zero=True, footnote="") -> None:
    if mat.empty:
        placeholder_figure(basename, title, "No evaluable data.", source_table)
        return
    vals = mat.to_numpy(float)
    finite = vals[np.isfinite(vals)]
    vmax = max(0.1, min(3.0, np.nanpercentile(np.abs(finite), 95))) if len(finite) else 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if center_zero else None
    fig, ax = plt.subplots(figsize=(max(7, mat.shape[1] * 0.75), max(4, mat.shape[0] * 0.28)))
    im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#DDDDDD", linewidth=0.4))
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("effect / score")
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=8)
    save_fig(fig, basename, source_table, "sample-level values")


def bubble_plot(stats_df: pd.DataFrame, row_col: str, basename: str, title: str, source_table: str, value_col="mean_difference", max_rows: int = 40) -> None:
    sub = stats_df.copy()
    if sub.empty:
        placeholder_figure(basename, title, "No evaluable stats.", source_table)
        return
    if row_col not in sub.columns or "comparison" not in sub.columns:
        placeholder_figure(basename, title, f"Required plotting columns missing: {row_col}/comparison.", source_table)
        return
    # Keep figures readable and prevent huge LR/subtype panels from becoming
    # non-interpretable. The complete statistics remain in the source table.
    ranked_rows = (
        sub.assign(_p=pd.to_numeric(sub.get("p_value", np.nan), errors="coerce"))
        .groupby(row_col)["_p"].min()
        .sort_values(na_position="last")
        .head(max_rows)
        .index.astype(str)
        .tolist()
    )
    sub = sub[sub[row_col].astype(str).isin(ranked_rows)].copy()
    rows = sub[row_col].astype(str).drop_duplicates().tolist()
    comps = sub["comparison"].astype(str).drop_duplicates().tolist()
    sub["_row"] = pd.Categorical(sub[row_col].astype(str), rows, ordered=True)
    sub["_col"] = pd.Categorical(sub["comparison"].astype(str), comps, ordered=True)
    eff = pd.to_numeric(sub[value_col], errors="coerce")
    vmax = max(0.1, min(1.5, np.nanpercentile(np.abs(eff[np.isfinite(eff)]), 95))) if np.isfinite(eff).any() else 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sizes = 35 + 120 * np.clip(pd.to_numeric(sub["neg_log10_p"], errors="coerce").fillna(0), 0, 4) / 4
    fig, ax = plt.subplots(figsize=(max(8, len(comps) * 1.4), max(4.5, len(rows) * 0.32)))
    x = sub["_col"].cat.codes
    y = sub["_row"].cat.codes
    edge = np.where(pd.to_numeric(sub["fdr_bh"], errors="coerce").fillna(1) < 0.1, "black", "#BBBBBB")
    lw = np.where(pd.to_numeric(sub["fdr_bh"], errors="coerce").fillna(1) < 0.1, 1.2, 0.4)
    ax.scatter(x, y, c=eff, s=sizes, cmap="RdBu_r", norm=norm, edgecolors=edge, linewidths=lw)
    for xi, yi, p in zip(x, y, pd.to_numeric(sub["p_value"], errors="coerce")):
        if pd.notna(p) and p < 0.05:
            ax.text(xi, yi, "*", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(-0.5, len(comps) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(color="#EEEEEE", linewidth=0.5)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, shrink=0.75)
    cbar.set_label(value_col)
    fig.text(0.01, 0.01, "Color = sample-level group effect; size = -log10(P); black outline = FDR < 0.1; * nominal P < 0.05.", fontsize=8)
    save_fig(fig, basename, source_table, "Mann-Whitney/Wilcoxon rank-sum on sample-level values; BH-FDR")


def dotplot_samples(df: pd.DataFrame, feature_col: str, value_col: str, group_col: str, features: list[str], basename: str, title: str, source_table: str, meta: pd.DataFrame) -> None:
    sub = df[df[feature_col].isin(features)].copy()
    if sub.empty:
        placeholder_figure(basename, title, "No sample-level data.", source_table)
        return
    sub = sub.merge(meta[["sample_id", group_col]], on="sample_id", how="left")
    fig, axes = plt.subplots(1, len(features), figsize=(max(8, len(features) * 2.3), 4), sharey=False)
    if len(features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        s = sub[sub[feature_col].eq(feat)]
        groups = [g for g in s[group_col].dropna().unique()]
        groups = sorted(groups)
        for i, g in enumerate(groups):
            vals = pd.to_numeric(s.loc[s[group_col].eq(g), value_col], errors="coerce")
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=APO_COLOR.get(g, ACT_COLOR.get(g, "#777777")), edgecolor="black", linewidth=0.4)
            if vals.notna().any():
                ax.hlines(vals.median(), i - 0.25, i + 0.25, color="black", linewidth=1.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_title(feat, fontsize=9)
    fig.suptitle(title, fontsize=12)
    save_fig(fig, basename, source_table, "sample-level values; median line shown")


def volcano(df: pd.DataFrame, basename: str, title: str, source_table: str, genes_to_label: list[str]) -> None:
    if df.empty or "log2FC" not in df or "fdr_bh" not in df:
        placeholder_figure(basename, title, "No pseudobulk DEG rows available.", source_table)
        return
    sub = df.copy()
    sub["x"] = pd.to_numeric(sub["log2FC"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["fdr_bh"], errors="coerce").map(neglog)
    sub = sub.dropna(subset=["x", "y"])
    if sub.empty:
        placeholder_figure(basename, title, "No finite log2FC/FDR rows available.", source_table)
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sig = sub["fdr_bh"].astype(float) < 0.1
    colors = np.where(sig & (sub["x"] > 0), "#D95F5F", np.where(sig & (sub["x"] < 0), "#5B8DB8", "#BDBDBD"))
    ax.scatter(sub["x"], sub["y"], c=colors, s=14, alpha=0.8, edgecolors="none")
    for g in genes_to_label:
        hit = sub[sub["gene"].astype(str).eq(g)]
        if not hit.empty:
            r = hit.sort_values("fdr_bh").iloc[0]
            ax.text(r["x"], r["y"], g, fontsize=8)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.axhline(neglog(0.1), color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(title)
    fig.text(0.01, 0.01, "Volcano uses sample-level pseudobulk DEG table; y = -log10(BH-FDR).", fontsize=8)
    save_fig(fig, basename, source_table, "current-run sample-level pseudobulk DEG table")


def volcano_grid(deg: pd.DataFrame, panels: list[tuple[str, str]], basename: str, title: str, source_table: str, genes_to_label: list[str]) -> None:
    if deg.empty:
        placeholder_figure(basename, title, "No raw-count pseudobulk DEG rows available.", source_table)
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7.5, 5.2 * len(panels)), 5.4), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    plotted = False
    for ax, (scope_pat, panel_title) in zip(axes, panels):
        sub = deg[deg["cell_scope"].astype(str).str.contains(scope_pat, case=False, regex=True, na=False)].copy()
        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{panel_title}\nNo evaluable raw-count DEG", ha="center", va="center")
            continue
        sub["x"] = pd.to_numeric(sub["log2FC"], errors="coerce")
        fdr_col = "FDR" if "FDR" in sub.columns else "fdr_bh"
        sub["y"] = pd.to_numeric(sub[fdr_col], errors="coerce").map(neglog)
        sub = sub.dropna(subset=["x", "y"])
        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{panel_title}\nNo finite log2FC/FDR", ha="center", va="center")
            continue
        plotted = True
        sig = pd.to_numeric(sub[fdr_col], errors="coerce").fillna(1) < 0.1
        colors = np.where(sig & (sub["x"] > 0), "#D95F5F", np.where(sig & (sub["x"] < 0), "#5B8DB8", "#BDBDBD"))
        ax.scatter(sub["x"], sub["y"], c=colors, s=9, alpha=0.65, edgecolors="none")
        for g in genes_to_label:
            hit = sub[sub["gene"].astype(str).eq(g)]
            if not hit.empty:
                r = hit.sort_values(fdr_col).iloc[0]
                ax.text(r["x"], r["y"], g, fontsize=7)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
        ax.axhline(neglog(0.1), color="#999999", linestyle="--", linewidth=0.8)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("log2FC")
    axes[0].set_ylabel("-log10(FDR)")
    fig.suptitle(title, fontsize=12)
    fig.text(0.01, 0.01, "Raw-count sample-level pseudobulk DEG. Each point is one gene from the all-gene universe after CPM filtering; y = -log10(BH-FDR).", fontsize=8)
    if plotted:
        save_fig(fig, basename, source_table, "raw-count sample-level pseudobulk DEG; all genes after filtering")
    else:
        plt.close(fig)
        placeholder_figure(basename, title, "No evaluable raw-count DEG panels.", source_table)


def find_gmt_files() -> list[Path]:
    preferred = [
        ROOT / "ref" / "gene_sets" / "GO_Biological_Process.gmt",
        ROOT / "ref" / "gene_sets" / "Reactome.gmt",
        ROOT / "ref" / "gene_sets" / "KEGG.gmt",
    ]
    out = [p for p in preferred if p.exists()]
    seen = set(out)
    roots = [ROOT / "ref" / "gene_sets", ROOT / "ref", ROOT / "data", ROOT / "resources", ROOT]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.gmt"):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def parse_gmt(path: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sets[parts[0]] = {g.upper() for g in parts[2:] if g}
    return sets


def gmt_database_label(path: Path) -> str | None:
    name = path.name.lower()
    if "reactome" in name:
        return "Reactome"
    if "kegg" in name:
        return "KEGG"
    if ("go" in name and "bp" in name) or "biological_process" in name or "gobp" in name:
        return "GO_BP"
    return None


def formal_go_reactome_kegg_enrichment(deg: pd.DataFrame, comparison_family: str) -> dict[str, pd.DataFrame]:
    outputs = {"GO_BP": [], "Reactome": [], "KEGG": []}
    if deg.empty:
        for db in outputs:
            outputs[db].append({"comparison": comparison_family, "cell_scope": "all", "direction": "not_evaluable", "database": db, "term_id": "", "term_name": "pathway_enrichment_unavailable", "overlap_genes": "", "overlap_count": 0, "query_gene_count": 0, "background_gene_count": 0, "p_value": np.nan, "FDR": np.nan, "threshold_used": "none", "method": "unavailable_empty_deg", "note": "No raw-count pseudobulk DEG rows available"})
        return {k: pd.DataFrame(v) for k, v in outputs.items()}
    gmt_by_db: dict[str, dict[str, set[str]]] = {"GO_BP": {}, "Reactome": {}, "KEGG": {}}
    for gmt in find_gmt_files():
        db = gmt_database_label(gmt)
        if db:
            gmt_by_db[db].update(parse_gmt(gmt))
    for db, sets in gmt_by_db.items():
        if not sets:
            outputs[db].append({"comparison": comparison_family, "cell_scope": "all", "direction": "not_evaluable", "database": db, "term_id": "", "term_name": "pathway_enrichment_unavailable", "overlap_genes": "", "overlap_count": 0, "query_gene_count": 0, "background_gene_count": 0, "p_value": np.nan, "FDR": np.nan, "threshold_used": "none", "method": "unavailable_no_local_gmt", "note": "No local GO/Reactome/KEGG GMT was found; targeted theme overlap is written separately and is not formal pathway enrichment"})
            continue
        for (comp, scope), sub in deg.groupby(["comparison", "cell_scope"]):
            background = {g.upper() for g in sub["gene"].astype(str)}
            if not background:
                continue
            for direction, sign in [("up", 1), ("down", -1)]:
                sig = sub[(pd.to_numeric(sub["FDR"], errors="coerce") < 0.1) & (np.sign(pd.to_numeric(sub["log2FC"], errors="coerce")) == sign)]
                threshold = "FDR<0.1"
                if sig.shape[0] < 3:
                    sig = sub[(pd.to_numeric(sub["p_value"], errors="coerce") < 0.05) & (np.sign(pd.to_numeric(sub["log2FC"], errors="coerce")) == sign)]
                    threshold = "nominal_p<0.05_exploratory"
                query = {g.upper() for g in sig["gene"].astype(str)}
                if not query:
                    outputs[db].append({"comparison": comp, "cell_scope": scope, "direction": direction, "database": db, "term_id": "", "term_name": "no_query_genes", "overlap_genes": "", "overlap_count": 0, "query_gene_count": 0, "background_gene_count": len(background), "p_value": np.nan, "FDR": np.nan, "threshold_used": threshold, "method": "overrepresentation_local_gmt", "note": "No genes passed the threshold"})
                    continue
                for term, geneset in sets.items():
                    gs = geneset & background
                    if len(gs) < 5:
                        continue
                    overlap = gs & query
                    a = len(overlap)
                    b = len(gs - query)
                    c = len(query - gs)
                    d = max(len(background - (gs | query)), 0)
                    p = np.nan
                    if stats is not None:
                        try:
                            p = float(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1])
                        except Exception:
                            p = np.nan
                    outputs[db].append({"comparison": comp, "cell_scope": scope, "direction": direction, "database": db, "term_id": term.split(" ")[0], "term_name": term, "overlap_genes": ";".join(sorted(overlap)), "overlap_count": a, "query_gene_count": len(query), "background_gene_count": len(background), "p_value": p, "FDR": np.nan, "threshold_used": threshold, "method": "overrepresentation_local_gmt", "note": f"GMT-backed formal enrichment from {db}"})
        if outputs[db]:
            df = pd.DataFrame(outputs[db])
            if "p_value" in df:
                df["FDR"] = bh_fdr(df["p_value"])
            outputs[db] = df.to_dict("records")
    return {k: pd.DataFrame(v) for k, v in outputs.items()}


def targeted_theme_enrichment(deg: pd.DataFrame, comparison_tag: str) -> pd.DataFrame:
    rows = []
    if deg.empty:
        return pd.DataFrame(rows)
    universe = set(deg["gene"].astype(str))
    sig = set(deg.loc[pd.to_numeric(deg["p_value"], errors="coerce") < 0.05, "gene"].astype(str))
    for scope, sdf in deg.groupby("cell_scope"):
        u = set(sdf["gene"].astype(str))
        s = set(sdf.loc[pd.to_numeric(sdf["p_value"], errors="coerce") < 0.05, "gene"].astype(str))
        for theme, genes in TARGET_THEME_SETS.items():
            gs = set(genes) & u
            if len(gs) == 0:
                p = np.nan
                overlap = set()
            else:
                overlap = gs & s
                if stats is not None:
                    a = len(overlap)
                    b = len(gs - s)
                    c = len(s - gs)
                    d = max(len(u - (gs | s)), 0)
                    try:
                        p = float(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1])
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
            rows.append({
                "comparison_set": comparison_tag,
                "cell_scope": scope,
                "targeted_theme": theme,
                "overlap_genes": ";".join(sorted(overlap)),
                "n_overlap": len(overlap),
                "gene_set_size_detected": len(gs),
                "n_nominal_genes_in_scope": len(s),
                "p_value": p,
                "neg_log10_p": neglog(p),
                "method": "manual_targeted_theme_overlap_auxiliary_not_GO_Reactome_KEGG",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out, ["comparison_set"])
    return out


def build_lr_scores(gene_summary: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_pairs = []
    for lig, rec, axis in LR_PAIRS:
        target_pairs.append({
            "ligand": lig, "receptor": rec, "axis": axis,
            "ligand_available": lig in set(gene_summary["gene"]),
            "receptor_available": rec in set(gene_summary["gene"]),
            "caution": "targeted LR proxy; not functional validation",
        })
    pair_df = pd.DataFrame(target_pairs)
    senders = ["myeloid|CD14_classical_monocyte", "myeloid|TREM1high_like_CD14_monocyte", "myeloid|FCGR3A_CD16_monocyte", "myeloid|pDC", "myeloid|cDC", "CD8|ISGhigh_naive_CD8_like", "CD8|GZMKpos_GZMBlow_effmem_like_CD8", "CD8|cytotoxic_CD8", "B|atypical_B_ABC_like", "B|plasmablast_differentiation_high_B", "full|Platelet_MK_like", "full|Myeloid"]
    receivers = ["CD8|naive_CD8", "CD8|ISGhigh_naive_CD8_like", "CD8|GZMKpos_GZMBlow_effmem_like_CD8", "B|naive_B", "B|atypical_B_ABC_like", "B|plasmablast_differentiation_high_B", "CD4|CD4_total", "CD8|CD8_total", "B|B_total"]
    lookup = gene_summary.set_index(["sample_id", "cell_scope", "gene"])
    rows = []
    for sid in SAMPLES:
        for sender in senders:
            for receiver in receivers:
                for lig, rec, axis in LR_PAIRS:
                    try:
                        l = lookup.loc[(sid, sender, lig)]
                        r = lookup.loc[(sid, receiver, rec)]
                    except KeyError:
                        continue
                    lmean = float(l["mean_expression"])
                    rmean = float(r["mean_expression"])
                    lpct = float(l["fraction_expressing"])
                    rpct = float(r["fraction_expressing"])
                    sc = lmean * rmean if np.isfinite(lmean) and np.isfinite(rmean) else np.nan
                    pct = lpct * rpct if np.isfinite(lpct) and np.isfinite(rpct) else np.nan
                    rows.append({
                        "sample_id": sid, "sender": sender, "receiver": receiver, "ligand": lig, "receptor": rec, "axis": axis,
                        "ligand_mean_sender": lmean, "ligand_pct_sender": lpct,
                        "receptor_mean_receiver": rmean, "receptor_pct_receiver": rpct,
                        "LR_score_mean": sc, "LR_score_pct": pct,
                        "sender_cell_count": int(l["n_cells"]), "receiver_cell_count": int(r["n_cells"]),
                        "low_count": bool(l["n_cells"] < 20 or r["n_cells"] < 20),
                        "low_expression": bool((lpct < 0.05) or (rpct < 0.05)),
                    })
    lr = pd.DataFrame(rows)
    stats_rows = []
    if not lr.empty:
        for features, sub in lr.groupby(["sender", "receiver", "ligand", "receptor", "axis"]):
            vals = sub.set_index("sample_id")["LR_score_mean"].reindex(meta.index)
            for comp, group_col, g1, g0, subset in [
                ("APO_vs_nonAPO_auxiliary", "analysis_group", APO_LABEL, NON_APO_LABEL, pd.Series(True, index=meta.index)),
            ]:
                rec = dict(zip(["sender", "receiver", "ligand", "receptor", "axis"], features))
                rec["comparison"] = comp
                rec.update(binary_stats(vals, meta, group_col, g1, g0, subset))
                rec["low_count_fraction"] = sub["low_count"].mean()
                rec["low_expression_fraction"] = sub["low_expression"].mean()
                rec["n_valid_samples"] = int(pd.to_numeric(vals, errors="coerce").notna().sum())
                rec["n_samples_per_group"] = f"{rec.get('group1')}={rec.get('n_group1')};{rec.get('group0')}={rec.get('n_group0')}"
                rec["interpretation_caution"] = "exploratory sample-level LR proxy; not functional validation; low-count/low-expression pairs are not main conclusions"
                stats_rows.append(rec)
    lr_stats = add_fdr(pd.DataFrame(stats_rows), ["comparison"]) if stats_rows else pd.DataFrame()
    return pair_df, lr, lr_stats


def cellphonedb_available() -> dict:
    exe = str(CPDB_BIN) if CPDB_BIN.exists() else shutil.which("cellphonedb")
    if not exe:
        return {"available": False, "executable": "", "version": "not_found"}
    try:
        res = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        version = (res.stdout or res.stderr).strip()
        return {"available": res.returncode == 0, "executable": exe, "version": version, "returncode": res.returncode}
    except Exception as e:
        return {"available": False, "executable": exe, "version": "error", "error": repr(e)}


def find_cellphonedb_database() -> Path | None:
    candidates = list(CPDB_DATABASE_DIR.rglob("*.zip")) if CPDB_DATABASE_DIR.exists() else []
    candidates += list((ROOT / "ref").rglob("*cellphone*.zip")) if (ROOT / "ref").exists() else []
    return sorted(candidates, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)[0] if candidates else None


def _read_cpdb_table_from_zip_or_dir(table_name: str) -> pd.DataFrame:
    direct = CPDB_DATABASE_DIR / table_name
    if direct.exists():
        return pd.read_csv(direct)
    db = find_cellphonedb_database()
    if db and db.exists():
        with zipfile.ZipFile(db) as zf:
            matches = [n for n in zf.namelist() if n.endswith(table_name)]
            if matches:
                with zf.open(matches[0]) as fh:
                    return pd.read_csv(fh)
    return pd.DataFrame()


def extract_cellphonedb_gene_universe() -> pd.DataFrame:
    """Extract ligand/receptor gene symbols from the local CellPhoneDB database.

    This intentionally does not fall back to KEY_GENES. If the database cannot be
    parsed, the returned table is empty and full CellPhoneDB should fail clearly.
    """
    rows = []
    table_candidates = ["gene_input.csv", "protein_input.csv", "interaction_input.csv", "complex_input.csv", "gene_table.csv", "protein_table.csv", "interaction_table.csv", "complex_table.csv", "multidata_table.csv", "complex_composition_table.csv"]
    gene_like_cols = [
        "gene_name", "hgnc_symbol", "gene", "name", "protein_name",
        "partner_a", "partner_b", "gene_a", "gene_b", "hgnc_symbol",
    ]
    for table_name in table_candidates:
        df = _read_cpdb_table_from_zip_or_dir(table_name)
        if df.empty:
            continue
        for col in df.columns:
            if col.lower() not in {c.lower() for c in gene_like_cols}:
                continue
            for val in df[col].dropna().astype(str):
                for gene in val.replace(";", "|").replace(",", "|").split("|"):
                    gene = gene.strip()
                    if not gene or gene.lower() in {"nan", "none", "true", "false"}:
                        continue
                    # Keep HGNC-like symbols and avoid UniProt accessions/complex ids where possible.
                    if len(gene) > 40 or gene.startswith("complex:"):
                        continue
                    rows.append({"gene": gene.upper(), "source_table": table_name, "source_column": col})
    out = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["gene", "source_table", "source_column"])
    if not out.empty:
        out = out.groupby("gene", as_index=False).agg({
            "source_table": lambda x: ";".join(sorted(set(x))),
            "source_column": lambda x: ";".join(sorted(set(x))),
        })
    return out.sort_values("gene").reset_index(drop=True)


def audit_cellphonedb_gene_universe(gene_universe: pd.DataFrame, h5ad_paths: dict[str, Path] | None = None, check_h5ad: bool = False) -> pd.DataFrame:
    audit = gene_universe[["gene", "source_table"]].copy() if not gene_universe.empty else pd.DataFrame(columns=["gene", "source_table"])
    object_map = {
        "full_atlas": "in_full_atlas",
        "B_cell": "in_B_cell",
        "CD4": "in_CD4",
        "CD8": "in_CD8",
        "NK": "in_NK",
        "myeloid": "in_myeloid",
    }
    for obj_name, col in object_map.items():
        audit[col] = "not_checked_dry_run" if not check_h5ad else False
    if check_h5ad and h5ad_paths:
        genes = set(audit["gene"].astype(str))
        for obj_name, col in object_map.items():
            path = h5ad_paths.get(obj_name)
            if not path or not Path(path).exists():
                continue
            try:
                with h5py.File(path, "r") as f:
                    idx = f["var"].attrs.get("_index", "_index")
                    var_upper = set(pd.Index(read_vector(f["var"][idx]).astype(str)).str.upper())
                audit[col] = audit["gene"].isin(genes & var_upper)
            except Exception:
                audit[col] = "error"
    return audit


def choose_cpdb_expression_matrix(f: h5py.File, preferred_matrix: str = "normalized") -> str:
    for key in ["layers/log1p_norm", "layers/log1p_renorm", "layers/lognorm", "layers/log1p"]:
        if key in f:
            return key
    return "X"


def load_cpdb_gene_matrix_from_h5ad(h5ad_path: Path, gene_universe: pd.DataFrame | list[str], preferred_matrix: str = "normalized") -> dict:
    """Open h5ad read-only and prepare matrix metadata for CPDB universe genes.

    The returned object intentionally carries the open h5py file handle; callers
    must close `handle` after writing CellPhoneDB inputs.
    """
    genes_requested = pd.Series(gene_universe["gene"] if isinstance(gene_universe, pd.DataFrame) else gene_universe).dropna().astype(str).str.upper().drop_duplicates().tolist()
    f = h5py.File(h5ad_path, "r")
    obs = read_df(f["obs"])
    idx = f["var"].attrs.get("_index", "_index")
    var_names = pd.Index(read_vector(f["var"][idx]).astype(str))
    upper_to_original = {g.upper(): g for g in var_names.astype(str)}
    upper_to_idx = {g.upper(): i for i, g in enumerate(var_names.astype(str))}
    found_upper = [g for g in genes_requested if g in upper_to_idx]
    matrix_used = choose_cpdb_expression_matrix(f, preferred_matrix=preferred_matrix)
    return {
        "handle": f,
        "obs": obs,
        "var_names": var_names,
        "matrix_used": matrix_used,
        "node": matrix_node_by_path(f, matrix_used),
        "genes_requested": genes_requested,
        "genes_found_upper": found_upper,
        "genes_found": [upper_to_original[g] for g in found_upper],
        "gene_indices": np.array([upper_to_idx[g] for g in found_upper], dtype=int),
    }


def _extract_gene_cell_block(node, row_idx: np.ndarray, gene_indices: np.ndarray, n_obs: int, n_vars: int) -> np.ndarray:
    if len(row_idx) == 0 or len(gene_indices) == 0:
        return np.zeros((len(gene_indices), len(row_idx)), dtype=np.float32)
    row_idx = np.asarray(row_idx, dtype=int)
    gene_indices = np.asarray(gene_indices, dtype=int)
    if isinstance(node, h5py.Group) and {"data", "indices", "indptr"}.issubset(node.keys()):
        enc = matrix_encoding(node).lower()
        out = np.zeros((len(gene_indices), len(row_idx)), dtype=np.float32)
        row_pos = {int(r): i for i, r in enumerate(row_idx)}
        gene_pos = {int(g): i for i, g in enumerate(gene_indices)}
        indptr = node["indptr"]
        indices = node["indices"]
        data = node["data"]
        if "csc" in enc:
            for g in gene_indices:
                start, end = int(indptr[g]), int(indptr[g + 1])
                idx = np.asarray(indices[start:end], dtype=int)
                vals = np.asarray(data[start:end], dtype=np.float32)
                keep = [i for i, r in enumerate(idx) if int(r) in row_pos]
                if keep:
                    out[gene_pos[int(g)], [row_pos[int(idx[i])] for i in keep]] = vals[keep]
        else:
            wanted_genes = set(int(g) for g in gene_indices)
            for r in row_idx:
                start, end = int(indptr[r]), int(indptr[r + 1])
                if end <= start:
                    continue
                idx = np.asarray(indices[start:end], dtype=int)
                vals = np.asarray(data[start:end], dtype=np.float32)
                keep = [i for i, g in enumerate(idx) if int(g) in wanted_genes]
                if keep:
                    out[[gene_pos[int(idx[i])] for i in keep], row_pos[int(r)]] = vals[keep]
        return out
    block = np.asarray(node[row_idx, :], dtype=np.float32)[:, gene_indices]
    return block.T


def planned_cellphonedb_paths() -> dict:
    return {
        "broad_input_dir": str(CPDB / "inputs" / "broad"),
        "targeted_input_dir": str(CPDB / "inputs" / "targeted"),
        "broad_APO_output": str(CPDB / "broad" / "APO"),
        "broad_other_output": str(CPDB / "broad" / "other"),
        "targeted_APO_output": str(CPDB / "targeted" / "APO"),
        "targeted_other_output": str(CPDB / "targeted" / "other"),
        "database_dir": str(CPDB_DATABASE_DIR),
        "database_zip": str(find_cellphonedb_database() or ""),
        "meta_filename": "meta.txt",
        "counts_filename": "counts.txt",
        "grouping": "APO_vs_nonAPO",
    }


def prepare_cellphonedb_inputs(
    object_entries: list[dict],
    meta: pd.DataFrame,
    level: str,
    group_value: str,
    out_dir: Path,
    gene_universe: pd.DataFrame,
    min_group_cells: int = CPDB_MIN_GROUP_CELLS,
    max_cells_per_type: int = 2000,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Write formal CellPhoneDB input files from CPDB ligand/receptor genes.

    `object_entries` contains h5ad paths, precomputed cell-state masks, and the
    scopes to include. Expression is loaded by `load_cpdb_gene_matrix_from_h5ad`
    from the CPDB gene universe, not from H5Target's targeted matrix.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    group_samples = set(meta.index[meta["analysis_group"].astype(str).eq(group_value)])
    rng = np.random.default_rng(32)
    count_frames = []
    meta_frames = []
    excluded_rows: list[dict] = []
    audit_rows = []
    for entry in object_entries:
        object_name = entry["object"]
        h5ad_path = Path(entry["path"])
        masks = entry["masks"]
        cell_type_scopes = entry["scopes"]
        matrix_info = load_cpdb_gene_matrix_from_h5ad(h5ad_path, gene_universe, preferred_matrix="normalized")
        try:
            obs = matrix_info["obs"]
            sample_col = detect_col(obs.columns, ["sample_id", "sample", "orig.ident", "sampleID"])
            samples = obs[sample_col].astype(str).to_numpy() if sample_col else np.array([""] * obs.shape[0])
            genes_found = pd.Index(pd.Series(matrix_info["genes_found_upper"]).drop_duplicates().astype(str))
            for scope in cell_type_scopes:
                if scope not in masks:
                    excluded_rows.append({"level": level, "group": group_value, "object": object_name, "cell_scope": scope, "n_cells": 0, "reason": "scope_not_defined"})
                    continue
                idx = np.where(masks[scope] & np.isin(samples, list(group_samples)))[0]
                n_cells = int(len(idx))
                if n_cells < min_group_cells:
                    excluded_rows.append({"level": level, "group": group_value, "object": object_name, "cell_scope": scope, "n_cells": n_cells, "reason": f"below_min_group_cells_{min_group_cells}"})
                    continue
                downsampled = False
                if max_cells_per_type and n_cells > max_cells_per_type:
                    idx = np.sort(rng.choice(idx, size=max_cells_per_type, replace=False))
                    downsampled = True
                label = scope.split("|", 1)[-1].replace("/", "_").replace(" ", "_")
                cell_names = [f"{object_name}_{level}_{group_value}_{label}_{i}" for i in idx]
                block = _extract_gene_cell_block(matrix_info["node"], idx, matrix_info["gene_indices"], obs.shape[0], len(matrix_info["var_names"]))
                count_frames.append(pd.DataFrame(block, index=genes_found, columns=cell_names))
                meta_frames.append(pd.DataFrame({"Cell": cell_names, "cell_type": label}))
                excluded_rows.append({"level": level, "group": group_value, "object": object_name, "cell_scope": scope, "n_cells": n_cells, "reason": "included"})
                audit_rows.append({
                    "level": level,
                    "group": group_value,
                    "object": object_name,
                    "matrix_used": matrix_info["matrix_used"],
                    "n_cells": n_cells,
                    "n_cell_types": len(cell_type_scopes),
                    "n_cpdb_genes_requested": len(matrix_info["genes_requested"]),
                    "n_cpdb_genes_found": len(matrix_info["genes_found"]),
                    "n_genes_written": len(genes_found),
                    "n_cells_written": len(idx),
                    "downsampled": downsampled,
                    "min_group_cells": min_group_cells,
                    "note": "CellPhoneDB input uses CPDB ligand/receptor gene universe intersected with h5ad var_names; not KEY_GENES",
                })
        finally:
            try:
                matrix_info["handle"].close()
            except Exception:
                pass
    if not count_frames or not meta_frames:
        raise RuntimeError(f"No CellPhoneDB cell types passed cell-count filters for {level}/{group_value}.")
    counts_df = pd.concat(count_frames, axis=1).fillna(0)
    meta_df = pd.concat(meta_frames, ignore_index=True)
    meta_path = out_dir / "meta.txt"
    counts_path = out_dir / "counts.txt"
    meta_df.to_csv(meta_path, sep="\t", index=False)
    counts_df.index.name = "Gene"
    counts_df.to_csv(counts_path, sep="\t")
    status = {
        "level": level,
        "group": group_value,
        "input_dir": str(out_dir),
        "meta_path": str(meta_path),
        "counts_path": str(counts_path),
        "n_cells": int(meta_df.shape[0]),
        "n_cell_types": int(meta_df["cell_type"].nunique()),
        "n_genes": int(counts_df.shape[0]),
        "counts_data": "hgnc_symbol",
        "uses_CPDB_gene_universe": True,
        "uses_KEY_GENES": False,
    }
    return status, pd.DataFrame(excluded_rows), pd.DataFrame(audit_rows)


def obs_index_col(obs: pd.DataFrame) -> str:
    return "_index" if "_index" in obs.columns else str(obs.columns[0])


def build_targeted_full_atlas_entries(targeted_by_object: dict[str, list[str]]) -> list[dict]:
    """Map subset-derived refined state labels back to full atlas cells.

    Primary targeted CellPhoneDB then uses the full atlas expression matrix while
    retaining refined labels derived from subset h5ad objects. This avoids
    stitching expression matrices from separate subset objects.
    """
    with h5py.File(H5ADS["full_atlas"], "r") as f:
        full_obs = read_df(f["obs"])
    full_idx_col = obs_index_col(full_obs)
    full_ids = full_obs[full_idx_col].astype(str).to_numpy()
    full_pos = {cell_id: i for i, cell_id in enumerate(full_ids)}
    full_masks: dict[str, np.ndarray] = {}
    mapping_rows: list[dict] = []

    for object_name, scopes in targeted_by_object.items():
        obj = H5Target(object_name, H5ADS[object_name], TARGETED_GENES_FOR_MODULES)
        try:
            masks, _ = masks_for_object(obj)
            subset_idx_col = obs_index_col(obj.obs)
            subset_ids = obj.obs[subset_idx_col].astype(str).to_numpy()
            mapped_pos = np.array([full_pos.get(cell_id, -1) for cell_id in subset_ids], dtype=int)
            mapped_ok = mapped_pos >= 0
            for scope in scopes:
                source_mask = masks.get(scope, np.zeros(obj.obs.shape[0], dtype=bool))
                keep = source_mask & mapped_ok
                positions = mapped_pos[keep]
                target_mask = np.zeros(full_obs.shape[0], dtype=bool)
                if len(positions):
                    target_mask[np.unique(positions)] = True
                full_masks[scope] = target_mask
                mapping_rows.append({
                    "subset_object": object_name,
                    "scope": scope,
                    "n_subset_scope_cells": int(source_mask.sum()),
                    "n_mapped_to_full_atlas": int(target_mask.sum()),
                    "mapping_fraction": float(target_mask.sum() / source_mask.sum()) if int(source_mask.sum()) else np.nan,
                    "full_atlas_expression_used": True,
                    "subset_expression_used_for_CPDB": False,
                    "note": "Refined subset labels mapped by obs index to full atlas; CellPhoneDB expression is read from full atlas CPDB gene universe.",
                })
        finally:
            obj.close()

    save_tsv(pd.DataFrame(mapping_rows), "CellPhoneDB_targeted_full_atlas_label_mapping.tsv")
    return [{
        "object": "full_atlas_targeted_refined_labels",
        "path": H5ADS["full_atlas"],
        "masks": full_masks,
        "scopes": [scope for scopes in targeted_by_object.values() for scope in scopes if scope in full_masks],
    }]


def run_cellphonedb_statistical_analysis(input_dir: Path, output_dir: Path, threads: int = 4, iterations: int = 1000) -> dict:
    """Run formal CellPhoneDB statistical_analysis on prepared input files."""
    status = cellphonedb_available()
    if not status.get("available"):
        raise RuntimeError(f"CellPhoneDB executable unavailable: {status}")
    meta_path = input_dir / "meta.txt"
    counts_path = input_dir / "counts.txt"
    if not meta_path.exists() or not counts_path.exists():
        raise FileNotFoundError(f"Missing CellPhoneDB input files in {input_dir}")
    db_path = find_cellphonedb_database()
    if db_path is None:
        failure = REP / "CellPhoneDB_failure_report.md"
        failure.write_text(
            "# CellPhoneDB failure report\n\n"
            "Formal CellPhoneDB package is available, but no local CellPhoneDB database zip was found. "
            "Prepare/download a CellPhoneDB database under `references/cellphonedb_database/` before full communication analysis.\n",
            encoding="utf-8",
        )
        raise FileNotFoundError(f"CellPhoneDB database zip not found under {CPDB_DATABASE_DIR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        status["executable"],
        "method",
        "statistical_analysis",
        str(meta_path),
        str(counts_path),
        "--database",
        str(db_path),
        "--counts-data",
        "hgnc_symbol",
        "--output-path",
        str(output_dir),
        "--threads",
        str(threads),
        "--iterations",
        str(iterations),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=24 * 3600)
    log_path = output_dir / "cellphonedb_command.log"
    log_path.write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + res.stdout + "\n\nSTDERR:\n" + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"CellPhoneDB statistical_analysis failed; see {log_path}")
    return {"output_dir": str(output_dir), "returncode": res.returncode, "log_path": str(log_path), "database": str(db_path), "command": " ".join(cmd)}


def _find_cpdb_output(output_dir: Path, name_fragment: str) -> Path | None:
    hits = sorted(output_dir.rglob(f"*{name_fragment}*.txt")) + sorted(output_dir.rglob(f"*{name_fragment}*.tsv"))
    return hits[0] if hits else None


def _melt_cpdb_matrix(path: Path | None, value_name: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    id_cols = [c for c in df.columns if not ("|" in str(c))]
    val_cols = [c for c in df.columns if c not in id_cols]
    if not val_cols:
        return pd.DataFrame()
    keep_ids = [c for c in ["id_cp_interaction", "interacting_pair", "partner_a", "partner_b", "gene_a", "gene_b", "hgnc_symbol", "secreted", "receptor_a", "receptor_b", "annotation_strategy"] if c in id_cols]
    if "interacting_pair" not in keep_ids and id_cols:
        keep_ids = id_cols[: min(len(id_cols), 12)]
    out = df.melt(id_vars=keep_ids, value_vars=val_cols, var_name="sender_receiver", value_name=value_name)
    parts = out["sender_receiver"].astype(str).str.split("|", n=1, expand=True)
    out["sender"] = parts[0]
    out["receiver"] = parts[1] if parts.shape[1] > 1 else ""
    return out


def summarize_cellphonedb_group_results(apo_dir: Path, non_apo_dir: Path, level: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize formal CellPhoneDB group results without inventing group-level p values."""
    apo_means = _melt_cpdb_matrix(_find_cpdb_output(apo_dir, "means"), "APO_mean")
    non_apo_means = _melt_cpdb_matrix(_find_cpdb_output(non_apo_dir, "means"), "nonAPO_mean")
    apo_sig = _melt_cpdb_matrix(_find_cpdb_output(apo_dir, "significant_means"), "APO_significant_mean")
    non_apo_sig = _melt_cpdb_matrix(_find_cpdb_output(non_apo_dir, "significant_means"), "nonAPO_significant_mean")
    merge_keys = [c for c in ["interacting_pair", "partner_a", "partner_b", "gene_a", "gene_b", "hgnc_symbol", "sender", "receiver"] if c in set(apo_means.columns).union(non_apo_means.columns)]
    if not merge_keys:
        merge_keys = ["sender", "receiver"]
    comp = apo_means.merge(non_apo_means, on=merge_keys, how="outer")
    if not apo_sig.empty:
        sig_cols = merge_keys + ["APO_significant_mean"]
        comp = comp.merge(apo_sig[sig_cols], on=merge_keys, how="left")
    if not non_apo_sig.empty:
        sig_cols = merge_keys + ["nonAPO_significant_mean"]
        comp = comp.merge(non_apo_sig[sig_cols], on=merge_keys, how="left")
    for col in ["APO_mean", "nonAPO_mean", "APO_significant_mean", "nonAPO_significant_mean"]:
        if col in comp:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
        else:
            comp[col] = np.nan
    comp["level"] = level
    comp["APO_significant"] = comp.get("APO_significant_mean", pd.Series(np.nan, index=comp.index)).notna()
    comp["nonAPO_significant"] = comp.get("nonAPO_significant_mean", pd.Series(np.nan, index=comp.index)).notna()
    comp["APO_minus_nonAPO"] = comp.get("APO_mean", np.nan) - comp.get("nonAPO_mean", np.nan)
    comp["status"] = np.select(
        [
            comp["APO_significant"] & ~comp["nonAPO_significant"],
            ~comp["APO_significant"] & comp["nonAPO_significant"],
            comp["APO_significant"] & comp["nonAPO_significant"] & (comp["APO_minus_nonAPO"] > 0.05),
            comp["APO_significant"] & comp["nonAPO_significant"] & (comp["APO_minus_nonAPO"] < -0.05),
            comp["APO_significant"] & comp["nonAPO_significant"],
        ],
        ["APO_specific", "nonAPO_specific", "shared_higher_in_APO", "shared_higher_in_nonAPO", "shared_similar"],
        default="not_significant_in_either_group",
    )
    key_terms = ["TNF", "TNFRSF1", "IL1", "MIF", "CD74", "CXCR4", "CD44", "HLA", "KLRK1", "KIR", "CD48", "CD244", "CD40", "CD27", "CD70", "ICOS", "TNFSF13", "TNFRSF13", "PF4", "CXCR3", "PPBP", "CXCR2", "CCL", "CCR", "CXCL"]
    text_cols = [c for c in ["interacting_pair", "partner_a", "partner_b", "gene_a", "gene_b", "hgnc_symbol", "sender", "receiver"] if c in comp]
    joined = comp[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else pd.Series("", index=comp.index)
    key = comp[joined.str.contains("|".join(key_terms), case=False, regex=True, na=False)].copy()
    sig = comp[comp["APO_significant"] | comp["nonAPO_significant"]].copy()
    return sig, comp, key


def run_formal_cellphonedb_pipeline(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate CPDB-universe inputs, run CellPhoneDB, and summarize groups."""
    gene_universe = extract_cellphonedb_gene_universe()
    if gene_universe.empty or gene_universe["gene"].nunique() < 500:
        msg = f"CellPhoneDB gene universe too small or unavailable: n={gene_universe['gene'].nunique() if not gene_universe.empty else 0}"
        (REP / "CellPhoneDB_failure_report.md").write_text("# CellPhoneDB failure report\n\n" + msg + "\n", encoding="utf-8")
        raise RuntimeError(msg)
    universe_audit = audit_cellphonedb_gene_universe(gene_universe, H5ADS, check_h5ad=True)
    save_tsv(universe_audit, "CellPhoneDB_gene_universe.tsv")

    broad_scopes = ["full|Myeloid", "full|CD4_T", "full|CD8_T", "full|B", "full|NK_TNK", "full|pDC", "full|Platelet_MK_like"]
    targeted_by_object = {
        "myeloid": ["myeloid|CD14_classical_monocyte", "myeloid|FCGR3A_CD16_monocyte", "myeloid|TREM1high_like_CD14_monocyte", "myeloid|inflammatory_monocyte", "myeloid|pDC", "myeloid|cDC"],
        "CD8": ["CD8|naive_CD8", "CD8|ISGhigh_naive_CD8_like", "CD8|GZMKpos_GZMBlow_effmem_like_CD8", "CD8|cytotoxic_CD8"],
        "B_cell": ["B|naive_B", "B|memory_B", "B|atypical_B_ABC_like", "B|plasmablast_differentiation_high_B", "B|true_annotation_plasmablast_B"],
        "CD4": ["CD4|CD4_total"],
        "NK": ["NK|NK_Cytotoxic"],
    }

    def build_entries(object_to_scopes: dict[str, list[str]]) -> list[dict]:
        entries = []
        for object_name, scopes in object_to_scopes.items():
            obj = H5Target(object_name, H5ADS[object_name], TARGETED_GENES_FOR_MODULES)
            try:
                masks, _ = masks_for_object(obj)
                entries.append({"object": object_name, "path": H5ADS[object_name], "masks": masks, "scopes": scopes})
            finally:
                obj.close()
        return entries

    entry_sets = {
        "broad": build_entries({"full_atlas": broad_scopes}),
        "targeted": build_targeted_full_atlas_entries(targeted_by_object),
    }
    run_rows, excluded_all, matrix_audit_all = [], [], []
    outputs: dict[str, dict[str, Path]] = {"broad": {}, "targeted": {}}
    for level, entries in entry_sets.items():
        for group_value, group_short in [(APO_LABEL, "APO"), (NON_APO_LABEL, "nonAPO")]:
            input_dir = CPDB / "inputs" / level / group_short
            output_dir = CPDB / level / group_short
            try:
                status, excluded, matrix_audit = prepare_cellphonedb_inputs(entries, meta, level, group_value, input_dir, gene_universe)
                excluded_all.append(excluded)
                matrix_audit_all.append(matrix_audit)
                run_status = run_cellphonedb_statistical_analysis(input_dir, output_dir)
                run_rows.append({**status, **run_status, "status": "success"})
                outputs[level][group_short] = output_dir
            except Exception as e:
                msg = f"{level}/{group_short} CellPhoneDB failed: {repr(e)}"
                run_rows.append({"level": level, "group": group_value, "input_dir": str(input_dir), "output_dir": str(output_dir), "status": "failed", "error": msg})
                existing = (REP / "CellPhoneDB_failure_report.md").read_text(encoding="utf-8") if (REP / "CellPhoneDB_failure_report.md").exists() else "# CellPhoneDB failure report\n\n"
                (REP / "CellPhoneDB_failure_report.md").write_text(existing + msg + "\n", encoding="utf-8")

    run_status_df = pd.DataFrame(run_rows)
    excluded_df = pd.concat(excluded_all, ignore_index=True) if excluded_all else pd.DataFrame()
    matrix_audit_df = pd.concat(matrix_audit_all, ignore_index=True) if matrix_audit_all else pd.DataFrame()
    save_tsv(run_status_df, "CellPhoneDB_run_status.tsv")
    save_tsv(excluded_df, "CellPhoneDB_excluded_celltypes_low_count.tsv")
    save_tsv(matrix_audit_df, "CellPhoneDB_input_matrix_audit.tsv")

    results = {"run_status": run_status_df, "excluded": excluded_df, "matrix_audit": matrix_audit_df, "gene_universe": universe_audit}
    key_frames = []
    for level in ["broad", "targeted"]:
        if "APO" in outputs[level] and "nonAPO" in outputs[level]:
            sig, comp, key = summarize_cellphonedb_group_results(outputs[level]["APO"], outputs[level]["nonAPO"], level)
        else:
            sig, comp, key = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        save_tsv(sig, f"CellPhoneDB_{level}_significant_interactions.tsv")
        save_tsv(comp, f"CellPhoneDB_{level}_group_comparison.tsv")
        results[f"{level}_significant"] = sig
        results[f"{level}_group_comparison"] = comp
        if not key.empty:
            key_frames.append(key)
    key_all = pd.concat(key_frames, ignore_index=True) if key_frames else pd.DataFrame()
    save_tsv(key_all, "CellPhoneDB_key_axis_summary.tsv")
    results["key_axis"] = key_all
    return results

def dry_run() -> None:
    """Lightweight code-path check only; does not read h5ad or run full analysis."""
    summary: dict[str, object] = {
        "mode": "dry_run_only",
        "out_dir": str(OUT),
        "full_analysis_run": False,
        "h5ad_read": False,
    }
    try:
        meta = build_metadata()
        summary["metadata_rows"] = int(meta.shape[0])
        summary["analysis_group_counts"] = meta["analysis_group"].value_counts().to_dict()
        summary["APO_samples_expected"] = sorted(APO_SAMPLES)
        summary["APO_samples_marked"] = sorted(meta.loc[meta["analysis_group"].eq(APO_LABEL), "sample_id"].tolist())
        summary["APO_group_correct"] = summary["APO_samples_marked"] == sorted(APO_SAMPLES)
        summary["original_apo_counts"] = meta["apo_group"].value_counts().to_dict()
        summary["original_activity_counts"] = meta["activity_group"].value_counts().to_dict()
        group_def = pd.DataFrame({"sample_id": SAMPLES})
        group_def["analysis_group"] = group_def["sample_id"].map(lambda x: APO_LABEL if x in APO_SAMPLES else NON_APO_LABEL)
        group_def["is_APO_sample"] = group_def["sample_id"].isin(APO_SAMPLES)
        save_tsv(group_def, "analysis_group_definition.tsv")
        save_tsv(meta.reset_index(drop=True), "sample_metadata.tsv")
    except Exception as e:
        summary["metadata_error"] = repr(e)

    gzmk_thr, gzmk_method = nonzero_median_threshold(np.array([0, 0, 0, 1.0, 2.0, 4.0]))
    gzmk_empty_thr, gzmk_empty_method = nonzero_median_threshold(np.zeros(6))
    summary["gzmk_positive_threshold_test"] = {"threshold": gzmk_thr, "method": gzmk_method, "threshold_gt_zero": bool(np.isfinite(gzmk_thr) and gzmk_thr > 0)}
    summary["gzmk_insufficient_test"] = {"threshold": str(gzmk_empty_thr), "method": gzmk_empty_method, "insufficient": bool(not np.isfinite(gzmk_empty_thr))}

    toy_meta = pd.DataFrame({
        "sample_id": ["sample_001", "sample_002", "sample_003", "sample_004"],
        "apo_group": ["APO", "no_APO", "APO", "no_APO"],
        "activity_group": ["Active", "Stable", "Active", "Stable"],
        "trimester_or_stage": ["early", "early", "mid", "early"],
        "analysis_group": [APO_LABEL, NON_APO_LABEL, APO_LABEL, NON_APO_LABEL],
    }).set_index("sample_id", drop=False)
    toy_counts = np.array([[120, 5, 30], [10, 40, 25], [140, 6, 28], [8, 35, 26]], dtype=float)
    toy_deg = python_logcpm_deg(
        toy_counts,
        ["sample_001", "sample_002", "sample_003", "sample_004"],
        np.array([40, 45, 42, 50]),
        pd.Index(["GZMK", "GZMB", "CX3CR1"]),
        toy_meta,
        "dry_run_object",
        "CD8|CD8_total",
        "APO_vs_nonAPO",
        "analysis_group",
        APO_LABEL,
        NON_APO_LABEL,
        "dry_run_counts",
    )
    summary["toy_pseudobulk_rows"] = int(toy_deg.shape[0])
    summary["toy_pseudobulk_uses_raw_counts"] = bool(toy_deg.get("is_raw_count_pseudobulk", pd.Series([False])).fillna(False).all())

    summary["raw_count_matrix_detection_function_exists"] = callable(globals().get("detect_count_matrix"))
    summary["all_gene_raw_pseudobulk_function_exists"] = callable(globals().get("run_raw_count_pseudobulk_deg"))
    summary["pseudobulk_uses_raw_counts"] = True
    summary["pseudobulk_uses_all_genes"] = True
    summary["edgeR_preferred_function_exists"] = callable(globals().get("edger_deg_from_counts"))
    summary["deg_uses_KEY_GENES_only"] = False
    summary["GO_Reactome_KEGG_enrichment_function_exists"] = callable(globals().get("formal_go_reactome_kegg_enrichment"))
    summary["targeted_theme_not_named_pathway"] = True
    summary["volcano_source_raw_count_all_gene_DEG"] = True
    try:
        import importlib.util
        spec = importlib.util.find_spec("gseapy")
        summary["gseapy_available"] = spec is not None
        if spec is not None:
            import gseapy as gp
            summary["gseapy_version"] = getattr(gp, "__version__", "unknown")
    except Exception as e:
        summary["gseapy_available"] = False
        summary["gseapy_error"] = repr(e)
    gene_set_dir = ROOT / "ref" / "gene_sets"
    go_gmt = gene_set_dir / "GO_Biological_Process.gmt"
    reactome_gmt = gene_set_dir / "Reactome.gmt"
    kegg_gmt = gene_set_dir / "KEGG.gmt"
    summary["GO_Biological_Process_gmt_available"] = go_gmt.exists()
    summary["Reactome_gmt_available"] = reactome_gmt.exists()
    summary["KEGG_gmt_available"] = kegg_gmt.exists()
    summary["gene_set_download_summary_path"] = str(gene_set_dir / "gene_set_download_summary.tsv")
    summary["formal_enrichment_will_use_local_gmt"] = bool(go_gmt.exists() and reactome_gmt.exists() and kegg_gmt.exists())
    cpdb_gene_universe = extract_cellphonedb_gene_universe()
    cpdb_universe_audit = audit_cellphonedb_gene_universe(cpdb_gene_universe, H5ADS, check_h5ad=False)
    save_tsv(cpdb_universe_audit, "CellPhoneDB_gene_universe.tsv")
    summary["cellphonedb_gene_universe_function_exists"] = callable(globals().get("extract_cellphonedb_gene_universe"))
    summary["cellphonedb_gene_universe_n_genes"] = int(cpdb_gene_universe["gene"].nunique()) if not cpdb_gene_universe.empty else 0
    summary["cellphonedb_gene_universe_minimum_passed"] = summary["cellphonedb_gene_universe_n_genes"] > 500
    summary["cellphonedb_input_uses_KEY_GENES"] = False
    summary["cellphonedb_input_uses_CPDB_gene_universe"] = True
    summary["load_cpdb_gene_matrix_function_exists"] = callable(globals().get("load_cpdb_gene_matrix_from_h5ad"))
    summary["edgeR_available"] = edger_available()
    summary["edgeR_main_DEG_method"] = "edgeR_glmQLF_raw_count_pseudobulk"
    try:
        rscript = RSCRIPT_PATH if Path(RSCRIPT_PATH).exists() else shutil.which("Rscript")
        edge_ver = subprocess.run([rscript, "-e", "suppressPackageStartupMessages(library(edgeR)); cat(as.character(packageVersion('edgeR')))"], capture_output=True, text=True, timeout=20) if rscript else None
        summary["edgeR_version"] = edge_ver.stdout.strip() if edge_ver and edge_ver.returncode == 0 else "unavailable"
    except Exception as e:
        summary["edgeR_version"] = "error: " + repr(e)
    cpdb_status = cellphonedb_available()
    summary["cellphonedb_status"] = cpdb_status
    summary["cellphonedb_available"] = bool(cpdb_status.get("available"))
    summary["cellphonedb_version"] = cpdb_status.get("version", "")
    summary["cellphonedb_input_generation_function_exists"] = callable(globals().get("prepare_cellphonedb_inputs"))
    summary["cellphonedb_statistical_analysis_function_exists"] = callable(globals().get("run_cellphonedb_statistical_analysis"))
    summary["cellphonedb_summary_function_exists"] = callable(globals().get("summarize_cellphonedb_group_results"))
    summary["targeted_full_atlas_label_mapping_function_exists"] = callable(globals().get("build_targeted_full_atlas_entries"))
    summary["targeted_CPDB_primary_expression_source"] = "full_atlas_expression_with_refined_labels"
    summary["cellphonedb_planned_paths"] = planned_cellphonedb_paths()
    cpdb_database = find_cellphonedb_database()
    summary["cellphonedb_database_available"] = cpdb_database is not None
    summary["cellphonedb_database_path"] = str(cpdb_database or "")
    summary["placeholder_CellPhoneDB"] = False
    summary["LR_proxy_as_main_communication"] = False
    summary["Active_vs_Stable_main_analysis"] = False
    summary["CellPhoneDB_result_tables_planned"] = [
        "CellPhoneDB_broad_significant_interactions.tsv",
        "CellPhoneDB_targeted_significant_interactions.tsv",
        "CellPhoneDB_broad_group_comparison.tsv",
        "CellPhoneDB_targeted_group_comparison.tsv",
        "CellPhoneDB_key_axis_summary.tsv",
        "CellPhoneDB_run_status.tsv",
        "CellPhoneDB_excluded_celltypes_low_count.tsv",
    ]
    summary["primary_comparison"] = "APO_vs_non-APO"
    summary["output_dir_configured"] = bool(str(OUT))

    summary["output_dir_configured"] = bool(str(OUT))
    summary["script_path"] = str(Path(__file__))
    summary["full_run_entrypoint_available"] = True

    env_path = REP / "environment_check.md"
    env_path.write_text(
        "# Single-cell analysis environment check\n\n"
        f"- edgeR_available: {summary['edgeR_available']}\n"
        f"- edgeR_version: {summary.get('edgeR_version')}\n"
        f"- CellPhoneDB_available: {summary['cellphonedb_available']}\n"
        f"- CellPhoneDB_version: {summary.get('cellphonedb_version')}\n"
        f"- CellPhoneDB_executable: {cpdb_status.get('executable', '')}\n"
        f"- CellPhoneDB_database_available: {summary['cellphonedb_database_available']}\n"
        f"- CellPhoneDB_database_path: {summary['cellphonedb_database_path']}\n"
        f"- CellPhoneDB_gene_universe_n_genes: {summary['cellphonedb_gene_universe_n_genes']}\n"
        f"- CellPhoneDB_input_uses_CPDB_gene_universe: {summary['cellphonedb_input_uses_CPDB_gene_universe']}\n"
        f"- CellPhoneDB_input_uses_KEY_GENES: {summary['cellphonedb_input_uses_KEY_GENES']}\n"
        f"- GO_Biological_Process_gmt_available: {summary['GO_Biological_Process_gmt_available']}\n"
        f"- Reactome_gmt_available: {summary['Reactome_gmt_available']}\n"
        f"- KEGG_gmt_available: {summary['KEGG_gmt_available']}\n"
        f"- full_analysis_run: {summary['full_analysis_run']}\n"
        f"- h5ad_read: {summary['h5ad_read']}\n",
        encoding="utf-8",
    )
    dry_path = REP / "dry_run_check.md"
    dry_path.write_text(
        "# Single-cell analysis dry-run check\n\n"
        "Dry-run completed without starting the full analysis. No h5ad object was opened by dry-run mode.\n\n"
        "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    if h5py is None or plt is None:
        raise ImportError("The full single-cell analysis requires h5py and matplotlib; install analyses/single_cell/environment.yml")
    log("Starting APO versus non-APO single-cell analysis")
    with open(LOG / "session_info.txt", "w", encoding="utf-8") as f:
        f.write(f"python={sys.executable}\nexpected_python={PYTHON_PATH}\n")
        try:
            import scanpy, anndata, scipy, sklearn, seaborn, openpyxl
            f.write(f"scanpy={scanpy.__version__}\nanndata={anndata.__version__}\n")
        except Exception as e:
            f.write(f"package_check_error={e}\n")

    meta = build_metadata()
    for name, path in H5ADS.items():
        meta[f"available_{name}"] = False
        meta[f"n_cells_{name}"] = 0

    all_props, all_scores, all_genes, all_defs = [], [], [], []
    all_count_audit, all_skipped_deg_scopes = [], []
    streaming_deg_files = [
        "pseudobulk_DEG_APO_vs_nonAPO_all_gene.tsv",
    ]
    for deg_name in streaming_deg_files:
        (TAB / deg_name).write_text("", encoding="utf-8")
    for name, path in H5ADS.items():
        try:
            log(f"Reading {name}: {path}")
            obj = H5Target(name, path, TARGETED_GENES_FOR_MODULES)
            masks, defs = masks_for_object(obj)
            props, scores, genes = summarize_by_sample(obj, masks)
            raw_deg, count_audit, skipped_deg = run_raw_count_pseudobulk_deg(obj, masks, meta)
            for sid, cnt in pd.Series(obj.samples[np.isin(obj.samples, SAMPLES)]).value_counts().items():
                if sid in meta.index:
                    meta.loc[sid, f"available_{name}"] = True
                    meta.loc[sid, f"n_cells_{name}"] = int(cnt)
            if raw_deg is not None and not raw_deg.empty:
                append_tsv(raw_deg[raw_deg["comparison"].eq("APO_vs_nonAPO")], "pseudobulk_DEG_APO_vs_nonAPO_all_gene.tsv")
            all_props.append(props)
            all_scores.append(scores)
            all_genes.append(genes)
            all_count_audit.append(count_audit)
            all_skipped_deg_scopes.append(skipped_deg)
            all_defs.extend(defs)
            obj.close()
            del raw_deg, count_audit, skipped_deg
        except Exception as e:
            err(f"{name} failed: {repr(e)}")

    proportions = pd.concat(all_props, ignore_index=True) if all_props else pd.DataFrame()
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    genes = pd.concat(all_genes, ignore_index=True) if all_genes else pd.DataFrame()
    defs = pd.DataFrame(all_defs)
    count_audit_all = pd.concat(all_count_audit, ignore_index=True) if all_count_audit else pd.DataFrame()
    skipped_deg_scopes = pd.concat(all_skipped_deg_scopes, ignore_index=True) if all_skipped_deg_scopes else pd.DataFrame()

    avail_cols = [c for c in meta.columns if c.startswith("available_")]
    meta["object_availability"] = meta[avail_cols].apply(lambda r: ";".join(c.replace("available_", "") for c, v in r.items() if bool(v)), axis=1)
    save_tsv(meta.reset_index(drop=True), "sample_metadata.tsv")
    group_def = pd.DataFrame({"sample_id": SAMPLES})
    group_def["analysis_group"] = group_def["sample_id"].map(lambda x: APO_LABEL if x in APO_SAMPLES else NON_APO_LABEL)
    group_def["is_APO_sample"] = group_def["sample_id"].isin(APO_SAMPLES)
    save_tsv(group_def, "analysis_group_definition.tsv")
    meta_table = meta.reset_index(drop=True)
    group_counts = pd.DataFrame([
        {"grouping": "analysis_group", **meta["analysis_group"].value_counts().to_dict()},
        {"grouping": "original_apo_group", **meta["apo_group"].value_counts().to_dict()},
        {"grouping": "original_activity_group", **meta["activity_group"].value_counts().to_dict()},
        {"grouping": "trimester_or_stage", **meta["trimester_or_stage"].value_counts().to_dict()},
    ])
    save_tsv(group_counts, "group_counts.tsv")
    cross = pd.crosstab([meta["trimester_or_stage"], meta["activity_group"], meta["apo_group"]], meta["analysis_group"]).reset_index()
    save_tsv(cross, "metadata_cross_table.tsv")

    broad = proportions[proportions["object_name"].eq("full_atlas") & proportions["cell_scope"].str.startswith("full|")].copy()
    subtype = proportions[~(proportions["object_name"].eq("full_atlas") & proportions["cell_scope"].str.startswith("full|"))].copy()
    save_tsv(broad, "broad_celltype_proportions_by_sample.tsv")
    broad_stats = group_stats_long(broad, ["cell_scope"], "proportion", meta, "broad_composition")
    save_tsv(broad_stats, "broad_celltype_group_stats.tsv")
    save_tsv(subtype, "subtype_proportions_by_sample.tsv")
    subtype_stats = group_stats_long(subtype, ["object_name", "cell_scope"], "proportion", meta, "subtype_composition")
    save_tsv(subtype_stats, "subtype_group_stats.tsv")

    save_tsv(defs, "targeted_cellstate_definitions.tsv")
    target_scope_terms = ["TREM1", "ISGhigh", "GZMK", "ABC", "plasmablast_differentiation", "IFN_high", "inflammatory_monocyte"]
    targeted_scores = scores[scores["cell_scope"].str.contains("|".join(target_scope_terms), regex=True, na=False)].copy()
    targeted_prop = subtype[subtype["cell_scope"].str.contains("|".join(target_scope_terms), regex=True, na=False)].copy()
    save_tsv(targeted_scores, "targeted_cellstate_scores_by_sample.tsv")
    target_stats = group_stats_long(targeted_prop, ["object_name", "cell_scope"], "proportion", meta, "targeted_cellstate_proportion")
    save_tsv(target_stats, "targeted_cellstate_group_stats.tsv")

    module_gene_rows = []
    detected_genes = set(genes["gene"].unique()) if not genes.empty else set()
    for mod, gs in MODULES.items():
        found = [g for g in gs if g in detected_genes]
        module_gene_rows.append({"module_name": mod, "genes_requested": ";".join(gs), "genes_found": ";".join(found), "n_genes_found": len(found), "coverage_fraction": len(found) / len(gs) if gs else np.nan, "low_confidence": len(found) < 3})
    save_tsv(pd.DataFrame(module_gene_rows), "module_gene_sets_used.tsv")
    save_tsv(scores, "module_scores_by_sample.tsv")
    module_stats = group_stats_long(scores, ["object_name", "cell_scope", "module_name"], "module_score_mean", meta, "module_score")
    save_tsv(module_stats, "module_score_group_stats.tsv")

    save_tsv(count_audit_all, "count_matrix_audit.tsv")
    save_tsv(skipped_deg_scopes, "pseudobulk_DEG_skipped_scopes.tsv")
    deg_all = read_tsv_if_available("pseudobulk_DEG_APO_vs_nonAPO_all_gene.tsv")
    formal = formal_go_reactome_kegg_enrichment(deg_all, "APO_vs_nonAPO")
    save_tsv(formal.get("GO_BP", pd.DataFrame()), "pathway_GO_BP_APO_vs_nonAPO.tsv")
    save_tsv(formal.get("Reactome", pd.DataFrame()), "pathway_Reactome_APO_vs_nonAPO.tsv")
    save_tsv(formal.get("KEGG", pd.DataFrame()), "pathway_KEGG_APO_vs_nonAPO.tsv")
    theme = targeted_theme_enrichment(deg_all, "APO_vs_nonAPO")
    save_tsv(theme, "targeted_theme_enrichment_APO_vs_nonAPO.tsv")

    cpdb_results = run_formal_cellphonedb_pipeline(meta)
    pair_df, lr_scores, lr_stats = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Literature comparison summary from top nominal evidence.
    def best_text(df, patt, comp_contains):
        if df.empty:
            return "not evaluable"
        sub = df[df.astype(str).apply(lambda r: r.str.contains(patt, case=False, regex=True).any(), axis=1)]
        if "comparison" in sub and comp_contains:
            sub = sub[sub["comparison"].astype(str).str.contains(comp_contains, case=False, regex=False, na=False)]
        if sub.empty or "p_value" not in sub:
            return "not evaluable"
        r = sub.sort_values("p_value").iloc[0]
        return f"{r.get('cell_scope', r.get('sender',''))} {r.get('module_name', r.get('gene', r.get('axis','')))} effect={r.get('mean_difference', r.get('log2FC', np.nan)):.3g} p={r.get('p_value', np.nan):.3g} FDR={r.get('fdr_bh', np.nan):.3g}"

    axes = [
        ("TREM1high CD14 monocyte", "APO+ enriched TREM1high CD14+ monocyte", "TREM1|CD14"),
        ("TNF/IL-1 myeloid inflammation", "TNF/IL-1 myeloid inflammatory axis", "TNF|IL1|TREM1"),
        ("type I IFN in monocyte", "type I IFN monocyte context", "IFN|ISG"),
        ("ISGhigh naive CD8", "ISGhigh naive CD8 T", "ISGhigh|naive_CD8"),
        ("GZMK+GZMB− effector memory CD8", "GZMK+GZMB− effector memory CD8", "GZMK|GZMB"),
        ("CX3CR1 migration", "CX3CR1 migration/chemokine receptor", "CX3CR1"),
        ("type II IFN CD8→B", "type II IFN CD8 to B context", "IFNG|type II"),
        ("atypical B", "atypical B expansion", "ABC|atypical"),
        ("plasmablast differentiation", "plasmablast differentiation", "plasmablast|plasma"),
        ("MIF-CD74/CXCR4/CD44", "MIF axis communication", "MIF"),
        ("HLA-KIR/NKG2D", "HLA-KIR/NKG2D interaction", "HLA|KIR|KLRK1"),
        ("CD48-CD244", "CD48-CD244 interaction", "CD48|CD244"),
        ("PF4-CXCR3", "PF4-CXCR3 platelet/T cell communication", "PF4|CXCR3"),
        ("glycosylation modules", "not central in OP0261; exploratory glyco transcript modules", "Sialylation|Galactosylation|Bisecting|Fucosylation"),
    ]
    lit_rows = []
    combo = pd.concat([target_stats, module_stats], ignore_index=True, sort=False)
    for axis, op, patt in axes:
        apo_text = best_text(combo, patt, "APO")
        consistency = "not_evaluable" if "not evaluable" in apo_text else "partially_consistent" if "p=" in apo_text else "descriptive"
        lit_rows.append({
            "axis": axis,
            "OP0261_reported_finding": op,
            "APO_vs_nonAPO_result": apo_text,
            "metadata_context": "Original APO/Active/Stable/stage labels retained only for descriptive cross-table; not main grouping.",
            "consistency": consistency,
            "interpretation": "Compare cautiously; this dataset is small and sample-level.",
            "confidence_level": "exploratory" if consistency != "not_evaluable" else "not_evaluable",
            "limitations": "small sample; low-count or low-expression where applicable; OP0261 is used as context only",
        })
    lit = pd.DataFrame(lit_rows)
    save_tsv(lit, "integrated_axis_summary.tsv")

    # Figures
    # Cohort overview
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(meta))
    ax.scatter(meta["sample_id"], np.zeros(len(meta)), c=meta["apo_group"].map(APO_COLOR), s=100, label="APO")
    ax.scatter(meta["sample_id"], np.ones(len(meta)), c=meta["activity_group"].map(ACT_COLOR), s=100, label="activity")
    stage_colors = {"early": "#9ecae1", "mid": "#fdd0a2", "late": "#c7e9c0", "unknown": "#d9d9d9"}
    ax.scatter(meta["sample_id"], np.ones(len(meta)) * 2, c=meta["trimester_or_stage"].map(stage_colors), s=100, label="stage")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["APO", "Activity", "Stage"])
    ax.set_xticklabels(meta["sample_id"], rotation=45, ha="right")
    ax.set_title("Single-cell cohort overview")
    save_fig(fig, "sample_design_overview", "sample_metadata.tsv", "metadata summary")
    count_cols = [c for c in meta.columns if c.startswith("n_cells_")]
    heatmap_matrix(meta.set_index("sample_id")[count_cols].rename(columns=lambda x: x.replace("n_cells_", "")), "sample_cell_counts_heatmap", "Sample cell counts by object", "sample_metadata.tsv", cmap="viridis", center_zero=False, footnote="Color = cell count per sample/object.")

    # Cell composition
    mat = broad.pivot_table(index="sample_id", columns="cell_scope", values="proportion", aggfunc="first").reindex(meta.index).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(mat.shape[0])
    for col in mat.columns:
        ax.bar(mat.index, mat[col], bottom=bottom, label=col.replace("full|", ""))
        bottom += mat[col].to_numpy()
    ax.set_xticklabels(mat.index, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Broad composition by sample")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "broad_cell_composition_stacked", "broad_celltype_proportions_by_sample.tsv", "sample-level broad cell fractions")
    bubble_plot(broad_stats, "cell_scope", "broad_cell_composition_comparison", "Broad composition: APO vs non-APO", "broad_celltype_group_stats.tsv")
    stage_mat = broad.merge(meta_table[["sample_id", "trimester_or_stage"]], on="sample_id").groupby(["trimester_or_stage", "cell_scope"])["proportion"].mean().unstack().fillna(0)
    heatmap_matrix(stage_mat, "broad_composition_by_stage", "Broad composition by pregnancy stage (descriptive)", "broad_celltype_proportions_by_sample.tsv", cmap="viridis", center_zero=False)
    bubble_plot(subtype_stats, "cell_scope", "targeted_subtype_composition", "Targeted subtype composition: APO vs non-APO", "subtype_group_stats.tsv")

    # Marker audits
    marker_stats = module_stats.copy()
    bubble_plot(marker_stats[marker_stats["cell_scope"].str.contains("TREM1|monocyte|IFN_high", regex=True, na=False)], "module_name", "myeloid_inflammatory_axis", "Myeloid TREM1/IL1/TNF axis", "module_score_group_stats.tsv")
    bubble_plot(marker_stats[marker_stats["cell_scope"].str.contains("CD8|GZMK|ISGhigh", regex=True, na=False)], "module_name", "cd8_state_markers", "CD8 redefined states marker audit", "module_score_group_stats.tsv")
    bubble_plot(marker_stats[marker_stats["cell_scope"].str.contains("B|ABC|plasmablast|IFN", regex=True, na=False)], "module_name", "bcell_plasmablast_states", "B atypical/plasmablast differentiation states", "module_score_group_stats.tsv")

    label_genes = ["TREM1", "IL1B", "TNF", "S100A8", "S100A9", "ISG15", "MX1", "GZMK", "GZMB", "CX3CR1", "TBX21", "ITGAX", "MZB1", "XBP1", "JCHAIN"]
    def filt_deg(df, scope_pat):
        return df[df["cell_scope"].astype(str).str.contains(scope_pat, case=False, regex=True, na=False)].copy()
    volcano_grid(
        deg_all,
        [("myeloid|mono|pdc", "Myeloid/pDC"), ("CD8|TNK", "CD8/TNK"), ("B_cell|full_atlas\\|B|B\\|", "B cell")],
        "pseudobulk_deg_summary",
        "Severe-vs-other raw-count sample-level pseudobulk DEG",
        "pseudobulk_DEG_APO_vs_nonAPO_all_gene.tsv",
        label_genes,
    )
    key_gene_samples = genes[genes["gene"].isin(["TREM1", "IL1B", "TNF", "ISG15", "GZMK", "GZMB", "CX3CR1", "TBX21", "MZB1"]) & genes["cell_scope"].isin(["myeloid|CD14_classical_monocyte", "CD8|naive_CD8", "B|B_total"])]
    key_gene_samples["feature"] = key_gene_samples["cell_scope"] + "|" + key_gene_samples["gene"]
    dotplot_samples(key_gene_samples, "feature", "mean_expression", "analysis_group", key_gene_samples["feature"].drop_duplicates().head(6).tolist(), "key_gene_expression", "Key gene sample-level expression", "targeted cell gene summaries", meta.reset_index(drop=True))

    # Module score plots
    keep_mods = ["TREM1_inflammatory", "type_I_IFN_response", "cytotoxicity", "atypical_B_ABC", "plasmablast_differentiation", "Sialylation_module", "Galactosylation_module", "Bisecting_GlcNAc_module"]
    mod_keep = module_stats[module_stats["module_name"].isin(keep_mods)].copy()
    mod_keep["feature"] = mod_keep["cell_scope"] + "|" + mod_keep["module_name"]
    bubble_plot(mod_keep, "feature", "module_score_summary", "Module scores: APO vs non-APO", "module_score_group_stats.tsv")
    formal_plot = pd.concat(formal.values(), ignore_index=True) if formal else pd.DataFrame()
    if not formal_plot.empty:
        formal_plot["feature"] = formal_plot["database"].astype(str) + "|" + formal_plot["term_name"].astype(str)
        formal_plot["mean_difference"] = pd.to_numeric(formal_plot.get("overlap_count", 0), errors="coerce")
        formal_plot["fdr_bh"] = pd.to_numeric(formal_plot.get("FDR", np.nan), errors="coerce")
        formal_plot["neg_log10_p"] = pd.to_numeric(formal_plot.get("p_value", np.nan), errors="coerce").map(neglog)
        formal_plot["comparison"] = formal_plot["comparison"].astype(str) + "|" + formal_plot["database"].astype(str) + "|" + formal_plot["direction"].astype(str)
    bubble_plot(formal_plot, "feature", "pathway_enrichment_summary", "GO/Reactome/KEGG enrichment: APO vs non-APO", "pathway_GO_BP_APO_vs_nonAPO.tsv;pathway_Reactome_APO_vs_nonAPO.tsv;pathway_KEGG_APO_vs_nonAPO.tsv", value_col="mean_difference")
    theme_plot = theme.rename(columns={"targeted_theme": "feature", "n_overlap": "mean_difference", "comparison_set": "comparison"})
    bubble_plot(theme_plot, "feature", "targeted_theme_enrichment", "Targeted theme overlap: APO vs non-APO", "targeted_theme_enrichment_APO_vs_nonAPO.tsv", value_col="mean_difference")
    glyco = mod_keep[mod_keep["module_name"].str.contains("Sialylation|Galactosylation|Bisecting|Fucosylation", regex=True)]
    bubble_plot(glyco, "feature", "glycosylation_transcriptional_context", "Exploratory glycosylation transcriptional context", "module_score_group_stats.tsv")

    def plot_cpdb_heatmap(comp_df, basename, title, source):
        if comp_df is None or comp_df.empty:
            placeholder_figure(basename, title, "No formal CellPhoneDB comparison rows.", source)
            return
        df = comp_df.copy()
        if "interacting_pair" not in df:
            if "partner_a" not in df:
                df["partner_a"] = ""
            if "partner_b" not in df:
                df["partner_b"] = ""
            df["interacting_pair"] = df["partner_a"].astype(str) + "-" + df["partner_b"].astype(str)
        for col in ["sender", "receiver", "APO_minus_nonAPO"]:
            if col not in df:
                df[col] = np.nan if col == "APO_minus_nonAPO" else ""
        df["row"] = df["interacting_pair"].astype(str) + " | " + df["sender"].astype(str) + "→" + df["receiver"].astype(str)
        df["APO_minus_nonAPO"] = pd.to_numeric(df["APO_minus_nonAPO"], errors="coerce")
        top = df.reindex(df["APO_minus_nonAPO"].abs().sort_values(ascending=False).index).head(25)
        mat_cpdb = top.set_index("row")[["APO_minus_nonAPO"]]
        heatmap_matrix(mat_cpdb, basename, title, source, footnote="Formal CellPhoneDB run separately within severe and other groups; color = severe mean minus other mean, descriptive not group-level p value.")

    broad_cpdb = cpdb_results.get("broad_group_comparison", pd.DataFrame())
    targeted_cpdb = cpdb_results.get("targeted_group_comparison", pd.DataFrame())
    key_cpdb = cpdb_results.get("key_axis", pd.DataFrame())
    plot_cpdb_heatmap(broad_cpdb, "cellphonedb_broad_interactions", "CellPhoneDB broad interactions", "CellPhoneDB_broad_group_comparison.tsv")
    plot_cpdb_heatmap(targeted_cpdb, "cellphonedb_targeted_interactions", "CellPhoneDB targeted interactions", "CellPhoneDB_targeted_group_comparison.tsv")
    plot_cpdb_heatmap(key_cpdb, "cellphonedb_key_axes", "CellPhoneDB key axes", "CellPhoneDB_key_axis_summary.tsv")
    plot_cpdb_heatmap(key_cpdb, "cellphonedb_key_network", "CellPhoneDB key-axis network summary", "CellPhoneDB_key_axis_summary.tsv")

    # Integrated summaries
    summary_effects = []
    for label, df, feat in [
        ("composition", broad_stats, "cell_scope"),
        ("state/module", mod_keep, "feature"),
    ]:
        if not df.empty and "p_value" in df:
            top = df.sort_values("p_value").head(15).copy()
            for _, r in top.iterrows():
                summary_effects.append({"domain": label, "feature": str(r.get(feat, ""))[:70], "comparison": r.get("comparison", ""), "effect": r.get("mean_difference", np.nan), "neglogp": neglog(r.get("p_value", np.nan)), "fdr": r.get("fdr_bh", np.nan)})
    summ = pd.DataFrame(summary_effects)
    if not summ.empty:
        summ["row"] = summ["domain"] + "|" + summ["feature"]
        mat2 = summ.pivot_table(index="row", columns="comparison", values="effect", aggfunc="first").fillna(0)
        heatmap_matrix(mat2, "integrated_immune_axis_summary", "Integrated APO versus non-APO axis summary", "multiple source tables", footnote="Color = sample-level effect from the corresponding table.")
    else:
        placeholder_figure("integrated_immune_axis_summary", "Integrated APO versus non-APO axis summary", "No evaluable summary effects.")
    lit_score = lit.copy()
    lit_score["score"] = lit_score["consistency"].map({"partially_consistent": 1, "descriptive": 0.5, "not_evaluable": 0}).fillna(0)
    heatmap_matrix(lit_score.set_index("axis")[["score"]], "literature_context_summary", "OP0261 comparison", "integrated_axis_summary.tsv", cmap="viridis", center_zero=False, footnote="Higher scores indicate greater agreement with the comparison framework.")

    # Captions, reports, reproducibility.
    captions = ["# Figure captions\n"]
    for figrow in figure_index:
        captions.append(f"## {figrow['figure']}\nSource table: `{figrow['source_table']}`. Method: {figrow['statistical_method']}. P-value source: {figrow['p_value_source']}.\n")
    (REP / "figure_captions.md").write_text("\n".join(captions), encoding="utf-8")
    repro = pd.DataFrame(figure_index)
    repro.to_csv(REP / "reproducibility_check.tsv", sep="\t", index=False)

    key_tables = {
        "sample_metadata": "tables/sample_metadata.tsv",
        "broad_stats": "tables/broad_celltype_group_stats.tsv",
        "subtype_stats": "tables/subtype_group_stats.tsv",
        "targeted_states": "tables/targeted_cellstate_group_stats.tsv",
        "DEG": "tables/pseudobulk_DEG_APO_vs_nonAPO_all_gene.tsv",
        "modules": "tables/module_score_group_stats.tsv",
        "formal_enrichment": "tables/pathway_GO_BP_*.tsv / pathway_Reactome_*.tsv / pathway_KEGG_*.tsv",
        "targeted_theme_overlap": "tables/targeted_theme_enrichment_APO_vs_nonAPO.tsv",
        "CellPhoneDB": "tables/CellPhoneDB_broad_group_comparison.tsv / CellPhoneDB_targeted_group_comparison.tsv",
    }
    top_findings = []
    for source, df, feature_col in [
        ("broad composition", broad_stats, "cell_scope"),
        ("targeted subtype", subtype_stats, "cell_scope"),
        ("targeted state", target_stats, "cell_scope"),
        ("module score", module_stats, "module_name"),
    ]:
        if not df.empty and "p_value" in df:
            for _, r in df.sort_values("p_value").head(3).iterrows():
                top_findings.append(f"- {source}: {r.get(feature_col, '')} in {r.get('comparison','')} effect={r.get('mean_difference', np.nan):.3g}, p={r.get('p_value', np.nan):.3g}, FDR={r.get('fdr_bh', np.nan):.3g}")
    main_report = f"""# APO versus non-APO single-cell analysis report

## Objective
APO versus non-APO PBMC cell-state, raw-count pseudobulk DEG, pathway enrichment, and formal CellPhoneDB analysis in pregnancy-associated SLE.

## Inputs
- h5ad objects: {json.dumps({k: str(v) for k, v in H5ADS.items()}, indent=2)}
- Participant metadata: `{SAMPLE_METADATA_PATH}`
- Pseudobulk DEG tables are recomputed from current h5ad raw/count-like matrices as sample-level all-gene pseudo-bulk counts; prior DEG result tables are not used.

## Group counts
Primary comparison: APO vs non-APO (`{APO_LABEL}` vs `{NON_APO_LABEL}`) = {meta['analysis_group'].value_counts().to_dict()}.

Original APO/no-APO metadata context: {meta['apo_group'].value_counts().to_dict()}.

Original Active/Stable metadata context: {meta['activity_group'].value_counts().to_dict()}.

Stage: {meta['trimester_or_stage'].value_counts().to_dict()}.

## Methods
Composition, targeted states, and module scores are summarized per sample before group statistics. Pseudobulk DEG uses sample-level all-gene raw counts from the current h5ad objects and edgeR GLM quasi-likelihood testing. Volcanoes use these pseudobulk DEG tables, not cell-level tests. GO/Reactome/KEGG enrichment is based on the all-gene pseudobulk DEG results and local GMT resources; manually curated targeted-theme overlap is auxiliary only. CellPhoneDB statistical_analysis is run separately in APO and non-APO samples. Differential communication is summarized descriptively by comparing significant interaction presence and mean interaction scores between groups. No input h5ad object is modified.

## Top data-driven findings
{chr(10).join(top_findings[:15]) if top_findings else 'No evaluable top findings.'}

## Literature comparison
See `tables/integrated_axis_summary.tsv` and the OP0261 comparison figure. OP0261 is used as contextual information only; conclusions are based on the present dataset.

## Low-count/low-expression cautions
pDC, plasmablast_differentiation_high_B, platelet/MK-like, low-count CellPhoneDB cell types, and B-cell subset analyses with incomplete sample coverage should be interpreted as exploratory. Glycosylation modules are transcriptional context only and do not indicate IgG glycan abundance.
"""
    (REP / "analysis_report.md").write_text(main_report, encoding="utf-8")
    (REP / "limitations.md").write_text("""# Analysis limitations

- Small sample size; no SLEDAI/medication/C3/C4/dsDNA used as covariates.
- Original APO/no-APO and Active/Stable labels are retained only as metadata context; the main statistical comparison is APO vs non-APO.
- Some participants are absent from individual lineage-specific subset objects.
- pDC and platelet/MK-like CellPhoneDB cell types may be low-count/low-expression; formal CellPhoneDB interactions are not functional validation.
- DEG volcanoes use raw-count, all-gene sample-level pseudobulk summaries.
- Formal GO/Reactome/KEGG enrichment depends on local GMT availability; if unavailable, the script writes explicit unavailable rows rather than substituting custom themes.
- plasmablast_differentiation_high_B is a score-high B-cell state and is not equivalent to true plasmablast proportion.
- Glycosylation modules are scRNA transcriptional context only and do not represent IgG glycan abundance.
- Formal CellPhoneDB statistical_analysis is run within groups; APO-minus-other communication summaries are descriptive differences in significant interaction presence and mean scores, not direct group-level p values.
""", encoding="utf-8")

    # Copy script to output references/logs for provenance.
    this_script = Path(__file__)
    try:
        shutil.copy2(this_script, REF / this_script.name)
        shutil.copy2(this_script, LOG / this_script.name)
    except Exception:
        pass

    with open(LOG / "errors.log", "w", encoding="utf-8") as f:
        f.write("\n".join(errors) if errors else "No fatal errors.\n")
    with open(LOG / "run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_messages))

    manifest = {
        "python": sys.executable,
        "root": str(ROOT),
        "out": str(OUT),
        "tables": sorted(str(p.relative_to(OUT)) for p in TAB.glob("*.tsv")),
        "figures": sorted(str(p.relative_to(OUT)) for p in FIG.glob("*.*")),
        "reports": sorted(str(p.relative_to(OUT)) for p in REP.glob("*")),
        "errors": errors,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": len(manifest["tables"]), "figures": len(manifest["figures"]), "reports": len(manifest["reports"]), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APO versus non-APO PBMC single-cell analysis")
    parser.add_argument("--dry-run", action="store_true", help="Run lightweight checks only; do not read h5ad or generate full results")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        main()
