# feature_annotations.py
# ---------------------------------------------------------------------
# Feature annotation for SLEmodel
# - RNA: ENSG -> HGNC symbol mapping
# - Mass: m/z -> metabolite name via exact mass matching
#         with endogenous-first priority strategy
# - Gly: pass-through (pre-annotated)
# ---------------------------------------------------------------------

from __future__ import annotations
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===== Configuration =====
GENE_MAP_CSV = Path("ref/ensembl_to_hgnc_GRCh38.csv")
METABOLITE_XLSX = Path("ref/serum_metabolite.xlsx")
EXOGENOUS_BLACKLIST_FILE = Path("ref/mass_exogenous_blacklist.txt")

# Mass-matching parameters for the study platform
# Platform: Bruker MALDI-TOF/TOF, positive ion, reflector mode
# Primary adducts: [M+Na]+ and [M+2Na-H]+
PROTON_MASS = 1.007276466812
NA_MASS = 22.989218
K_MASS = 38.963158
NH4_MASS = 18.033823

# Adducts with platform-aligned priority
# Lower priority number = higher preference
ADDUCTS = [
    {"name": "[M+Na]+",    "delta": NA_MASS,                    "priority": 1},
    {"name": "[M+2Na-H]+", "delta": 2 * NA_MASS - PROTON_MASS,  "priority": 2},
    {"name": "[M+K]+",     "delta": K_MASS,                     "priority": 3},
    {"name": "[M+H]+",     "delta": PROTON_MASS,                "priority": 4},
    {"name": "[M+NH4]+",   "delta": NH4_MASS,                   "priority": 5},
]

# Peak clustering used an 80-ppm alignment tolerance; annotation uses 60 ppm.
DEFAULT_MATCH_PPM = 60.0

# Priority penalty for exogenous compounds (added to adduct priority)
EXOGENOUS_PRIORITY_PENALTY = 100

# ===== Globals (Lazy Loading) =====
_ENSG2SYM: Optional[Dict[str, str]] = None
_SERUM_DB: Optional[List[Dict]] = None
_EXOGENOUS_SET: Optional[Set[str]] = None

# ----------------- Loaders -----------------

def _load_gene_map():
    """Load ENSG to HGNC mapping."""
    global _ENSG2SYM
    if _ENSG2SYM is not None:
        return

    _ENSG2SYM = {}
    if not GENE_MAP_CSV.exists():
        logger.warning(f"Gene map file not found: {GENE_MAP_CSV}. RNA features will use raw IDs.")
        return

    try:
        df = pd.read_csv(GENE_MAP_CSV)
        id_col = next((c for c in df.columns if "ensembl" in c.lower()), None)
        sym_col = next((c for c in df.columns if "hgnc" in c.lower() or "symbol" in c.lower()), None)

        if id_col and sym_col:
            df[id_col] = df[id_col].astype(str).str.split('.').str[0]
            _ENSG2SYM = dict(zip(df[id_col], df[sym_col]))
            logger.info(f"Loaded {len(_ENSG2SYM)} gene mappings from {GENE_MAP_CSV}")
    except Exception as e:
        logger.error(f"Failed to load gene map: {e}")


def _load_serum_db():
    """Load metabolite database from partner's xlsx (serum_metabolite.xlsx)."""
    global _SERUM_DB
    if _SERUM_DB is not None:
        return

    _SERUM_DB = []
    if not METABOLITE_XLSX.exists():
        logger.warning(f"Metabolite DB not found: {METABOLITE_XLSX}. Mass features will be unannotated.")
        return

    try:
        df = pd.read_excel(METABOLITE_XLSX)

        # Identify columns (robust to namespace prefix like 'ns1:')
        cols = {c.split(':')[-1].lower().replace('_', '').replace(' ', ''): c for c in df.columns}

        # Find name column
        name_col = None
        for cand in ['name', 'metabolitename', 'compoundname']:
            if cand in cols:
                name_col = cols[cand]
                break

        # Find monoisotopic mass column
        mass_col = None
        for cand in ['monisotopicmolecularweight', 'monoisotopicmolecularweight',
                     'monoisotopicmass', 'exactmass', 'monomw']:
            if cand in cols:
                mass_col = cols[cand]
                break

        if not name_col or not mass_col:
            logger.warning(f"Could not find name/mass columns in {METABOLITE_XLSX}. "
                          f"Available (normalized): {list(cols.keys())}")
            return

        count = 0
        for _, row in df.iterrows():
            name = row[name_col]
            mass = row[mass_col]

            if pd.isna(name) or pd.isna(mass):
                continue
            try:
                mass_val = float(mass)
                if mass_val > 0:
                    _SERUM_DB.append({
                        "name": str(name).strip(),
                        "neutral_mass": mass_val
                    })
                    count += 1
            except (ValueError, TypeError):
                continue

        logger.info(f"Loaded {count} metabolites from serum DB ({METABOLITE_XLSX})")

    except Exception as e:
        logger.error(f"Failed to load metabolite DB: {e}")


def _load_exogenous_blacklist():
    """Load exogenous compound blacklist for deprioritization."""
    global _EXOGENOUS_SET
    if _EXOGENOUS_SET is not None:
        return

    _EXOGENOUS_SET = set()

    # Default built-in blacklist (always applied)
    default_exogenous = {
        "mephentermine", "fosfomycin", "ethambutol", "fluorouracil",
        "metformin", "chloramphenicol", "isometheptene", "ampicillin",
        "ibuprofen", "acetaminophen", "paracetamol", "aspirin",
        "allyl isothiocyanate", "bovinocidin", "m-xylene", "o-xylene", "p-xylene",
        "chrysoeriol", "luteolin", "apigenin", "kaempferol", "quercetin",
        "catechin", "epicatechin", "resveratrol", "curcumin",
        "hydroxychloroquine", "azathioprine", "mycophenolate", "cyclosporine",
        "tacrolimus", "prednisone", "prednisolone", "dexamethasone"
    }
    _EXOGENOUS_SET.update(default_exogenous)

    # Load from file if exists
    if EXOGENOUS_BLACKLIST_FILE.exists():
        try:
            with open(EXOGENOUS_BLACKLIST_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    _EXOGENOUS_SET.add(line.lower().strip())
            logger.info(f"Loaded {len(_EXOGENOUS_SET)} exogenous compounds from blacklist")
        except Exception as e:
            logger.warning(f"Failed to load exogenous blacklist: {e}")


def _is_exogenous(name: str) -> bool:
    """Check if a compound name is in the exogenous blacklist."""
    _load_exogenous_blacklist()
    return name.lower().strip() in _EXOGENOUS_SET


def _ensure_maps_loaded():
    _load_gene_map()
    _load_serum_db()
    _load_exogenous_blacklist()


# ----------------- Mass Matching Logic -----------------

def _parse_mz_from_feature(feature_name: str) -> Optional[float]:
    """
    Parse observed m/z from feature name.
    Supports formats:
      - "mass::mz_123.45678"
      - "mz_123.45678"
      - "mz_bin_123.45"
    """
    try:
        # Remove modality prefix if present
        raw = feature_name.split('::')[-1] if '::' in feature_name else feature_name

        # Extract numeric part after 'mz_' or 'mz_bin_'
        if raw.startswith('mz_bin_'):
            num_str = raw[7:]
        elif raw.startswith('mz_'):
            num_str = raw[3:]
        else:
            num_str = raw

        return float(num_str)
    except (ValueError, IndexError):
        return None


def _match_mass_feature(feature_name: str, match_ppm: float = DEFAULT_MATCH_PPM) -> Tuple[str, str, bool]:
    """
    Match a mass feature to the serum DB using exact mass + adduct search.

    Strategy: Endogenous-first priority
    - Candidates are sorted by: (is_exogenous, adduct_priority, ppm_error)
    - Endogenous metabolites are always preferred over exogenous ones
    - Among same category, prefer platform-validated adducts ([M+Na]+ > [M+2Na-H]+)

    Returns: (display_label, hover_detail, is_annotated)
    """
    mz_obs = _parse_mz_from_feature(feature_name)
    if mz_obs is None:
        return feature_name, "Parse Error", False

    candidates = []

    if _SERUM_DB:
        for met in _SERUM_DB:
            neutral_mass = met['neutral_mass']
            name = met['name']

            for adduct in ADDUCTS:
                mz_theoretical = neutral_mass + adduct['delta']

                if mz_theoretical <= 0:
                    continue

                ppm_error = abs(mz_obs - mz_theoretical) / mz_theoretical * 1e6

                if ppm_error <= match_ppm:
                    is_exo = _is_exogenous(name)
                    # Effective priority: exogenous gets penalty
                    effective_priority = adduct['priority'] + (EXOGENOUS_PRIORITY_PENALTY if is_exo else 0)

                    candidates.append({
                        "name": name,
                        "adduct": adduct['name'],
                        "ppm": ppm_error,
                        "adduct_priority": adduct['priority'],
                        "effective_priority": effective_priority,
                        "theo_mz": mz_theoretical,
                        "neutral_mass": neutral_mass,
                        "is_exogenous": is_exo
                    })

    if candidates:
        # Sort by: effective_priority (endogenous first), then ppm error
        candidates.sort(key=lambda x: (x['effective_priority'], x['ppm']))
        best = candidates[0]

        # Separate endogenous and exogenous candidates for reporting
        endo_candidates = [c for c in candidates if not c['is_exogenous']]
        exo_candidates = [c for c in candidates if c['is_exogenous']]

        # Display label: metabolite name
        display_label = best['name']

        # Build hover detail with full transparency
        hover_lines = [
            f"{'[EXOGENOUS] ' if best['is_exogenous'] else ''}Annotation: {best['name']}",
            f"Adduct: {best['adduct']}",
            f"Obs m/z: {mz_obs:.4f}",
            f"Theo m/z: {best['theo_mz']:.4f}",
            f"Error: {best['ppm']:.1f} ppm",
            f"Neutral mass: {best['neutral_mass']:.4f}"
        ]

        # Report alternative endogenous matches
        if len(endo_candidates) > 1:
            hover_lines.append("--- Other endogenous matches ---")
            for alt in endo_candidates[1:min(4, len(endo_candidates))]:
                hover_lines.append(f"  {alt['name']} {alt['adduct']} ({alt['ppm']:.1f} ppm)")

        # Report exogenous matches (if best is endogenous, show what was deprioritized)
        if exo_candidates and not best['is_exogenous']:
            hover_lines.append("--- Deprioritized exogenous matches ---")
            for alt in exo_candidates[:3]:
                hover_lines.append(f"  {alt['name']} {alt['adduct']} ({alt['ppm']:.1f} ppm)")
        elif exo_candidates and best['is_exogenous'] and len(exo_candidates) > 1:
            hover_lines.append("--- Other exogenous matches ---")
            for alt in exo_candidates[1:3]:
                hover_lines.append(f"  {alt['name']} {alt['adduct']} ({alt['ppm']:.1f} ppm)")

        hover_detail = "\n".join(hover_lines)

        # Mark as annotated even if exogenous (but display will show [EXOGENOUS] tag)
        return display_label, hover_detail, True

    else:
        display_label = f"m/z {mz_obs:.4f}"
        hover_detail = f"Unannotated m/z {mz_obs:.4f}\n(No match within {match_ppm:.0f} ppm in Serum DB)"
        return display_label, hover_detail, False


# ----------------- Public API -----------------

def get_display_labels(full_feature_name: str,
                       feature_names_optional: List[str] = None,
                       top_mass_display: int = None,
                       match_ppm: float = DEFAULT_MATCH_PPM) -> Tuple[str, str, bool]:
    """
    Main entry point for feature annotation.

    Args:
        full_feature_name: e.g., "rna::ENSG0001", "mass::mz_100.01", "gly::GP1"
        match_ppm: PPM tolerance for mass matching (default 60)

    Returns:
        (Display Name, Hover Text, Is Annotated Boolean)
    """
    _ensure_maps_loaded()

    # Parse modality prefix
    if "::" in full_feature_name:
        modality, raw_feat = full_feature_name.split("::", 1)
    else:
        if full_feature_name.startswith("mz_"):
            modality, raw_feat = "mass", full_feature_name
        elif full_feature_name.startswith("ENS"):
            modality, raw_feat = "rna", full_feature_name
        elif full_feature_name.startswith("GP") or full_feature_name.startswith("DG"):
            modality, raw_feat = "gly", full_feature_name
        else:
            modality, raw_feat = "unknown", full_feature_name

    # --- RNA Logic ---
    if modality == "rna":
        ensg_clean = raw_feat.split('.')[0]
        symbol = _ENSG2SYM.get(ensg_clean, raw_feat) if _ENSG2SYM else raw_feat

        display = symbol
        hover = f"{symbol}\n(Ensembl: {raw_feat})"
        is_ann = (symbol != raw_feat)
        return display, hover, is_ann

    # --- Mass Logic ---
    if modality == "mass":
        return _match_mass_feature(full_feature_name, match_ppm=match_ppm)

    # --- Glycan Logic ---
    if modality == "gly":
        return raw_feat, f"Glycan: {raw_feat}", True

    # --- Default ---
    return raw_feat, raw_feat, False


def build_feature_label_maps(full_feature_names: List[str],
                             match_ppm: float = DEFAULT_MATCH_PPM,
                             top_mass_display: int = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, bool]]:
    """
    Batch process a list of feature names.

    Returns:
        display_map: feature_name -> display label
        hover_map: feature_name -> hover text
        annotated_map: feature_name -> is_annotated flag
    """
    _ensure_maps_loaded()

    disp, hov, ann = {}, {}, {}

    for name in full_feature_names:
        d, h, a = get_display_labels(name, match_ppm=match_ppm, top_mass_display=top_mass_display)
        disp[name] = d
        hov[name] = h
        ann[name] = a

    return disp, hov, ann


def is_exogenous_annotation(display_name: str) -> bool:
    """
    Check if a display name corresponds to an exogenous compound.
    Useful for downstream filtering in visualization.
    """
    _load_exogenous_blacklist()
    return display_name.lower().strip() in _EXOGENOUS_SET


def add_display_labels_to_df(df: pd.DataFrame,
                             feature_col: str = "feature",
                             modality_col: str = "modality",
                             match_ppm: float = DEFAULT_MATCH_PPM) -> pd.DataFrame:
    """
    Helper to add annotation columns to a DataFrame.
    """
    df_out = df.copy()

    unique_feats = df_out.apply(
        lambda row: f"{row[modality_col]}::{row[feature_col]}", axis=1
    ).unique()

    disp_map, hov_map, ann_map = build_feature_label_maps(list(unique_feats), match_ppm=match_ppm)

    def _apply(row):
        key = f"{row[modality_col]}::{row[feature_col]}"
        return pd.Series([disp_map.get(key), hov_map.get(key), ann_map.get(key)])

    df_out[['display_label', 'hover_label', 'is_annotated']] = df_out.apply(_apply, axis=1)

    # Add exogenous flag
    df_out['is_exogenous'] = df_out['display_label'].apply(
        lambda x: is_exogenous_annotation(x) if pd.notna(x) else False
    )

    return df_out


# ----------------- Batch Annotation for Enrichment -----------------

def annotate_mass_features_for_enrichment(
    feature_names: List[str],
    match_ppm: float = DEFAULT_MATCH_PPM,
    allowed_adducts: Optional[List[str]] = None,
    exclude_exogenous: bool = False
) -> pd.DataFrame:
    """
    Generate a detailed annotation table for mass features (for KEGG enrichment pipeline).

    Args:
        feature_names: List of feature names (e.g., ["mass::mz_123.456", ...])
        match_ppm: PPM tolerance
        allowed_adducts: List of allowed adduct forms (default: all)
        exclude_exogenous: If True, exclude exogenous compounds from output

    Returns:
        DataFrame with columns: feature, candidate, adduct, ppm, neutral_mass,
                               is_annotated, is_exogenous, effective_priority
    """
    _ensure_maps_loaded()

    if allowed_adducts is None:
        allowed_set = {a['name'].upper() for a in ADDUCTS}
    else:
        allowed_set = {a.strip().upper() for a in allowed_adducts}

    rows = []

    for feat in feature_names:
        mod = feat.split("::")[0] if "::" in feat else "?"
        if mod != "mass":
            continue

        mz_obs = _parse_mz_from_feature(feat)
        if mz_obs is None:
            continue

        candidates_for_feat = []

        if _SERUM_DB:
            for met in _SERUM_DB:
                neutral_mass = met['neutral_mass']
                name = met['name']

                for adduct in ADDUCTS:
                    if adduct['name'].upper() not in allowed_set:
                        continue

                    mz_theoretical = neutral_mass + adduct['delta']
                    if mz_theoretical <= 0:
                        continue

                    ppm_error = abs(mz_obs - mz_theoretical) / mz_theoretical * 1e6

                    if ppm_error <= match_ppm:
                        is_exo = _is_exogenous(name)

                        if exclude_exogenous and is_exo:
                            continue

                        effective_priority = adduct['priority'] + (EXOGENOUS_PRIORITY_PENALTY if is_exo else 0)

                        candidates_for_feat.append({
                            "feature": feat,
                            "candidate": name,
                            "adduct": adduct['name'],
                            "ppm": ppm_error,
                            "adduct_priority": adduct['priority'],
                            "effective_priority": effective_priority,
                            "neutral_mass": neutral_mass,
                            "theo_mz": mz_theoretical,
                            "obs_mz": mz_obs,
                            "is_exogenous": is_exo
                        })

        if candidates_for_feat:
            # Sort by effective priority
            candidates_for_feat.sort(key=lambda x: (x['effective_priority'], x['ppm']))

            # Normalize weights: endogenous candidates get higher weight
            endo_count = sum(1 for c in candidates_for_feat if not c['is_exogenous'])
            exo_count = len(candidates_for_feat) - endo_count

            for c in candidates_for_feat:
                if not c['is_exogenous']:
                    # Endogenous: equal weight among endogenous
                    c['weight_norm'] = 1.0 / max(endo_count, 1)
                else:
                    # Exogenous: much lower weight (only used if no endogenous)
                    if endo_count > 0:
                        c['weight_norm'] = 0.01 / max(exo_count, 1)  # Minimal weight
                    else:
                        c['weight_norm'] = 1.0 / max(exo_count, 1)  # Normal weight if only exogenous

                c['is_annotated'] = True
                rows.append(c)
        else:
            # No match
            rows.append({
                "feature": feat,
                "candidate": f"m/z {mz_obs:.4f}",
                "adduct": "",
                "ppm": float('nan'),
                "adduct_priority": 999,
                "effective_priority": 999,
                "neutral_mass": float('nan'),
                "theo_mz": float('nan'),
                "obs_mz": mz_obs,
                "is_exogenous": False,
                "weight_norm": 1.0,
                "is_annotated": False
            })

    return pd.DataFrame(rows)


# ----------------- Utility for downstream analysis -----------------

def get_annotation_summary(feature_names: List[str], match_ppm: float = DEFAULT_MATCH_PPM) -> Dict:
    """
    Generate annotation summary statistics for reporting.
    """
    _ensure_maps_loaded()

    stats = {
        "total_mass_features": 0,
        "annotated_endogenous": 0,
        "annotated_exogenous_only": 0,
        "unannotated": 0,
        "total_rna_features": 0,
        "rna_with_symbol": 0,
        "total_gly_features": 0
    }

    for feat in feature_names:
        mod = feat.split("::")[0] if "::" in feat else "?"

        if mod == "mass":
            stats["total_mass_features"] += 1
            disp, hover, is_ann = get_display_labels(feat, match_ppm=match_ppm)

            if is_ann:
                if _is_exogenous(disp):
                    stats["annotated_exogenous_only"] += 1
                else:
                    stats["annotated_endogenous"] += 1
            else:
                stats["unannotated"] += 1

        elif mod == "rna":
            stats["total_rna_features"] += 1
            disp, _, is_ann = get_display_labels(feat)
            if is_ann:
                stats["rna_with_symbol"] += 1

        elif mod == "gly":
            stats["total_gly_features"] += 1

    return stats
