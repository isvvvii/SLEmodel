# met_name_resolver.py
# ---------------------------------------------------------------------
# Resolve mass feature candidate names → canonical metabolite keys
# using a synonym dictionary built from serum_metabolite.xlsx,
# with fuzzy matching + ppm/adduct evidence weighting + full audit.
# ---------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import re
import math
import warnings

import pandas as pd
import numpy as np

# Optional fuzzy matcher
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAVE_RF = True
except Exception:
    import difflib
    _HAVE_RF = False

# ---------------------- Normalization utils ---------------------- #
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega"
}

_SALT_TOKENS = {
    "sodium","potassium","calcium","magnesium","ammonium",
    "hydrochloride","nitrate","sulfate","sulphate","phosphate","chloride",
    "acetate","tartrate","maleate","mesylate","tosylate","citrate"
}
_HYDRATE_TOKENS = {"hydrate","monohydrate","dihydrate","trihydrate"}

def _replace_greek(s: str) -> str:
    for k,v in _GREEK.items():
        s = s.replace(k, v)
    return s

def normalize_name(s: str) -> str:
    """Lowercase, greek→latin, strip punctuation, collapse spaces; remove obvious salt/hydrate tokens."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = _replace_greek(s)
    # keep alnum and spaces
    s = re.sub(r"[^\w\s\-\/\(\)\[\]\+]", " ", s)
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # remove salt/hydrate trailing tokens conservatively
    toks = [t for t in s.split() if t not in _SALT_TOKENS and t not in _HYDRATE_TOKENS]
    return " ".join(toks)

def base_name(s: str) -> str:
    """A 'softer' normalization: drop stereochemistry prefixes D-/L- and (R)/(S)."""
    s = normalize_name(s)
    s = re.sub(r"(^|\s)[dl]-", " ", s).strip()
    s = re.sub(r"\((r|s)\)", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_key(s: str) -> str:
    """Token-set key: sorted unique tokens for robust exact-ish matches."""
    toks = sorted(set(base_name(s).split()))
    return " ".join(toks)

# ---------------------- Description synonym harvest ---------------------- #
_desc_patterns = [
    r"also known as ([^.;]+)",
    r"aka ([^.;]+)",
    r"also called ([^.;]+)",
    r"or ([^.;]+)"
]

def harvest_synonyms_from_desc(desc: str) -> List[str]:
    out: List[str] = []
    if not isinstance(desc, str) or not desc.strip():
        return out
    low = desc.strip()
    for pat in _desc_patterns:
        for m in re.finditer(pat, low, flags=re.IGNORECASE):
            chunk = m.group(1)
            parts = re.split(r",| or ", chunk)
            for p in parts:
                p = p.strip().strip(".;")
                if len(p) >= 2 and not p.isdigit():
                    out.append(p)
    seen = set()
    dedup = []
    for x in out:
        k = normalize_name(x)
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup

# ---------------------- Data classes ---------------------- #
@dataclass
class CanonicalAttrs:
    display_name: str
    mono_mass: Optional[float] = None
    formula: Optional[str] = None
    synonyms_norm: Optional[List[str]] = None

@dataclass
class ResolveParams:
    fuzzy_strict: int = 95
    fuzzy_loose: int = 80
    ppm_tau: float = 7.0
    adduct_weights: Dict[str, float] = None
    allow_soft_assign: bool = False

    def __post_init__(self):
        if self.adduct_weights is None:
            self.adduct_weights = {
                "[m+h]+": 1.0, "[m-h]-": 1.0,
                "[m+na]+": 0.9, "[m+k]+": 0.7
            }

# ---------------------- Synonym dictionary ---------------------- #
def load_met_synonyms(xlsx_path: Path,
                      overrides_csv: Optional[Path] = None,
                      blacklist_csv: Optional[Path] = None
                      ) -> Tuple[Dict[str, str], Dict[str, CanonicalAttrs], set]:
    """
    Returns:
      syn2cano: synonym_norm -> canonical_key
      cano_attrs: canonical_key -> CanonicalAttrs
      blacklist: set of synonym_norm to ignore
    """
    syn2cano: Dict[str, str] = {}
    cano_attrs: Dict[str, CanonicalAttrs] = {}
    blacklist: set = set()

    if blacklist_csv and Path(blacklist_csv).exists():
        try:
            bl = pd.read_csv(blacklist_csv)
            for x in bl.iloc[:,0].astype(str).tolist():
                blacklist.add(normalize_name(x))
        except Exception as e:
            warnings.warn(f"Failed to load blacklist: {e}")

    if not Path(xlsx_path).exists():
        warnings.warn(f"Serum synonym xlsx not found: {xlsx_path}. Proceeding without synonym expansion.")
        return syn2cano, cano_attrs, blacklist

    df = pd.read_excel(xlsx_path)
    col_name = next((c for c in df.columns if str(c).endswith("name")), df.columns[0])
    col_desc = next((c for c in df.columns if "cs_description" in str(c)), None)
    col_iupac = next((c for c in df.columns if "iupac_name" in str(c)), None)
    col_trad  = next((c for c in df.columns if "traditional_iupac" in str(c)), None)
    col_mass  = next((c for c in df.columns if "monisotopic" in str(c).lower()), None)
    col_formula = next((c for c in df.columns if "chemical_formula" in str(c)), None)

    for _, row in df.iterrows():
        raw = str(row.get(col_name, "")).strip()
        if not raw:
            continue
        disp = raw
        mono = None
        try:
            mono = float(row.get(col_mass)) if col_mass else None
        except Exception:
            mono = None
        formula = str(row.get(col_formula)) if col_formula else None

        cano = base_name(disp)
        syns = set()
        syns.add(disp)
        if col_trad and isinstance(row.get(col_trad), str):
            syns.add(row.get(col_trad))
        if col_iupac and isinstance(row.get(col_iupac), str):
            syns.add(row.get(col_iupac))
        if col_desc and isinstance(row.get(col_desc), str):
            for s in harvest_synonyms_from_desc(row.get(col_desc)):
                syns.add(s)

        syn_norms = [normalize_name(x) for x in syns if isinstance(x, str) and x.strip()]
        syn_norms = [s for s in syn_norms if s and s not in blacklist]

        if cano not in cano_attrs:
            cano_attrs[cano] = CanonicalAttrs(display_name=disp, mono_mass=mono, formula=formula, synonyms_norm=syn_norms)
        else:
            prev = cano_attrs[cano]
            merged = set(prev.synonyms_norm or []) | set(syn_norms)
            prev.synonyms_norm = sorted(merged)

        for sn in syn_norms:
            syn2cano[sn] = cano

    if overrides_csv and Path(overrides_csv).exists():
        try:
            odf = pd.read_csv(overrides_csv)
            for _, r in odf.iterrows():
                syn_raw = str(r.get("synonym_raw") or r.get("synonym") or "").strip()
                tgt = str(r.get("canonical_name") or "").strip()
                if not syn_raw or not tgt:
                    continue
                sn = normalize_name(syn_raw)
                ckey = base_name(tgt)
                syn2cano[sn] = ckey
                if ckey not in cano_attrs:
                    cano_attrs[ckey] = CanonicalAttrs(display_name=tgt, mono_mass=None, formula=None, synonyms_norm=[sn])
                else:
                    prev = cano_attrs[ckey]
                    merged = set(prev.synonyms_norm or []) | {sn}
                    prev.synonyms_norm = sorted(merged)
        except Exception as e:
            warnings.warn(f"Failed to load manual overrides: {e}")

    return syn2cano, cano_attrs, blacklist

# ---------------------- SMPDB expansion ---------------------- #
def expand_smpdb_sets(smpdb_csv: Path, syn2cano: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Expand SMPDB pathway→metabolite sets into canonical name-space via synonyms.

    Returns:
      expanded_sets: Dict[pathway_name] -> List[canonical_key]
      smpdb_met_to_cano: Dict[original_smpdb_metabolite_name] -> canonical_key (for audit)
    """
    df = pd.read_csv(smpdb_csv)
    cols = {c.lower().strip(): c for c in df.columns}
    term_col = cols.get("pathway", next((cols[k] for k in cols if "pathway" in k or "name" in k), df.columns[0]))
    met_col = next((cols[k] for k in cols if "metabolite" in k or "compound" in k or "name" in k and cols[k] != term_col), df.columns[1])
    df = df[[term_col, met_col]].dropna()

    expanded: Dict[str, set] = {}
    smpdb_met_to_cano: Dict[str, str] = {}

    for pw, sub in df.groupby(term_col):
        sink = expanded.setdefault(str(pw), set())
        for met in sub[met_col].astype(str):
            met_orig = met
            norm = normalize_name(met)
            if norm in syn2cano:
                cano = syn2cano[norm]
                sink.add(cano)
                smpdb_met_to_cano[met_orig] = cano
            else:
                sink.add(norm)
                smpdb_met_to_cano[met_orig] = norm

    return {k: sorted(v) for k,v in expanded.items()}, smpdb_met_to_cano


# ---------------------- m/z / adduct helpers ---------------------- #
_H_MASS = 1.007276466812
_NA_MASS = 22.989218
_K_MASS  = 38.963158

def infer_neutral_mass(mz: Optional[float], adduct: Optional[str]) -> Optional[float]:
    if mz is None or not isinstance(mz, (float,int)):
        return None
    if not adduct:
        return None
    a = adduct.lower().strip()
    try:
        if a in ("[m+h]+",):
            return float(mz) - _H_MASS
        if a in ("[m-h]-",):
            return float(mz) + _H_MASS
        if a in ("[m+na]+",):
            return float(mz) - _NA_MASS
        if a in ("[m+k]+",):
            return float(mz) - _K_MASS
    except Exception:
        return None
    return None

# ---------------------- Candidate resolver ---------------------- #
@dataclass
class ResolvedCandidate:
    canonical: str
    weight: float
    method: str
    matched_synonym: str
    name_score: float
    mass_score: float
    adduct_score: float

def _fuzzy_best(query_norm: str, choices: Iterable[str]) -> Tuple[str, int]:
    if _HAVE_RF:
        got = rf_process.extractOne(query_norm, list(choices), scorer=rf_fuzz.token_set_ratio)
        if got:
            return got[0], int(got[1])
        return "", 0
    else:
        got = difflib.get_close_matches(query_norm, list(choices), n=1, cutoff=0.0)
        if got:
            score = int(100 * difflib.SequenceMatcher(None, query_norm, got[0]).ratio())
            return got[0], score
        return "", 0

def resolve_candidates(candidate_names: List[str],
                       adduct: Optional[str],
                       ppm: Optional[float],
                       mz: Optional[float],
                       syn2cano: Dict[str, str],
                       cano_attrs: Dict[str, CanonicalAttrs],
                       params: ResolveParams) -> List[ResolvedCandidate]:
    """
    Return canonical candidates with evidence-weighted scores (not normalized).
    """
    results: List[ResolvedCandidate] = []
    if not candidate_names:
        return results

    neutral_mass = infer_neutral_mass(mz, adduct)
    adduct_key = (adduct or "").lower().strip()
    adduct_score = params.adduct_weights.get(adduct_key, 0.6 if adduct_key else 1.0)

    mass_score = 1.0
    if isinstance(ppm, (int,float)) and ppm >= 0:
        mass_score = math.exp(- (float(ppm) / max(1e-6, params.ppm_tau))**2)

    for raw in candidate_names:
        qn = normalize_name(raw)
        if not qn:
            continue

        method = "none"
        name_score = 0.0
        canonical = None
        matched_syn = ""

        if qn in syn2cano:
            canonical = syn2cano[qn]
            name_score = 1.0
            method = "exact_syn"
            matched_syn = qn
        else:
            best_syn, score = _fuzzy_best(qn, syn2cano.keys())
            if score >= params.fuzzy_strict:
                canonical = syn2cano[best_syn]
                name_score = 0.9
                method = "fuzzy_strict"
                matched_syn = best_syn
            elif score >= params.fuzzy_loose:
                canonical = syn2cano[best_syn]
                name_score = 0.7
                method = "fuzzy_loose"
                matched_syn = best_syn
            else:
                canonical = qn
                name_score = 0.4
                method = "fallback_norm"
                matched_syn = qn

        mscore = mass_score
        if canonical in cano_attrs and neutral_mass is not None and cano_attrs[canonical].mono_mass:
            try:
                db_mass = float(cano_attrs[canonical].mono_mass)
                ppm_est = abs(neutral_mass - db_mass) / max(1e-12, db_mass) * 1e6
                mscore = math.exp(- (ppm_est / max(1e-6, params.ppm_tau))**2)
            except Exception:
                pass

        weight = name_score * mscore * adduct_score
        results.append(ResolvedCandidate(
            canonical=canonical, weight=weight, method=method,
            matched_synonym=matched_syn, name_score=name_score,
            mass_score=mscore, adduct_score=adduct_score
        ))

    acc: Dict[str, ResolvedCandidate] = {}
    for rc in results:
        if rc.canonical not in acc:
            acc[rc.canonical] = rc
        else:
            prev = acc[rc.canonical]
            acc[rc.canonical] = ResolvedCandidate(
                canonical=rc.canonical,
                weight=prev.weight + rc.weight,
                method=prev.method if prev.weight >= rc.weight else rc.method,
                matched_synonym=prev.matched_synonym if prev.weight >= rc.weight else rc.matched_synonym,
                name_score=max(prev.name_score, rc.name_score),
                mass_score=max(prev.mass_score, rc.mass_score),
                adduct_score=max(prev.adduct_score, rc.adduct_score)
            )

    return list(acc.values())
