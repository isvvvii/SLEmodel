# dataset.py

import torch
from torch.utils.data import Dataset
import logging
from .utils import get_ot_coupling

logger = logging.getLogger(__name__)


class StaticMatchedDataset(Dataset):
    """Dataset with fixed barycentric pseudo-pairs."""
    def __init__(
        self,
        gly_data,
        gly_labels,
        all_mass_data,
        all_rna_data,
        mass_coupling,
        rna_coupling,
        gly_ids,
    ):
        self.gly_data = gly_data
        self.gly_labels = gly_labels

        self.matched_mass = (mass_coupling @ all_mass_data).clone()
        self.matched_rna = (rna_coupling @ all_rna_data).clone()

        self.gly_ids = gly_ids

    def __len__(self):
        return len(self.gly_data)

    def __getitem__(self, idx):
        return {
            "gly_x": self.gly_data[idx],
            "mass_x": self.matched_mass[idx],
            "rna_x": self.matched_rna[idx],
            "label": self.gly_labels[idx],
            "id": self.gly_ids[idx],
            "original_idx": idx,
            "is_matched": True,
        }


class DynamicHybridDataset(Dataset):
    """Training dataset with periodically updated pseudo-pairs."""
    def __init__(
        self,
        gly_data,
        gly_labels,
        all_mass_data,
        all_rna_data,
        ps_gly,
        ps_mass,
        ps_rna,
        mass_reuse_factor=24,
        rna_reuse_factor=80,
        dynamic_resample=True,
        reg_mass=5e-4,
        reg_rna=5e-3,
        ot_method_mass: str = "standard",
        ot_method_rna: str = "standard",
        ot_cost_metric_mass: str = "sqeuclidean",
        ot_cost_metric_rna: str = "sqeuclidean",
        resample_every: int = 5,
        sinkhorn_numItermax_mass: int = 400,
        sinkhorn_stopThr_mass: float = 1e-3,
        sinkhorn_numItermax_rna: int = 600,
        sinkhorn_stopThr_rna: float = 1e-3,
    ):
        self.gly_data = gly_data
        self.gly_labels = gly_labels
        self.all_mass_data = all_mass_data
        self.all_rna_data = all_rna_data

        self.ps_gly = ps_gly
        self.ps_mass = ps_mass
        self.ps_rna = ps_rna

        self.dynamic_resample = dynamic_resample

        # modality-specific OT configs
        self.reg_mass = float(reg_mass)
        self.reg_rna = float(reg_rna)
        self.ot_method_mass = ot_method_mass
        self.ot_method_rna = ot_method_rna
        self.ot_cost_metric_mass = ot_cost_metric_mass
        self.ot_cost_metric_rna = ot_cost_metric_rna

        # speed controls
        self.resample_every = max(int(resample_every), 1)
        self.sinkhorn_numItermax_mass = int(sinkhorn_numItermax_mass)
        self.sinkhorn_stopThr_mass = float(sinkhorn_stopThr_mass)
        self.sinkhorn_numItermax_rna = int(sinkhorn_numItermax_rna)
        self.sinkhorn_stopThr_rna = float(sinkhorn_stopThr_rna)

        self._resample_calls = 0

        self.mass_subset_size = max(
            1,
            min(len(all_mass_data), int(len(gly_data) / max(len(all_mass_data), 1) * mass_reuse_factor)),
        )
        self.rna_subset_size = max(
            1,
            min(len(all_rna_data), int(len(gly_data) / max(len(all_rna_data), 1) * rna_reuse_factor)),
        )

        logger.info(f"Dynamic matching subset sizes: Mass={self.mass_subset_size}, RNA={self.rna_subset_size}")
        logger.info(f"Using modality-specific reg: Mass={self.reg_mass}, RNA={self.reg_rna}")
        logger.info(f"Using OT methods: Mass={self.ot_method_mass}, RNA={self.ot_method_rna}")
        logger.info(f"OT cost metrics: Mass={self.ot_cost_metric_mass}, RNA={self.ot_cost_metric_rna}")
        logger.info(
            f"OT speed controls: resample_every={self.resample_every}, "
            f"mass(iters={self.sinkhorn_numItermax_mass}, thr={self.sinkhorn_stopThr_mass}), "
            f"rna(iters={self.sinkhorn_numItermax_rna}, thr={self.sinkhorn_stopThr_rna})"
        )

        # placeholders
        self.matched_mass_data = torch.zeros(len(gly_data), all_mass_data.shape[1])
        self.matched_rna_data = torch.zeros(len(gly_data), all_rna_data.shape[1])

        # initial pseudo-samples
        self.resample_and_match()

    def __len__(self):
        return len(self.gly_data)

    def resample_and_match(self):
        if not self.dynamic_resample:
            if hasattr(self, "_static_generated"):
                return
            self._static_generated = True

        self._resample_calls += 1
        if self.dynamic_resample and self.resample_every > 1:
            if (self._resample_calls - 1) % self.resample_every != 0:
                return

        logger.info("Resampling and generating new pseudo-samples for the epoch...")

        mass_indices = torch.randperm(len(self.all_mass_data))[: self.mass_subset_size]
        rna_indices = torch.randperm(len(self.all_rna_data))[: self.rna_subset_size]

        mass_subset = self.all_mass_data[mass_indices]
        rna_subset = self.all_rna_data[rna_indices]

        ps_mass_subset = self.ps_mass[mass_indices]
        ps_rna_subset = self.ps_rna[rna_indices]

        mass_coupling = get_ot_coupling(
            self.ps_gly,
            ps_mass_subset,
            reg=self.reg_mass,
            method=self.ot_method_mass,
            sinkhorn_numItermax=self.sinkhorn_numItermax_mass,
            sinkhorn_stopThr=self.sinkhorn_stopThr_mass,
            purpose="train",
            cost_metric=self.ot_cost_metric_mass,
        )
        rna_coupling = get_ot_coupling(
            self.ps_gly,
            ps_rna_subset,
            reg=self.reg_rna,
            method=self.ot_method_rna,
            sinkhorn_numItermax=self.sinkhorn_numItermax_rna,
            sinkhorn_stopThr=self.sinkhorn_stopThr_rna,
            purpose="train",
            cost_metric=self.ot_cost_metric_rna,
        )

        self.matched_mass_data = mass_coupling @ mass_subset
        self.matched_rna_data = rna_coupling @ rna_subset

        logger.info("Pseudo-samples for the epoch have been generated.")

    def __getitem__(self, idx):
        gly_x = self.gly_data[idx]
        label = self.gly_labels[idx]

        mass_x = self.matched_mass_data[idx]
        rna_x = self.matched_rna_data[idx]

        real_gly = self.gly_data[torch.randint(len(self.gly_data), (1,)).item()]
        real_mass = self.all_mass_data[torch.randint(len(self.all_mass_data), (1,)).item()]
        real_rna = self.all_rna_data[torch.randint(len(self.all_rna_data), (1,)).item()]

        return {
            "gly_x": gly_x,
            "mass_x": mass_x,
            "rna_x": rna_x,
            "label": label,
            "is_matched": True,
            "real_gly": real_gly,
            "real_mass": real_mass,
            "real_rna": real_rna,
        }

class SingleModalityDataset(Dataset):
    """Dataset used for one-modality ablations."""
    def __init__(self, active_data, labels, ids, modality, all_dims):
        self.active_data = active_data
        self.labels = labels
        self.ids = ids
        self.modality = modality
        self.all_dims = all_dims

    def __len__(self):
        return len(self.active_data)

    def __getitem__(self, idx):
        gly_x = self.active_data[idx] if self.modality == 'gly' else torch.zeros(self.all_dims['gly'])
        mass_x = self.active_data[idx] if self.modality == 'mass' else torch.zeros(self.all_dims['mass'])
        rna_x = self.active_data[idx] if self.modality == 'rna' else torch.zeros(self.all_dims['rna'])

        return {
            'gly_x': gly_x,
            'mass_x': mass_x,
            'rna_x': rna_x,
            'label': self.labels[idx],
            'id': self.ids[idx],
            'original_idx': idx,
            'is_matched': False,
            'active_modality': self.modality
        }

class RandomMatchedDataset(Dataset):
    """Within-split random matching used for the alignment ablation."""
    def __init__(self, gly_data, gly_labels, mass_data, rna_data, gly_ids,
                 mass_ids=None, rna_ids=None, seed=42, split_type="train"):
        self.gly_data = gly_data
        self.gly_labels = gly_labels
        self.mass_data = mass_data
        self.rna_data = rna_data
        self.gly_ids = gly_ids
        self.mass_ids = mass_ids
        self.rna_ids = rna_ids
        self.split_type = split_type

        g = torch.Generator()
        g.manual_seed(seed)

        n_gly = len(gly_data)
        n_mass = len(mass_data)
        n_rna = len(rna_data)

        logger.info(f"RandomMatchedDataset ({split_type}): "
                   f"Gly={n_gly}, Mass={n_mass}, RNA={n_rna} samples")

        self.mass_indices = torch.randint(0, n_mass, (n_gly,), generator=g)
        self.rna_indices = torch.randint(0, n_rna, (n_gly,), generator=g)

        mass_reuse_count = len(set(self.mass_indices.tolist()))
        rna_reuse_count = len(set(self.rna_indices.tolist()))
        logger.info(f"Random matching ({split_type}): "
                   f"Using {mass_reuse_count}/{n_mass} mass samples, "
                   f"{rna_reuse_count}/{n_rna} rna samples")

    def __len__(self):
        return len(self.gly_data)

    def __getitem__(self, idx):
        gly_x = self.gly_data[idx]
        mass_x = self.mass_data[self.mass_indices[idx]]
        rna_x = self.rna_data[self.rna_indices[idx]]

        return {
            'gly_x': gly_x,
            'mass_x': mass_x,
            'rna_x': rna_x,
            'label': self.gly_labels[idx],
            'id': self.gly_ids[idx],
            'original_idx': idx,
            'is_matched': True,
            'matching_type': 'random',
            'split_type': self.split_type
        }
