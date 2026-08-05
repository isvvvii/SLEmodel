# models.py

import torch
import torch.nn as nn
from . import config as cfg

class PropensityScoreClassifier(nn.Module):
    """Multilayer classifier used to estimate propensity scores."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, cfg.NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    """Encode shared and modality-specific representations."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        private = self.private_encoder(x)
        return shared, private

class Decoder(nn.Module):
    """Reconstruct one modality from shared and private representations."""
    def __init__(self, output_dim, latent_dim):
        super().__init__()
        input_dim = latent_dim * 2

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, output_dim),
        )

    def forward(self, z_shared, z_private):
        z_combined = torch.cat((z_shared, z_private), dim=1)
        return self.decoder(z_combined)

class CrossModalAttentionFusion(nn.Module):
    """Fuse shared representations with multi-head self-attention."""
    def __init__(self, latent_dim, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, gly_z, mass_z, rna_z, return_attn: bool = False):
        x = torch.stack([gly_z, mass_z, rna_z], dim=1)  # [B, 3, D]

        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            need_weights=return_attn,
            average_attn_weights=True
        )

        x = self.norm(x + attn_output)
        fused_z = x.mean(dim=1)  # [B, D]

        if return_attn:
            return fused_z, attn_weights
        return fused_z


class DecoupledModel(nn.Module):
    """SLEmodel architecture used in the primary analysis."""
    def __init__(self, gly_dim, mass_dim, rna_dim):
        super().__init__()

        latent_dim = cfg.LATENT_DIM
        dropout_rate = cfg.DROPOUT_RATE

        self.gly_encoder = Encoder(gly_dim, latent_dim)
        self.mass_encoder = Encoder(mass_dim, latent_dim)
        self.rna_encoder = Encoder(rna_dim, latent_dim)

        self.gly_decoder = Decoder(gly_dim, latent_dim)
        self.mass_decoder = Decoder(mass_dim, latent_dim)
        self.rna_decoder = Decoder(rna_dim, latent_dim)

        num_heads = int(getattr(cfg, "ATTENTION_NUM_HEADS", 1))
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}")
        self.fusion_module = CrossModalAttentionFusion(latent_dim, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim // 2, cfg.NUM_CLASSES)
        )

        self.cached_attention = None

    def forward(self, gly_x, mass_x, rna_x, return_attn: bool = False, mask_shared: dict | None = None):
        """Return class logits, reconstructions and latent representations."""
        z_s_gly, z_p_gly = self.gly_encoder(gly_x)
        z_s_mass, z_p_mass = self.mass_encoder(mass_x)
        z_s_rna, z_p_rna = self.rna_encoder(rna_x)

        if mask_shared is not None:
            if mask_shared.get('gly', False):  z_s_gly = torch.zeros_like(z_s_gly)
            if mask_shared.get('mass', False): z_s_mass = torch.zeros_like(z_s_mass)
            if mask_shared.get('rna', False):  z_s_rna = torch.zeros_like(z_s_rna)

        if return_attn:
            z_s_fused, attn_w = self.fusion_module(z_s_gly, z_s_mass, z_s_rna, return_attn=True)
            self.cached_attention = attn_w.detach() if attn_w is not None else None
        else:
            z_s_fused = self.fusion_module(z_s_gly, z_s_mass, z_s_rna, return_attn=False)
            attn_w = None
            self.cached_attention = None

        recon_gly = self.gly_decoder(z_s_fused, z_p_gly)
        recon_mass = self.mass_decoder(z_s_fused, z_p_mass)
        recon_rna = self.rna_decoder(z_s_fused, z_p_rna)
        reconstructions = {'gly': recon_gly, 'mass': recon_mass, 'rna': recon_rna}

        representations = {
            'shared':  {'gly': z_s_gly, 'mass': z_s_mass, 'rna': z_s_rna},
            'private': {'gly': z_p_gly, 'mass': z_p_mass, 'rna': z_p_rna},
        }
        if attn_w is not None:
            representations['attn'] = attn_w

        logits = self.classifier(z_s_fused)
        return logits, reconstructions, representations

    def reconstruct_single_modality(self, data, modality: str):
        """Reconstruct a single measured modality."""
        if modality == 'gly':
            z_s, z_p = self.gly_encoder(data)
            recon = self.gly_decoder(z_s, z_p)
        elif modality == 'mass':
            z_s, z_p = self.mass_encoder(data)
            recon = self.mass_decoder(z_s, z_p)
        elif modality == 'rna':
            z_s, z_p = self.rna_encoder(data)
            recon = self.rna_decoder(z_s, z_p)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return recon, z_s, z_p
