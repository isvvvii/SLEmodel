
import torch
import torch.nn as nn
from . import config as cfg

class PropensityScoreClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = cfg.NUM_CLASSES

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    """Encode shared and modality-specific representations."""
    def __init__(self, input_dim, latent_dim, use_decoupling=True):
        super().__init__()
        self.use_decoupling = use_decoupling

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        if self.use_decoupling:
            self.private_encoder = nn.Sequential(
                nn.Linear(input_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, latent_dim)
            )

    def forward(self, x):
        shared = self.shared_encoder(x)
        if self.use_decoupling:
            private = self.private_encoder(x)
        else:
            private = torch.zeros_like(shared)
        return shared, private

class Decoder(nn.Module):
    """Reconstruct a modality from shared and private representations."""
    def __init__(self, output_dim, latent_dim, use_decoupling=True):
        super().__init__()
        self.use_decoupling = use_decoupling

        input_dim = latent_dim * 2 if use_decoupling else latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, output_dim),
        )

    def forward(self, z_shared, z_private):
        if self.use_decoupling:
            z_combined = torch.cat((z_shared, z_private), dim=1)
        else:
            z_combined = z_shared
        return self.decoder(z_combined)

class CrossModalAttentionFusion(nn.Module):
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
        )  # attn_weights: [B, 3, 3] if requested else None

        x = self.norm(x + attn_output)
        fused_z = x.mean(dim=1)  # [B, D]

        if return_attn:
            return fused_z, attn_weights
        return fused_z

class ConcatMLPFusion(nn.Module):
    """Fuse concatenated shared representations with an MLP."""
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, gly_z, mass_z, rna_z):
        concat_z = torch.cat([gly_z, mass_z, rna_z], dim=1)
        return self.mlp(concat_z)

class DecoupledModel(nn.Module):
    def __init__(self, gly_dim, mass_dim, rna_dim,
                 latent_dim=None,
                 use_cross_attention=True,
                 use_decoupling=True,
                 attention_num_heads=1,
                 fusion_method='attention',
                 dropout_rate=None,
                 single_modality_mode=None):
        super().__init__()

        if latent_dim is None: latent_dim = cfg.LATENT_DIM
        if dropout_rate is None: dropout_rate = cfg.DROPOUT_RATE

        self.single_modality_mode = single_modality_mode
        self.use_decoupling = use_decoupling
        self.use_cross_attention = use_cross_attention
        self.fusion_method = fusion_method

        self.gly_encoder = Encoder(gly_dim, latent_dim, use_decoupling)
        self.mass_encoder = Encoder(mass_dim, latent_dim, use_decoupling)
        self.rna_encoder = Encoder(rna_dim, latent_dim, use_decoupling)

        self.gly_decoder = Decoder(gly_dim, latent_dim, use_decoupling)
        self.mass_decoder = Decoder(mass_dim, latent_dim, use_decoupling)
        self.rna_decoder = Decoder(rna_dim, latent_dim, use_decoupling)

        if fusion_method == 'attention' and use_cross_attention:
            self.fusion_module = CrossModalAttentionFusion(latent_dim, attention_num_heads)
        elif fusion_method == 'concat_mlp':
            self.fusion_module = ConcatMLPFusion(latent_dim)

        classifier_input_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim // 2, cfg.NUM_CLASSES)
        )

    def forward(self, gly_x, mass_x, rna_x, return_attn: bool = False, mask_shared=None):
        """Return class logits, reconstructions and latent representations."""
        attn_w = None

        # ----------------------------
        # ----------------------------
        if self.single_modality_mode:
            mode = self.single_modality_mode

            if mode == 'gly':
                z_s_gly, z_p_gly = self.gly_encoder(gly_x)
                if mask_shared and mask_shared.get('gly', False):
                    z_s_gly = torch.zeros_like(z_s_gly)
                z_s_fused = z_s_gly

                recon_gly = self.gly_decoder(z_s_fused, z_p_gly)
                recon_mass = torch.zeros_like(mass_x)
                recon_rna = torch.zeros_like(rna_x)

                representations = {
                    'shared':  {'gly': z_s_gly, 'mass': torch.zeros_like(z_s_gly), 'rna': torch.zeros_like(z_s_gly)},
                    'private': {'gly': z_p_gly, 'mass': torch.zeros_like(z_p_gly), 'rna': torch.zeros_like(z_p_gly)},
                }

            elif mode == 'mass':
                z_s_mass, z_p_mass = self.mass_encoder(mass_x)
                if mask_shared and mask_shared.get('mass', False):
                    z_s_mass = torch.zeros_like(z_s_mass)
                z_s_fused = z_s_mass

                recon_gly = torch.zeros_like(gly_x)
                recon_mass = self.mass_decoder(z_s_fused, z_p_mass)
                recon_rna = torch.zeros_like(rna_x)

                representations = {
                    'shared':  {'gly': torch.zeros_like(z_s_mass), 'mass': z_s_mass, 'rna': torch.zeros_like(z_s_mass)},
                    'private': {'gly': torch.zeros_like(z_p_mass), 'mass': z_p_mass, 'rna': torch.zeros_like(z_p_mass)},
                }

            elif mode == 'rna':
                z_s_rna, z_p_rna = self.rna_encoder(rna_x)
                if mask_shared and mask_shared.get('rna', False):
                    z_s_rna = torch.zeros_like(z_s_rna)
                z_s_fused = z_s_rna

                recon_gly = torch.zeros_like(gly_x)
                recon_mass = torch.zeros_like(mass_x)
                recon_rna = self.rna_decoder(z_s_fused, z_p_rna)

                representations = {
                    'shared':  {'gly': torch.zeros_like(z_s_rna), 'mass': torch.zeros_like(z_s_rna), 'rna': z_s_rna},
                    'private': {'gly': torch.zeros_like(z_p_rna), 'mass': torch.zeros_like(z_p_rna), 'rna': z_p_rna},
                }

            else:
                raise ValueError(f"Unknown single_modality_mode: {mode}")

        # ----------------------------
        # ----------------------------
        else:
            z_s_gly, z_p_gly = self.gly_encoder(gly_x)
            z_s_mass, z_p_mass = self.mass_encoder(mass_x)
            z_s_rna, z_p_rna = self.rna_encoder(rna_x)

            if mask_shared is not None:
                if mask_shared.get('gly', False):
                    z_s_gly = torch.zeros_like(z_s_gly)
                if mask_shared.get('mass', False):
                    z_s_mass = torch.zeros_like(z_s_mass)
                if mask_shared.get('rna', False):
                    z_s_rna = torch.zeros_like(z_s_rna)

            if self.fusion_method == 'attention' and self.use_cross_attention:
                if return_attn:
                    z_s_fused, attn_w = self.fusion_module(z_s_gly, z_s_mass, z_s_rna, return_attn=True)
                else:
                    z_s_fused = self.fusion_module(z_s_gly, z_s_mass, z_s_rna, return_attn=False)

            elif self.fusion_method == 'concat_mlp':
                z_s_fused = self.fusion_module(z_s_gly, z_s_mass, z_s_rna)

            else:
                z_s_fused = (z_s_gly + z_s_mass + z_s_rna) / 3

            recon_gly = self.gly_decoder(z_s_fused, z_p_gly)
            recon_mass = self.mass_decoder(z_s_fused, z_p_mass)
            recon_rna = self.rna_decoder(z_s_fused, z_p_rna)

            representations = {
                'shared':  {'gly': z_s_gly, 'mass': z_s_mass, 'rna': z_s_rna},
                'private': {'gly': z_p_gly, 'mass': z_p_mass, 'rna': z_p_rna},
            }

        if attn_w is not None:
            representations['attn'] = attn_w  # [B, 3, 3]

        logits = self.classifier(z_s_fused)
        reconstructions = {'gly': recon_gly, 'mass': recon_mass, 'rna': recon_rna}
        return logits, reconstructions, representations

    def reconstruct_single_modality(self, data, modality: str):
        if self.single_modality_mode and self.single_modality_mode != modality:
            if modality == 'gly':
                latent_dim = self.gly_encoder.shared_encoder[-1].out_features
            elif modality == 'mass':
                latent_dim = self.mass_encoder.shared_encoder[-1].out_features
            elif modality == 'rna':
                latent_dim = self.rna_encoder.shared_encoder[-1].out_features

            z_s = torch.zeros(data.shape[0], latent_dim).to(data.device)
            z_p = torch.zeros(data.shape[0], latent_dim).to(data.device)
            return torch.zeros_like(data), z_s, z_p

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
            return None, None, None

        return recon, z_s, z_p
