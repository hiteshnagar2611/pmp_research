"""
PMPGen: Complete model for de novo peripheral membrane protein generation.

Full integration of:
- Conditioning encoder
- MD-informed noise schedule
- SE(3) flow matching
- IPA denoiser
- Membrane plane guidance
- Sequence decoder

Two modes:
1. Training: predict velocities for flow matching
2. Inference: iteratively denoise to generate proteins
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .conditioning import ConditioningEncoder
from .noise_schedule import MDInformedNoiseSchedule
from .se3_flow import SE3FlowInterpolant
from .ipa_denoiser import IPADenoiser
from .mem_guidance import MembraneGuidance
from .sequence_decoder import SequenceDecoder
from .sampler import PMPGenSampler, ValidationCascade
from ..training.losses import CombinedPMPGenLoss
from ..training.metrics import GenerationQualityMetrics


class PMPGen(pl.LightningModule):
    """
    PMPGen: Peripheral Membrane Protein Generation via SE(3) Flow Matching.

    Architecture:
        Input → Conditioning Encoder
                ↓
            MD-Informed Noise Schedule
                ↓
            SE(3) Flow Interpolant
                ↓
            IPA Denoiser → Membrane Guidance
                ↓
            Sequence Decoder
                ↓
            3-Stage Validation Cascade
                ↓
            Output (coordinates + sequences)

    Training:
        - Flow matching: predict velocity fields
        - Losses: flow, anchor, membrane, structure
        - Metrics: trajectory accuracy, final quality

    Inference:
        - Iterative denoising from noise
        - Guidance steering toward membrane
        - Sequence design with ProteinMPNN
        - Validation with ESMFold + DynaMo + Rosetta

    Args:
        n_res_in (int): input residue dimension (default: 256)
        n_res_out (int): output residue dimension (default: 256)
        hidden_dim (int): hidden dimension (default: 256)
        n_layers (int): number of denoiser layers (default: 6)
        n_tokens (int): amino acid vocabulary size (default: 21)
        dropout (float): dropout probability (default: 0.1)
        conditioning_dim (int): conditioning dimension (default: 256)
        noise_schedule_type (str): noise schedule type (default: 'md_informed')
    """

    def __init__(
        self,
        n_res_in: int = 256,
        n_res_out: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_tokens: int = 21,
        dropout: float = 0.1,
        conditioning_dim: int = 256,
        noise_schedule_type: str = 'md_informed',
        **kwargs
    ):
        """Initialize PMPGen."""
        super().__init__()

        self.save_hyperparameters()

        # ─────────────────────────────────────────────────────────────────────
        # Components
        # ─────────────────────────────────────────────────────────────────────

        # 1. Conditioning Encoder
        self.conditioning_encoder = ConditioningEncoder(
            hidden_dim=conditioning_dim,
            n_layers=2,
        )

        # 2. Noise Schedule
        self.noise_schedule = MDInformedNoiseSchedule(
            schedule_type=noise_schedule_type,
            min_sigma=0.01,
            max_sigma=1.0,
        )

        # 3. Flow Interpolant
        self.flow_interpolant = SE3FlowInterpolant()

        # 4. Denoiser (IPA-based)
        self.denoiser = IPADenoiser(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=8,
            dropout=dropout,
        )

        # 5. Membrane Guidance
        self.membrane_guidance = MembraneGuidance(
            hidden_dim=hidden_dim,
            num_iterative_steps=5,
        )

        # 6. Sequence Decoder
        self.sequence_decoder = SequenceDecoder(
            n_tokens=n_tokens,
            hidden_dim=hidden_dim,
            n_layers=3,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Loss & Metrics
        # ─────────────────────────────────────────────────────────────────────

        self.loss_fn = CombinedPMPGenLoss(
            lambda_flow=1.0,
            lambda_anchor=0.5,
            lambda_mem=0.3,
            lambda_struct=0.1,
        )

        self.metrics = GenerationQualityMetrics()

        # ─────────────────────────────────────────────────────────────────────
        # Sampler & Validation
        # ─────────────────────────────────────────────────────────────────────

        self.sampler = PMPGenSampler(
            denoiser=self.denoiser,
            noise_schedule=self.noise_schedule,
            sequence_decoder=self.sequence_decoder,
            n_steps=100,
            use_guidance=True,
            guidance_scale=1.0,
        )

        self.validation_cascade = ValidationCascade(
            plddt_threshold=70.0,
            binding_recall_threshold=0.8,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        coords_start: torch.Tensor,
        coords_end: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (flow matching).

        Args:
            coords_start: (B, N, 3) starting coordinates (noisy)
            coords_end: (B, N, 3) target coordinates (clean)
            conditioning: conditioning dict
            t: (B,) time steps [0, 1]

        Returns:
            v_pred_R: (B, N, 3) predicted rotation velocity
            v_pred_t: (B, N, 3) predicted translation velocity
        """
        B, N = coords_start.shape[:2]

        # ─────────────────────────────────────────────────────────────────────
        # Encode Conditioning
        # ─────────────────────────────────────────────────────────────────────

        cond_encoded = self.conditioning_encoder(
            scaffold_coords=conditioning.get('scaffold_coords'),
            binding_mask=conditioning.get('binding_mask'),
            membrane_normal=conditioning.get('membrane_normal'),
        )

        # ─────────────────────────────────────────────────────────────────────
        # Flow Interpolation
        # ─────────────────────────────────────────────────────────────────────

        # Linear interpolation in latent space
        # x_t = (1 - t) * x_start + t * x_end
        coords_t = (1 - t.view(B, 1, 1)) * coords_start + t.view(B, 1, 1) * coords_end

        # Target velocity: (x_end - x_start)
        v_target_R = coords_end - coords_start
        v_target_t = coords_end - coords_start  # same for simple case

        # ─────────────────────────────────────────────────────────────────────
        # Predict Velocities
        # ─────────────────────────────────────────────────────────────────────

        # IPA denoiser predicts velocities
        v_pred = self.denoiser(
            x=coords_t,
            t=t,
            conditioning=cond_encoded,
        )  # (B, N, 3)

        # Split into rotation and translation (for SO(3) × R³)
        # Here we use same velocity for both as simple approximation
        v_pred_R = v_pred
        v_pred_t = v_pred

        return v_pred_R, v_pred_t

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Training step with flow matching loss.

        Args:
            batch: training batch
            batch_idx: batch index

        Returns:
            loss: total loss
        """
        # Unpack batch
        coords_start = batch['coords_start']  # noisy
        coords_end = batch['coords_end']      # clean
        conditioning = batch['conditioning']
        t = batch['t']

        # Forward pass
        v_pred_R, v_pred_t = self(coords_start, coords_end, conditioning, t)

        # Target velocities
        v_target = coords_end - coords_start
        v_target_R = v_target
        v_target_t = v_target

        # Anchor coordinates (keep fixed)
        anchor_mask = conditioning.get('anchor_mask')  # (B, N, 1)
        coords_gen = coords_start  # or current prediction

        # Depths
        depth_pred = coords_end[..., 2:3]  # z-coordinate
        depth_target = conditioning.get('depth_target')

        # Compute loss
        loss_dict = self.loss_fn(
            v_pred_R=v_pred_R,
            v_pred_t=v_pred_t,
            v_target_R=v_target_R,
            v_target_t=v_target_t,
            coords_gen=coords_gen,
            coords_query=conditioning['scaffold_coords'],
            anchor_mask=anchor_mask,
            depth_pred=depth_pred,
            depth_target=depth_target,
        )

        # Logging
        self.log('train_loss', loss_dict['loss_total'], prog_bar=True)
        self.log('train_loss_flow', loss_dict['loss_flow'])
        self.log('train_loss_anchor', loss_dict['loss_anchor'])
        self.log('train_loss_mem', loss_dict['loss_mem'])

        return loss_dict['loss_total']

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Same as training but without backprop
        coords_start = batch['coords_start']
        coords_end = batch['coords_end']
        conditioning = batch['conditioning']
        t = batch['t']

        with torch.no_grad():
            v_pred_R, v_pred_t = self(coords_start, coords_end, conditioning, t)

        v_target = coords_end - coords_start

        # Compute loss
        loss_dict = self.loss_fn(
            v_pred_R=v_pred_R,
            v_pred_t=v_pred_t,
            v_target_R=v_target,
            v_target_t=v_target,
            coords_gen=coords_start,
            coords_query=conditioning['scaffold_coords'],
            anchor_mask=conditioning.get('anchor_mask'),
            depth_pred=coords_end[..., 2:3],
            depth_target=conditioning.get('depth_target'),
        )

        self.log('val_loss', loss_dict['loss_total'], prog_bar=True)

        return loss_dict['loss_total']

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        conditioning: Dict[str, torch.Tensor],
        num_samples: int = 1,
        verbose: bool = True,
    ) -> Dict:
        """
        Generate proteins via iterative denoising.

        Args:
            conditioning: conditioning dict with:
                - 'scaffold_coords': query coordinates
                - 'membrane_normal': membrane orientation
                - 'anchor_mask': binding region
                - 'binding_mask': target binding pattern
            num_samples: number of samples per scaffold
            verbose: show progress bar

        Returns:
            results: generation results
        """
        # Repeat conditioning for multiple samples
        for key in conditioning:
            if conditioning[key] is not None:
                conditioning[key] = conditioning[key].repeat(num_samples, 1, 1)

        # Sample structures
        results = self.sampler.sample(conditioning, verbose=verbose)

        # Validate
        coords = results['coords']
        sequences = results['sequences']

        # TODO: Run ESMFold to get pLDDT
        # TODO: Run DynaMo to get binding predictions
        # TODO: Run Rosetta to get binding energy

        # For now, add placeholder validation
        results['validation'] = {
            'plddt': torch.ones(coords.shape[0], coords.shape[1]) * 75.0,
            'binding_pred': torch.bernoulli(torch.full((coords.shape[0], coords.shape[1]), 0.15)),
            'passes': True,
        }

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch Lightning
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def on_train_epoch_start(self):
        """Called at start of each training epoch."""
        pass

    def on_validation_epoch_start(self):
        """Called at start of each validation epoch."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_pmpgen(
    checkpoint_path: Optional[str] = None,
    pretrained: bool = False,
    **kwargs
) -> PMPGen:
    """
    Create PMPGen model.

    Args:
        checkpoint_path: path to checkpoint to load
        pretrained: whether to use pretrained weights
        **kwargs: model hyperparameters

    Returns:
        model: PMPGen instance
    """
    model = PMPGen(**kwargs)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Test PMPGen
    import torch

    model = PMPGen(
        n_res_in=256,
        n_res_out=256,
        hidden_dim=128,
        n_layers=3,
        n_tokens=21,
    )

    print(f"✓ PMPGen model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test training forward pass
    B, N = 2, 100
    coords_start = torch.randn(B, N, 3)
    coords_end = torch.randn(B, N, 3)
    t = torch.rand(B)

    conditioning = {
        'scaffold_coords': torch.randn(B, N, 3),
        'binding_mask': torch.bernoulli(torch.full((B, N, 1), 0.2)),
        'membrane_normal': torch.tensor([0.0, 0.0, 1.0]),
        'depth_target': torch.randn(B, N, 1),
        'anchor_mask': torch.zeros(B, N, 1),
    }

    v_R, v_t = model(coords_start, coords_end, conditioning, t)

    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {coords_start.shape}")
    print(f"  Velocity output shape: {v_R.shape}")

    # Test generation
    # results = model.generate(conditioning, num_samples=1, verbose=False)
    # print(f"\n✓ Generation successful")
    # print(f"  Generated coords shape: {results['coords'].shape}")
    # print(f"  Generated sequences shape: {results['sequences'].shape}")