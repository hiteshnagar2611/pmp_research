"""
Model interpretability: visualize attention weights and learned representations.

Tools:
  - Extract attention weights from multi-head attention layers
  - Plot attention heatmaps (which residues attend to each other)
  - Visualize cross-attention between structure and dynamics
  - Project learned representations (t-SNE, UMAP)
  - Analyze which features drive binding predictions
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class AttentionMapVisualizer:
    """
    Extract and visualize attention weights from DynaMo model.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Args:
            model: DynaMo model
            device: "cuda" or "cpu"
        """
        self.model = model
        self.device = device
        self.attention_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        # Hook on cross-attention layer to capture weights
        def cross_attn_hook(module, input, output):
            # output includes attention weights if return_attn=True
            if isinstance(output, tuple):
                self.attention_maps["cross_attention"] = output[1]
            return output

        self.model.cross_attn.register_forward_hook(cross_attn_hook)

    def extract_attention(
        self,
        H_geom: torch.Tensor,      # (N, hidden_dim)
        H_star: torch.Tensor,      # (N, hidden_dim)
    ) -> torch.Tensor:
        """
        Extract attention weights for a batch.

        Returns:
            attn_weights: (n_heads, N, N) attention matrix
        """
        with torch.no_grad():
            H_geom = H_geom.to(self.device)
            H_star = H_star.to(self.device)

            _, attn_weights = self.model.cross_attn(H_geom, H_star, return_attn=True)

        return attn_weights.detach().cpu()

    def plot_attention_heatmap(
        self,
        attn_weights: torch.Tensor,  # (n_heads, N, N)
        residue_ids: List[int] = None,
        head_idx: int = 0,
        save_path: str = None,
        figsize: Tuple = (12, 10),
    ):
        """
        Plot attention heatmap for a single head.

        Args:
            attn_weights: (n_heads, N, N) attention weights
            residue_ids: residue indices for x-axis labels
            head_idx: which head to plot
            save_path: where to save figure
            figsize: figure size
        """
        attn_head = attn_weights[head_idx].numpy()  # (N, N)
        N = attn_head.shape[0]

        if residue_ids is None:
            residue_ids = list(range(1, N + 1))

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(attn_head, cmap="Blues", aspect="auto")

        # Labels
        ax.set_xlabel("Key Residue Index")
        ax.set_ylabel("Query Residue Index")
        ax.set_title(f"Cross-Attention Weights (Head {head_idx})")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight")

        # Set ticks (show every 5th residue to avoid clutter)
        tick_positions = np.arange(0, N, max(1, N // 10))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

        if len(residue_ids) == N:
            ax.set_xticklabels([residue_ids[i] for i in tick_positions], rotation=45)
            ax.set_yticklabels([residue_ids[i] for i in tick_positions])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()

    def plot_head_comparison(
        self,
        attn_weights: torch.Tensor,  # (n_heads, N, N)
        save_path: str = None,
    ):
        """
        Plot all attention heads in a grid for comparison.

        Args:
            attn_weights: (n_heads, N, N)
            save_path: where to save
        """
        n_heads = attn_weights.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_heads)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        for head_idx in range(n_heads):
            attn_head = attn_weights[head_idx].numpy()

            ax = axes[head_idx]
            im = ax.imshow(attn_head, cmap="Blues", aspect="auto")
            ax.set_title(f"Head {head_idx}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")

        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()

    def analyze_attention_patterns(
        self,
        attn_weights: torch.Tensor,  # (n_heads, N, N)
        targets: torch.Tensor = None,  # (N,) binary binding labels
    ) -> Dict:
        """
        Analyze attention patterns: do binding residues attend to each other?

        Args:
            attn_weights: (n_heads, N, N)
            targets: optional binding labels

        Returns:
            analysis: dict with statistics
        """
        # Aggregate across heads
        attn_avg = attn_weights.mean(dim=0).numpy()  # (N, N)

        analysis = {
            "mean_attention": attn_avg.mean(),
            "max_attention": attn_avg.max(),
            "min_attention": attn_avg.min(),
            "diagonal_attention": np.diag(attn_avg).mean(),  # self-attention
        }

        # If labels provided, analyze binding-to-binding attention
        if targets is not None:
            targets_np = targets.cpu().numpy().astype(bool)
            binding_idx = np.where(targets_np)[0]

            if len(binding_idx) > 0:
                # Average attention between binding residues
                binding_attn = attn_avg[np.ix_(binding_idx, binding_idx)]
                analysis["binding_to_binding_attention"] = binding_attn.mean()

                # Average attention from binding to non-binding
                non_binding_idx = np.where(~targets_np)[0]
                if len(non_binding_idx) > 0:
                    inter_attn = attn_avg[np.ix_(binding_idx, non_binding_idx)]
                    analysis["binding_to_nonbinding_attention"] = inter_attn.mean()

        return analysis


class RepresentationVisualizer:
    """
    Visualize learned representations using dimensionality reduction.
    """

    def __init__(self):
        self.representations = None
        self.labels = None

    def extract_representations(
        self,
        model: nn.Module,
        test_loader,
        device: str = "cuda",
        layer: str = "output",  # which layer to extract from
    ):
        """
        Extract representations from a layer of the model.

        Args:
            model: torch model
            test_loader: data loader
            device: "cuda" or "cpu"
            layer: which layer ("output", "fused", "gvp_output", etc.)
        """
        model.eval()
        all_reps = []
        all_labels = []

        # Register hook to extract intermediate representations
        def extract_hook(module, input, output):
            all_reps.append(output.detach().cpu())

        # Get the layer and register hook
        layer_module = dict(model.named_modules()).get(layer)
        if layer_module is not None:
            layer_module.register_forward_hook(extract_hook)

        with torch.no_grad():
            for batch in test_loader:
                # Forward pass
                model(...)  # adjust based on your model

                # Collect labels
                targets = batch["targets"].reshape(-1)
                all_labels.append(targets)

        self.representations = torch.cat(all_reps, dim=0)  # (total_residues, dim)
        self.labels = torch.cat(all_labels, dim=0)  # (total_residues,)

    def plot_pca(
        self,
        save_path: str = None,
        figsize: Tuple = (10, 8),
    ):
        """
        Plot PCA projection of representations.

        Args:
            save_path: where to save
            figsize: figure size
        """
        if self.representations is None:
            print("No representations extracted. Call extract_representations() first.")
            return

        # PCA to 2D
        pca = PCA(n_components=2)
        reps_2d = pca.fit_transform(self.representations.numpy())

        fig, ax = plt.subplots(figsize=figsize)

        # Plot by class
        binding_mask = self.labels.numpy() == 1
        non_binding_mask = self.labels.numpy() == 0

        ax.scatter(
            reps_2d[binding_mask, 0],
            reps_2d[binding_mask, 1],
            c="red",
            label="Binding",
            alpha=0.6,
            s=50,
        )
        ax.scatter(
            reps_2d[non_binding_mask, 0],
            reps_2d[non_binding_mask, 1],
            c="blue",
            label="Non-binding",
            alpha=0.6,
            s=50,
        )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Representation Space (PCA)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()

    def plot_tsne(
        self,
        save_path: str = None,
        figsize: Tuple = (10, 8),
        perplexity: int = 30,
    ):
        """
        Plot t-SNE projection of representations.

        Args:
            save_path: where to save
            figsize: figure size
            perplexity: t-SNE perplexity parameter
        """
        if self.representations is None:
            print("No representations extracted. Call extract_representations() first.")
            return

        # t-SNE (slower but often better visualization)
        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(self.representations) // 3),
            random_state=42,
            n_iter=1000,
        )
        reps_2d = tsne.fit_transform(self.representations.numpy())

        fig, ax = plt.subplots(figsize=figsize)

        binding_mask = self.labels.numpy() == 1
        non_binding_mask = self.labels.numpy() == 0

        ax.scatter(
            reps_2d[binding_mask, 0],
            reps_2d[binding_mask, 1],
            c="red",
            label="Binding",
            alpha=0.6,
            s=50,
        )
        ax.scatter(
            reps_2d[non_binding_mask, 0],
            reps_2d[non_binding_mask, 1],
            c="blue",
            label="Non-binding",
            alpha=0.6,
            s=50,
        )

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Representation Space (t-SNE)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


def extract_attention_weights(
    model: nn.Module,
    H_geom: torch.Tensor,
    H_star: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract attention weights from cross-attention layer.

    Returns:
        attn_weights: (n_heads, N, N)
    """
    visualizer = AttentionMapVisualizer(model, device)
    attn_weights = visualizer.extract_attention(H_geom, H_star)
    return attn_weights


def visualize_cross_attention(
    model: nn.Module,
    H_geom: torch.Tensor,
    H_star: torch.Tensor,
    residue_ids: List[int] = None,
    save_dir: str = None,
    device: str = "cuda",
):
    """
    Visualize cross-attention for all heads.

    Args:
        model: DynaMo model
        H_geom: geometry path representation
        H_star: dynamics path representation
        residue_ids: residue indices for labels
        save_dir: where to save figures
        device: "cuda" or "cpu"
    """
    visualizer = AttentionMapVisualizer(model, device)
    attn_weights = visualizer.extract_attention(H_geom, H_star)

    print(f"Extracted attention weights shape: {attn_weights.shape}")

    # Plot all heads
    if save_dir:
        save_path = f"{save_dir}/cross_attention_all_heads.png"
    else:
        save_path = None

    visualizer.plot_head_comparison(attn_weights, save_path=save_path)


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    residue_ids: List[int] = None,
    head_idx: int = 0,
    save_path: str = None,
):
    """
    Simple function to plot a single attention heatmap.

    Args:
        attn_weights: (n_heads, N, N)
        residue_ids: residue indices
        head_idx: which head to plot
        save_path: where to save
    """
    visualizer = AttentionMapVisualizer(model=None)
    visualizer.plot_attention_heatmap(
        attn_weights,
        residue_ids=residue_ids,
        head_idx=head_idx,
        save_path=save_path,
    )


def feature_importance_analysis(
    model: nn.Module,
    test_loader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Analyze which input features are most important for predictions.

    Uses gradient-based feature importance (saliency).

    Returns:
        importance: dict of feature name → importance score
    """
    model.eval()

    # Feature names
    feature_names = [
        "PLM_embedding",
        "backbone_dihedral",
        "physicochemical",
        "solvent_exposure",
        "pmp_specific",
        "vectors",
    ]

    importance_scores = {name: 0.0 for name in feature_names}

    total_batches = 0

    for batch in test_loader:
        # Get input features
        H_static = batch["H_static"].to(device).requires_grad_(True)
        H_snapshots = batch["H_snapshots"].to(device)
        rmsf = batch["rmsf"].to(device)
        depth = batch["depth"].to(device)
        kd = batch["kd"].to(device)
        charge = batch["charge"].to(device)
        sasa = batch["sasa"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        logits, _ = model(
            H_static=H_static,
            H_snapshots=H_snapshots,
            rmsf=rmsf,
            depth=depth,
            kd=kd,
            charge=charge,
            sasa=sasa,
        )

        # Backward on loss
        loss = nn.BCEWithLogitsLoss()(logits.reshape(-1), targets.reshape(-1).float())
        loss.backward()

        # Compute saliency (gradient magnitude)
        if H_static.grad is not None:
            saliency = torch.abs(H_static.grad).mean(dim=(0, 2))  # (1280,) → scalar
            importance_scores["PLM_embedding"] += saliency.item()

        total_batches += 1

    # Average
    for key in importance_scores:
        importance_scores[key] /= total_batches

    return importance_scores
