"""
BBox-only InstanceDiffusion grounding projection.

This is a simplified version of INSTDIFFTextBoundingboxProjection
that only uses bounding boxes, removing point, scribble, polygon, and segmentation branches.

The code is identical to the original diffusers implementation,
only removing unused branches (point, scribble, polygon, seg).
"""

import torch
import torch.nn as nn

# Use the original diffusers Fourier bbox embedding helper.  The symbol was
# renamed across diffusers versions, so keep both import paths explicit.
try:
    from diffusers.models.embeddings import get_fourier_embeds
except ImportError:
    from diffusers.models.embeddings import get_fourier_embeds_from_boundingbox as get_fourier_embeds


class INSTDIFFTextBoundingboxProjectionBBoxOnly(nn.Module):
    """
    Simplified InstanceDiffusion grounding projection using only bounding boxes.

    This is a bbox-only version of INSTDIFFTextBoundingboxProjection,
    removing point, scribble, polygon, and segmentation branches.

    Compatible with UniFusion (bbox-only) weight structure using linears_list.

    Parameters:
        positive_len (`int`, defaults to 768): Dimension of text embeddings.
        mid_dim (`int`, defaults to 3072): Hidden dimension of MLP.
        out_dim (`int`, defaults to 768): Output dimension.
        fourier_freqs (`int`, defaults to 16): Number of Fourier frequencies for position encoding.
    """

    def __init__(
        self,
        positive_len: int = 768,
        mid_dim: int = 3072,
        out_dim: int = 768,
        fourier_freqs: int = 16,
    ):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim
        self.fourier_embedder_dim = fourier_freqs

        # position_dim: 2 (sin/cos) * 4 (xyxy) * fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        # Use linears_list to match UniFusion weight structure
        input_dim = self.positive_len + self.position_dim
        self.linears_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, out_dim),
            )
        ])

        # Learnable null embeddings
        self.null_positive_feature = nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            boxes: [B, N, 4] normalized bounding boxes (x1, y1, x2, y2)
            masks: [B, N] validity mask (1=valid, 0=padding)
            positive_embeddings: [B, N, positive_len] text embeddings per instance

        Returns:
            objs: [B, N, out_dim] grounding tokens
        """
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)  # [B, N, 1]

        # Fourier embedding for bbox positions
        xyxy_embedding = get_fourier_embeds(self.fourier_embedder_dim, boxes)  # [B, N, position_dim]

        # Learnable null embeddings
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # Replace padding with null embeddings
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # Concatenate text and position embeddings, then project
        objs = self.linears_list[0](torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

        return objs
