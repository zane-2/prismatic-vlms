"""
siglip_vit.py
"""

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone
import torch

# Registry =>> Supported SigLIP Vision Backbones (from TIMM) =>> Note:: Using SigLIP w/ Patch = 14 (but SO400M Arch)
SIGLIP_VISION_BACKBONES = {
    "siglip-vit-b16-224px": "vit_base_patch16_siglip_224",
    "siglip-vit-b16-256px": "vit_base_patch16_siglip_256",
    "siglip-vit-b16-384px": "vit_base_patch16_siglip_384",
    "siglip-vit-so400m": "vit_so400m_patch14_siglip_224",
    "siglip-vit-so400m-384px": "vit_so400m_patch14_siglip_384",
    "video-siglip-vit-so400m-384px": "vit_so400m_patch14_siglip_384"
}


class SigLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 1) -> None:
        super().__init__(
            vision_backbone_id,
            SIGLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )

    
class VideoSigLIPViTBackbone(SigLIPViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 8) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
        )
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.num_frames, f"Expected input tensor to have {self.num_frames} frames in the second dimension, but got {x.shape[1]} frames instead.  Total shape was: {x.shape}"

        x = x.view(-1, *x.shape[2:])
        output_embeds = super().forward(x)
        return output_embeds.reshape(-1, self.num_frames * self.featurizer.patch_embed.num_patches, self.featurizer.embed_dim)
    
    @property
    def num_patches(self) -> int:
        return self.num_frames * self.featurizer.patch_embed.num_patches
