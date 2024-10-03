"""
clip_vit.py
"""

import torch

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone


# Registry =>> Supported CLIP Vision Backbones (from TIMM)
CLIP_VISION_BACKBONES = {
    "clip-vit-b": "vit_base_patch16_clip_224.openai",
    "clip-vit-l": "vit_large_patch14_clip_224.openai",
    "clip-vit-l-336px": "vit_large_patch14_clip_336.openai",
}


# [IMPORTANT] By Default, TIMM initialized OpenAI CLIP models with the standard GELU activation from PyTorch.
#             HOWEVER =>> Original OpenAI models were trained with the quick_gelu *approximation* -- while it's
#                         a decent approximation, the resulting features are *worse*; this was a super tricky bug
#                         to identify, but luckily there's an easy fix (`override_act_layer`)
class CLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            CLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            override_act_layer="quick_gelu" if CLIP_VISION_BACKBONES[vision_backbone_id].endswith(".openai") else None,
        )


class VideoCLIPViTBackbone(CLIPViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 16) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.num_frames = num_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the input tensor to transform from (B, T, C, H, W) to (B * T, C, H, W)
        x = x.view(-1, *x.shape[2:])
        output_embeds = super().forward(x)
        import pdb; pdb.set_trace() # Ensure that the output_embeds tensor has the correct shape
        # Reshape the output tensor to transform from (B * T, H * W, C) to (B, T * H * W, C)
        return output_embeds.view(-1, self.num_frames * self.featurizer.patch_embed.num_patches, self.featurizer.embed_dim)
    
    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches * self.num_frames
