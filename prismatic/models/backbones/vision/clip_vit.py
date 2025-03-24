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
    "video-clip-vit-b": "vit_base_patch16_clip_224.openai",
    "video-clip-vit-l": "vit_large_patch14_clip_224.openai",
    "video-clip-vit-l-336px": "vit_large_patch14_clip_336.openai",
}


# [IMPORTANT] By Default, TIMM initialized OpenAI CLIP models with the standard GELU activation from PyTorch.
#             HOWEVER =>> Original OpenAI models were trained with the quick_gelu *approximation* -- while it's
#                         a decent approximation, the resulting features are *worse*; this was a super tricky bug
#                         to identify, but luckily there's an easy fix (`override_act_layer`)
class CLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 1) -> None:
        super().__init__(
            vision_backbone_id,
            CLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            override_act_layer="quick_gelu" if CLIP_VISION_BACKBONES[vision_backbone_id].endswith(".openai") else None,
        )


class VideoCLIPViTBackbone(CLIPViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 8) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.num_frames = num_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the input tensor to transform from (B, T, C, H, W) to (B * T, C, H, W)

        assert x.shape[1] == self.num_frames, f"Expected input tensor to have {self.num_frames} frames in the second dimension, but got {x.shape[1]} frames instead.  Total shape was: {x.shape}"
        # self.eval()  # Disable dropout and other stochastic layers to compare embeddings

        #output_embed_first_frame = super().forward(x[:, 0])
        #output_embed_first_frame_2 = super().forward(x[:, 0])
        #print("Values before reshape:", x[0][0][0][100][100:104])
        #print("Values before reshape:", x[0][1][0][100][100:104])
        #print("Values before reshape:", x[0][2][0][100][100:104])
        #print("Values before reshape:", x[0][3][0][100][100:104])
        x = x.view(-1, *x.shape[2:])
        #print("Values after reshape:", x[0][0][100][100:104])
        #print("Values after reshape:", x[1][0][100][100:104])
        #print("Values after reshape:", x[2][0][100][100:104])
        #print("Values after reshape:", x[3][0][100][100:104])
        #import pdb; pdb.set_trace()

        output_embeds = super().forward(x)
        #print("First frame embeds (1)", output_embed_first_frame[0][0][0:4])
        #print("First frame embeds (2)", output_embed_first_frame_2[0][0][0:4])
        #print("All frames embeds", output_embeds[0][0][0:4])
        # Reshape the output tensor to transform from (B * T, H * W, C) to (B, T * H * W, C)
        # TODO: Figure out why this is different?????
        #import pdb; pdb.set_trace()
        return output_embeds.reshape(-1, self.num_frames * self.featurizer.patch_embed.num_patches, self.featurizer.embed_dim)
    
    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches * self.num_frames
