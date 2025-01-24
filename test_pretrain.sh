# torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "testing" \
#  --model.vision_backbone_id "clip-vit-l" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "vicuna-v15-7b" \
#  --model.finetune_per_device_batch_size 1 
#  "gemma-2b-instruct" \
#  --dataset.dataset_id "llava-v15" \ "video-llava-v15" \

# video-clip-vit-b
# "phi-3-instruct-4b"
# "letterbox" 
# llava-v15

# class Prism_7B_CLIP(Exp_7B_One_Stage):
#     model_id: str = "prism-clip+7b"
#     vision_backbone_id: str = "clip-vit-l-336px"
#     image_resize_strategy: str = "resize-naive"
#     llm_backbone_id: str = "llama2-7b-pure"
#     finetune_epochs: int = 2

# FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun ...
torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "prism-clip+7b-webvid-train-45k-diff-prompts-k=512-frames=4-gpus=4-epochs=2-009" \
  --dataset.type "webvid" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "video-clip-vit-l-336px" \
  --model.arch_specifier "no-align+gelu-mlp" \
  --model.finetune_global_batch_size 4 \
  --model.finetune_per_device_batch_size 1 \
  --model.num_frames 4 \
  --model.init_from_model "prism-clip+7b" \
  --model.repo_id "TRI-ML/prismatic-vlms" \
  --model.finetune_epochs 2

# prism-clip+7b-webvid-train-50kx4-qna-action-object-scene-temporal-frames=4-gpus=4-epochs=1-006, prism-clip+7b-webvid-train-45k-diff-prompts-k=2-frames=4-gpus=4-epochs=2-007
# "prism-clip+7b", "prism-llama3-instruct+8b+clip" 
# "RylanSchaeffer/prismatic-vlms", "TRI-ML/prismatic-vlms"
# llm backbone: "llama2-7b-pure" (prism-clip+7b), "phi-2-3b", "phi-3-instruct-4b"  "llama2-7b-pure"  llama3-8b-instruct
# vision_backbone_id: "dinosiglip-vit-so-384px", "clip-vit-l-336px", "video-clip-vit-l-336px"
# arch_speifier: "no-align+fused-gelu-mlp", "no-align+gelu-mlp"
# dataset: "llava-v15" | "video-llava-v15"
# resize op: "resize-naive" | "letterbox"
# dataset_id: "video-llava-v15" | "webvid"

# Run from the root of the repository
# torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
#   --model.type "one-stage+7b" \
#   --model.model_id "testing" \
#   --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#   --model.image_resize_strategy "letterbox" \
#   --model.llm_backbone_id "vicuna-v15-7b" 
