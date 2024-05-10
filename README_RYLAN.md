Create a file `.hf_token` in the project root directory with your HuggingFace token.


```bash
# Run from the root of the repository
export CUDA_VISIBLE_DEVICES=5,6
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "rylan_attempt_1" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "letterbox" \
  --model.llm_backbone_id "vicuna-v15-7b" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"
```
