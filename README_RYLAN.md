Create a file `.hf_token` in the project root directory with your HuggingFace token.


## Basic Training Run

The below works on 2 A100s:

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


## Gemma 2B IT

```bash
# Run from the root of the repository
export CUDA_VISIBLE_DEVICES=5,6
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "gemma-instruct+2b+dinosiglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --model.llm_backbone_id "gemma-2b-instruct" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"
```


## Gemma 8B IT


## Mistral 7B Instruct v0.2


```bash
# Run from the root of the repository
export CUDA_VISIBLE_DEVICES=5,6
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "mistral-instruct-v0.2+7b+dinosiglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --model.llm_backbone_id "mistral-v0.2-7b-instruct" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"
```
