# '''
# 
# '''
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "prism-clip+7b-webvid-train-45k-cluster-size=4-worse-than-random-epochs=2-frames=4-gpus=2-027" \
  --dataset.type "webvid" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "video-clip-vit-l-336px" \
  --model.arch_specifier "no-align+gelu-mlp" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --model.num_frames 4 \
  --model.init_from_model "prism-clip+7b" \
  --model.repo_id "TRI-ML/prismatic-vlms" \
  --model.finetune_epochs 5