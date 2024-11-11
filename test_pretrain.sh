#torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "testing" \
#  --model.vision_backbone_id "clip-vit-l" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "vicuna-v15-7b" \
#  --model.finetune_per_device_batch_size 1 
#"gemma-2b-instruct" \
#  --dataset.dataset_id "llava-v15" \ "video-llava-v15" \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "testing" \
  --dataset.type "video-llava-v15" \
  --model.image_resize_strategy "letterbox" \
  --model.llm_backbone_id "phi-3-instruct-4b" \
  --model.vision_backbone_id "video-clip-vit-b" \
  --model.finetune_global_batch_size 1 \
  --model.finetune_per_device_batch_size 1 