# '''
# phi-3-4b + DinoSIGLIP
# total input frames = 4, videos per cluster = 4
# '''

torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "prism-clip+7b-webvid-train-45k-cluster-size=4-epochs=5-frames=4-gpus=2-002" \
  --dataset.type "webvid" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "phi-3-instruct-4b" \
  --model.vision_backbone_id "video-dinosiglip-vit-so-384px" \
  --model.arch_specifier "no-align+gelu-mlp" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --model.num_frames 4 \
  --model.init_from_model "phi-instruct-3+4b+dinosiglip" \
  --model.repo_id "RylanSchaeffer/prismatic-vlms" \
  --model.finetune_epochs 5 \
  --dataset.type "webvid" \
  --dataset.finetune_stage_components '["/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_train_45k_cluster_size=4.json", "/vision/u/silsingh/prismatic-vlms/webvid_num_frames=1"]' \
  --val_dataset.type "webvid_val" \
  --val_dataset.finetune_stage_components '["/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_val_5k_cluster_size=4.json", "/vision/u/silsingh/prismatic-vlms/webvid_num_frames=1"]' \
  