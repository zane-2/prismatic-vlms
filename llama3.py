import requests
import torch

from PIL import Image
from pathlib import Path
import json

from huggingface_hub import hf_hub_download
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlms import PrismaticVLM

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_id = "llama3-instruct+8b+clip"
repo_id = "RylanSchaeffer/prismatic-vlms"

config_json = hf_hub_download(repo_id=repo_id, filename=f"{model_id}/config.json")
checkpoint_pt = hf_hub_download(repo_id=repo_id, filename=f"{model_id}/checkpoints/latest-checkpoint.pt")

with open(config_json, "r") as f:
    model_cfg = json.load(f)["model"]

vision_backbone, image_transform = get_vision_backbone_and_transform(
    model_cfg["vision_backbone_id"],
    model_cfg["image_resize_strategy"],
    model_cfg.get("num_frames", None)
)
llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
    model_cfg["llm_backbone_id"],
    llm_max_length=model_cfg.get("llm_max_length", 2048),
    hf_token=hf_token,
    inference_mode=True,
)
vlm = PrismaticVLM.from_pretrained(
    checkpoint_pt,
    model_cfg["model_id"],
    vision_backbone,
    llm_backbone,
    arch_specifier=model_cfg["arch_specifier"],
)
vlm.to(device, dtype=torch.bfloat16)


# 
image = Image.open('./0000.png').convert("RGB")
prompt_text = "Input: Describe what is happening in the image.\nOutput:"
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=False,
    temperature=1.0,
    max_new_tokens=512,
    min_length=1,
)

print(generated_text)
import pdb; pdb.set_trace()

