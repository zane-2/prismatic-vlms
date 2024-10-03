from prismatic import available_model_ids_and_names, available_model_ids, get_model_description
from pprint import pprint

# List all Pretrained VLMs (by HF Hub IDs)
pprint(available_model_ids())

# List all Pretrained VLMs with both HF Hub IDs AND semantically meaningful names from paper
pprint(available_model_ids_and_names())

# Print and return a targeted description of a model (by name or ID) 
#   =>> See `prismatic/models/registry.py` for explicit schema
description = get_model_description("Prism-DINOSigLIP 13B (Controlled)")