import os
import csv
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# os.environ["HUGGINGFACE_HUB_CACHE"] = '/vision/u/silsingh/.cache/huggingface/hub'
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import requests
import torch
import torchvision
from PIL import Image

import json

from prismatic import load
from prismatic.overwatch import initialize_overwatch

import cv2
# import cProfile

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

## this path has a list of frames for the video
# frames_path = '../NExT-OE/4924794333'

import decord
decord.bridge.set_bridge('torch')


def sample_frames_from_video_path(video_path, num_frames=32):
    # Load the video
    vr = decord.VideoReader(video_path)
    # Sample frames
    frame_indices = np.linspace(0, len(vr)-1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices)
    return frames

@dataclass
class GenerateConfig:
    # fmt: off
    model_path: Union[str, Path] = (                                    # Path to Pretrained VLM (on disk or HF Hub)
        "prism-dinosiglip+7b"
    )

    # HF Hub Credentials (required for Gated Models like Llama-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

    # fmt: on


@draccus.wrap()
def set_up_prismatic_vlm(cfg: GenerateConfig):
    overwatch.info(f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print('device:', device) # cuda

    # Load the pretrained VLM --> uses default `load()` function
    vlm = load(cfg.model_path, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)

    return vlm

def evaluate_on_random_baseline():
    cfg = GenerateConfig()
    cfg.model_path = '/vision/u/zanedurante/ckpts/new_vlm/checkpoints/test_ckpt.pt'  # RANDOM BASELINE
    
    vlm = set_up_prismatic_vlm(cfg)
    prompt_builder = vlm.get_prompt_builder()
    
    # evaluating on the NEXT-OE dataset
    NUM_FRAMES = 8
    RAW_VIDEO_FILES_PATH = '/vision/u/silsingh/NExT-QA/NExTVideo_test_videos'
    TEST_CSV_FILE_PATH = '/vision/u/silsingh/NExT-OE/test_data_nextoe/test.csv'

    vlm_answers = defaultdict(dict)

    # answers["video_id"] = {"qid": "vlm generated answer"}
    with open(TEST_CSV_FILE_PATH, 'r') as fp:
        csv_reader = csv.reader(fp)

        for i, row in tqdm(enumerate(csv_reader)):
            if i > 0:
                video_id = row[1]
                question = row[5]
                qid = row[7]
                qtype = row[8]

                video_path = os.path.join(RAW_VIDEO_FILES_PATH, f"{video_id}.mp4")
                assert os.path.exists(video_path), f"{video_path} does not exist!"
            
                vid_frames = sample_frames_from_video_path(video_path, NUM_FRAMES)
                vid_frames = vid_frames.permute(0,3,1,2)   # .cpu()

                # print(type(vid_frames), vid_frames.shape)
                # os.makedirs(VID_ID, exist_ok=True)
                image = [torchvision.transforms.functional.to_pil_image(vid_frames[i]) for i in range(NUM_FRAMES)]
                # for i in range(NUM_FRAMES):
                #     frame = vid_frames[i]
                #     pil_img = 
                #     image.append(pil_img)

                # building the prompts for vlm
                prompt_builder.add_turn(role="human", message=question)
                prompt_text = prompt_builder.get_prompt()

                generated_text = vlm.generate(
                                    image,
                                    prompt_text,
                                    do_sample=cfg.do_sample,
                                    temperature=cfg.temperature,
                                    max_new_tokens=cfg.max_new_tokens,
                                    min_length=cfg.min_length,
                                )
                                
                prompt_builder.add_turn(role="gpt", message=generated_text)
                vlm_answers[video_id][qid] = generated_text


    SAVE_PATH = f"../NExT-OE/results/vlm_v0.json"
    with open(SAVE_PATH, 'w') as outfile:
        json.dump(vlm_answers, outfile, indent=2)



def extract_center_frame(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # # Check if the video was opened successfully
    # if not video.isOpened():
    #     print(f"Error: Cannot open video file {video_path}")
    #     return None

    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the center frame index
    center_frame_idx = total_frames // 2

    # Set the video to the center frame
    video.set(cv2.CAP_PROP_POS_FRAMES, center_frame_idx)

    # Read the frame
    ret, frame = video.read()
    if ret:
        # Save the frame as an image
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        # cv2.imwrite(save_path, frame)
        # print(f"Center frame saved to {save_path}")
    else:
        print("Error: Could not read the frame.")
        return None

    # Release the video object
    video.release()
    return pil_image  # PIL RGB image


def evaluate_zero_shot(model_path='prism-dinosiglip+7b'):
    cfg = GenerateConfig()
    cfg.model_path = model_path
    
    vlm = set_up_prismatic_vlm(cfg)
    # prompt_builder = vlm.get_prompt_builder()  # TODO an issue with the prompt builder is that it builds up the context from previous questions that are passed to the VLM (not desirable)
    
    # evaluating on the NEXT-OE dataset
    # NUM_FRAMES = 8
    RAW_VIDEO_FILES_PATH = '/vision/u/silsingh/NExT-QA/NExTVideo_test_videos'
    TEST_CSV_FILE_PATH = '/vision/u/silsingh/NExT-OE/test_data_nextoe/test.csv'

    vlm_answers = defaultdict(dict)

    # answers["video_id"] = {"qid": "vlm generated answer"}
    with open(TEST_CSV_FILE_PATH, 'r') as fp:
        csv_reader = csv.reader(fp)

        for i, row in tqdm(enumerate(csv_reader)):
            if i > 0:
                video_id = row[1]
                question = row[5]
                qid = row[7]
                qtype = row[8]

                video_path = os.path.join(RAW_VIDEO_FILES_PATH, f"{video_id}.mp4")
                assert os.path.exists(video_path), f"{video_path} does not exist!"
            
                image = extract_center_frame(video_path)

                # building the prompts for vlm
                # prompt_builder.add_turn(role="human", message=question)
                # prompt_text = prompt_builder.get_prompt()
                prompt_text = f"In: {question}\nOut: " # question [TODO] prompts??
                # import pdb; pdb.set_trace()
                generated_text = vlm.generate(
                                    image,
                                    prompt_text,
                                    do_sample=cfg.do_sample,
                                    temperature=cfg.temperature,
                                    max_new_tokens=cfg.max_new_tokens,
                                    min_length=cfg.min_length,
                                )
                # import pdb; pdb.set_trace()
                # prompt_builder.add_turn(role="gpt", message=generated_text)
                vlm_answers[video_id][qid] = generated_text


    SAVE_PATH = f"../NExT-OE/results/{model_path}.json"
    with open(SAVE_PATH, 'w') as outfile:
        json.dump(vlm_answers, outfile, indent=2)


if __name__ == "__main__":
    # model_path = 'prism-dinosiglip+7b'
    model_paths = [
        # "prism-clip-controlled+7b",
        # "prism-clip-controlled+13b",
        # "prism-clip+7b",
        # "prism-clip+13b",
        # "prism-siglip-controlled+7b",
        # "prism-siglip-controlled+13b",
        # "prism-siglip+7b",
        # "prism-siglip+13b",
        # "prism-dinosiglip-controlled+7b",
        # "prism-dinosiglip-controlled+13b",
        # "prism-dinosiglip+7b",
        # "prism-dinosiglip+13b",
        "prism-dinosiglip-224px-controlled+7b",
        "prism-dinosiglip-224px+7b"
    ]
    for model_path in tqdm(model_paths):
        evaluate_zero_shot(model_path)
    
    # question = 'is the baby old enough to converse'
    
    # image = []
    # frames_path = '../NExT-OE/4924794333'
    # for frame in os.listdir(frames_path):
    #     if not frame.startswith('.'):
    #         image.append(Image.open(os.path.join(frames_path, frame)).convert("RGB"))

    # print('vlm response:', generated_text)

