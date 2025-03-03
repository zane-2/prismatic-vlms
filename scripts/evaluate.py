import os
import csv
from tqdm import tqdm
import numpy as np
import random
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
## frames_path = '../NExT-OE/4924794333'

import decord
decord.bridge.set_bridge('torch')
import re

# return yes/no if there is a counting pattern in the generated text (for e.g., 1. 2. 3. ..) 
def is_counting_pattern(gen_text):
    pattern = r'\b\d+\.\s' # r'\b\d+[\.:]\s' # r'\b\d+[\.,]?\b'
    matches = re.findall(pattern, gen_text)
    return len(matches) > 0


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
    vlm.eval()

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


def eval_finetuned_prismatic_models_nextoe():
    cfg = GenerateConfig()
    cfg.model_path = 'runs/webvid+prism-clip+7b-webvid-train-45k-cluster-size=4-random-epochs=2-frames=4-gpus=2-021+stage-finetune+x7/checkpoints/step-005739-epoch-01-loss=2.3971.pt'
    
    vlm = set_up_prismatic_vlm(cfg)
    
    # evaluating on the NEXT-OE dataset
    NUM_FRAMES = 4
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

                image = [torchvision.transforms.functional.to_pil_image(vid_frames[i]) for i in range(NUM_FRAMES)]
                
                # building the prompts for vlm
                prompt_text = f"In: {question}\nOut: "

                # import pdb; pdb.set_trace()
                generated_text = vlm.generate(
                                    image,
                                    prompt_text,
                                    do_sample=cfg.do_sample,
                                    temperature=cfg.temperature,
                                    max_new_tokens=cfg.max_new_tokens,
                                    min_length=cfg.min_length,
                                )
                                
                vlm_answers[video_id][qid] = generated_text


    SAVE_PATH = f"../NExT-OE/results/webvid-train-45k-cluster-size=4-random-frames=4-epoch=1.json"
    with open(SAVE_PATH, 'w') as outfile:
        json.dump(vlm_answers, outfile, indent=2)



def eval_finetuned_prismatic_models_nextoe_faster():
    cfg = GenerateConfig()
    cfg.model_path = 'runs/webvid+webvid-train-45k-cluster-size=4-phi-3-dinosiglip-epochs=5-frames=4-gpus=4-028+stage-finetune+x7/checkpoints/step-014350-epoch-05-loss=4.2467.pt'
    
    vlm = set_up_prismatic_vlm(cfg)
    
    # evaluating on the NEXT-OE dataset
    NUM_FRAMES = 4
    RAW_VIDEO_FILES_PATH = '/vision/u/silsingh/NExT-QA/NExTVideo_test_videos'
    RAW_FRAMES_PATH = '/vision/u/silsingh/NExT-QA/NExTVideo_test_videos_frames=4'
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

                # video_path = os.path.join(RAW_VIDEO_FILES_PATH, f"{video_id}.mp4")
                # assert os.path.exists(video_path), f"{video_path} does not exist!"
            
                # vid_frames = sample_frames_from_video_path(video_path, NUM_FRAMES)
                # vid_frames = vid_frames.permute(0,3,1,2)   # .cpu()

                # image = [torchvision.transforms.functional.to_pil_image(vid_frames[i]) for i in range(NUM_FRAMES)]
                image = []
                for fr in os.listdir(os.path.join(RAW_FRAMES_PATH, video_id)):
                    image.append(Image.open(os.path.join(RAW_FRAMES_PATH, video_id, fr)).convert("RGB"))
                
                # building the prompts for vlm
                prompt_text = f"In: {question}\nOut: "

                generated_text = vlm.generate(
                                image,
                                prompt_text,
                                do_sample=cfg.do_sample,
                                temperature=cfg.temperature,
                                max_new_tokens=cfg.max_new_tokens,
                                min_length=cfg.min_length,
                            )
                                
                vlm_answers[video_id][qid] = generated_text


    SAVE_PATH = f"../NExT-OE/results/webvid-train-45k-cluster-size=4-phi-3-dinosiglip-frames=4-028-epoch=5.json"
    with open(SAVE_PATH, 'w') as outfile:
        json.dump(vlm_answers, outfile, indent=2)




def eval_finetuned_prismatic_models(model_path, webvid_val_path, save_path, sample_size=50, num_frames=4, save_eval_images=None):
    cfg = GenerateConfig()
    cfg.model_path = model_path
    vlm = set_up_prismatic_vlm(cfg)

    WEBVID_VIDEOS_PATH = "webvid"
    DEFAULT_PROMPT = "Describe what is happening in the video."
    with open(webvid_val_path, "r") as fp:
        val_videos = json.load(fp)

    val_subset = val_videos[:sample_size]
    eval_metadata = []

    for val_example in tqdm(val_subset):
        metadata = {}
        video_id = val_example["id"]
        metadata["id"] = video_id

        prompt = val_example["conversations"][0]["value"].replace("<image>\n", "")

        gt_caption = val_example["conversations"][1]["value"]
        metadata["gt_caption"] = gt_caption

        frames = val_example["frames"][:num_frames]
        frames_pil = []
        for frame in frames:
            frames_pil.append(Image.open(f"{WEBVID_VIDEOS_PATH}/{frame}").convert("RGB"))
        
        if save_eval_images:
            os.makedirs(save_eval_images, exist_ok=True)
            w,h = frames_pil[0].size
            grid = Image.new('RGB', size=(w*2, h*num_frames//2))
             
            for i,fr in enumerate(frames_pil):
                grid.paste(fr, box=(i%2*w, i//2*h))

            grid.save(os.path.join(save_eval_images, f"{video_id}.png"))

        import pdb; pdb.set_trace()
        prompt_text = f"Input: {prompt}\nOutput: "
        generated_text = vlm.generate(
                            frames_pil,
                            prompt_text,
                            do_sample=cfg.do_sample,
                            temperature=cfg.temperature,
                            max_new_tokens=cfg.max_new_tokens,
                            min_length=cfg.min_length,
                        )
        metadata["model_output"] = [
            {
                "prompt": prompt,
                "generated": generated_text
            }
        ]
        
        import pdb; pdb.set_trace()
        prompt_text = f"Input: {DEFAULT_PROMPT}\nOutput: "
        generated_text = vlm.generate(
                            frames_pil,
                            prompt_text,
                            do_sample=cfg.do_sample,
                            temperature=cfg.temperature,
                            max_new_tokens=cfg.max_new_tokens,
                            min_length=cfg.min_length,
                        )

        metadata["model_output"].append(
            {
                "prompt": DEFAULT_PROMPT,
                "generated": generated_text
            }
        )

        eval_metadata.append(metadata)
    
    with open(save_path, 'w') as outfile:
        json.dump(eval_metadata, outfile, indent=2)



def precompute_N_frames_NExT_OE(num_frames=16):
    RAW_VIDEO_FILES_PATH = '/vision/u/silsingh/NExT-QA/NExTVideo_test_videos'
    TEST_CSV_FILE_PATH = '/vision/u/silsingh/NExT-OE/test_data_nextoe/test.csv'
    SAVE_FRAMES_PATH = f'/vision/u/silsingh/NExT-QA/NExTVideo_test_videos_frames={num_frames}'
    os.makedirs(SAVE_FRAMES_PATH, exist_ok=True)

    with open(TEST_CSV_FILE_PATH, 'r') as fp:
        csv_reader = csv.reader(fp)

        for i, row in tqdm(enumerate(csv_reader)):
            if i > 0:
                video_id = row[1]
                question = row[5]
                qid = row[7]
                qtype = row[8]
                os.makedirs(os.path.join(SAVE_FRAMES_PATH, video_id), exist_ok=True)

                video_path = os.path.join(RAW_VIDEO_FILES_PATH, f"{video_id}.mp4")
                assert os.path.exists(video_path), f"{video_path} does not exist!"
            
                vid_frames = sample_frames_from_video_path(video_path, num_frames)
                vid_frames = vid_frames.permute(0,3,1,2)   # .cpu()

                frames = [torchvision.transforms.functional.to_pil_image(vid_frames[i]) for i in range(num_frames)]
                for idx,frame in enumerate(frames):
                    frame.save(os.path.join(SAVE_FRAMES_PATH, video_id, f"{str(idx).zfill(4)}.png"))
                


if __name__ == "__main__":
    ###### FIND COUNTING PATTERNS IN THE GENERATED TEXTS  ###########
    # output_path = "../NExT-OE/results/webvid-train-45k-diff-prompts-frames=4.json"   
    # with open(output_path, 'r') as fp:
    #     model_outputs = json.load(fp)

    # counting_present = 0
    # total = 0
    # video_ids = []
    # for video_id, outputs in tqdm(model_outputs.items()):
    #     for qid, gen_text in outputs.items():
    #         total += 1
    #         if is_counting_pattern(gen_text):
    #             counting_present += 1
    #             video_ids.append((video_id,qid))
        
    # print(video_ids)
    # print(f"Counting present: {counting_present}/{total}.")


    ### PRECOMPUTE N FRAMES FOR EACH NEXT-QA VIDEO
    # precompute_N_frames_NExT_OE()

    ########## EVALUATION OF FINETUNED MODELS #############
    # eval_finetuned_prismatic_models_nextoe()
    eval_finetuned_prismatic_models_nextoe_faster()

    # model_path = "runs/webvid+prism-clip+7b-webvid-train-45k-frames=4-gpus=4-epochs=2-001+stage-finetune+x7/checkpoints/step-022956-epoch-01-loss=2.0254.pt" # "webvid+prism-clip+7b-webvid-train-45k-diff-prompts-frames=4-gpus=4-epochs=2-002+stage-finetune+x7/checkpoints/step-022956-epoch-01-loss=2.1013.pt"
    # webvid_val_path = "webvid_val_5k_diff_prompts.json"
    # save_path = "eval_webvid_45k_single_prompt.json"
    # # save_eval_images = "finteuned_webvid_45k_eval_50_video_frames"
    # eval_finetuned_prismatic_models(model_path, webvid_val_path, save_path)


    # model_path = 'prism-dinosiglip+7b'
    # model_paths = [
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
    #     "prism-dinosiglip-224px-controlled+7b",
    #     "prism-dinosiglip-224px+7b"
    # ]
    # for model_path in tqdm(model_paths):
    #     evaluate_zero_shot(model_path)
    
    # question = 'is the baby old enough to converse'
    
    # image = []
    # frames_path = '../NExT-OE/4924794333'
    # for frame in os.listdir(frames_path):
    #     if not frame.startswith('.'):
    #         image.append(Image.open(os.path.join(frames_path, frame)).convert("RGB"))

    # print('vlm response:', generated_text)

