# CREATING THE WEBVID DATASET

import os
import csv
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import random

import ast

import decord
decord.bridge.set_bridge('torch')

BASE_PATH = "/simurgh/u/zanedurante/webvid/webvid"
SAVE_PATH = "/vision/u/silsingh/prismatic-vlms/webvid"
NUM_FRAMES = 8

def sample_frames_from_video_path(video_path, num_frames=NUM_FRAMES):
    # Load the video
    vr = decord.VideoReader(video_path)
    # Sample frames
    frame_indices = np.linspace(0, len(vr)-1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices)
    return frames


'''
data format:
{
    'id': '0000000000',
    'frames': ['/path/to/frame0', '/path/to/frame1', '/path/to/frame2', '/path/to/frame3', '/path/to/frame4', '/path/to/frame5', '/path/to/frame6', '/path/to/frame7'],
    'conversations': [
        {'from': 'human', 'value': '<image>\nDescribe what is happening in the video.'},
        {'from': 'gpt', 'value': '<caption>'}
    ]
}
'''

# webvid_metadata = []
# with open('webvid_50k.json', 'r') as fp:
#     webvid_metadata = json.load(fp)
# print(f"metadata loaded: {len(webvid_metadata)}")


### CREATING THE TRAIN/VAL SPLIT
# NUM_VAL = 5000
# print(len(webvid_metadata))

# webvid_train_subset = webvid_metadata[:-NUM_VAL]
# webvid_val_subset = webvid_metadata[-NUM_VAL:]
# print(len(webvid_train_subset), len(webvid_val_subset))
# print('total:', len(webvid_train_subset)+len(webvid_val_subset))

# with open('webvid_train_45k.json', 'w') as outfile:
#     json.dump(webvid_train_subset, outfile, indent=2)

# with open('webvid_val_5k.json', 'w') as outfile:
#     json.dump(webvid_val_subset, outfile, indent=2)

if __name__=="__main__":
    ###### EXTRACTING VIDEO IDS FOR THE 50K VIDEOS WE WOULD BE USING IN THE EXPERIMENTS ####
    # with open('webvid_50k.json', 'r') as fp:
    #     webvid_videos = json.load(fp)

    # webvid_video_ids = []
    # for vid in tqdm(webvid_videos):
    #     webvid_video_ids.append(
    #         {
    #             "id": vid["id"]
    #         }
    #     )

    # with open("webvid_50k_video_ids.json", 'w') as outfile:
    #     json.dump(webvid_video_ids, outfile, indent=2)
    

    ##### PARSE DIFFERENT TYPES OF QUESTIONS: {ACTION, OBJECT, SCENE, TEMPORAL} ###########
    # filename = "generated_qna/50k/video_questions_{}_n=10-start=0.txt"
    # with open(filename.format("action"), 'r') as fp:
    #     action_qna = fp.readlines()[0]
    #     action_dicts = ast.literal_eval(action_qna)

    # with open(filename.format("object"), 'r') as fp:
    #     object_qna = fp.readlines()[0]
    #     object_dicts = ast.literal_eval(object_qna)

    # with open(filename.format("scene"), 'r') as fp:
    #     scene_qna = fp.readlines()[0]
    #     scene_dicts = ast.literal_eval(scene_qna)

    # with open(filename.format("temporal"), 'r') as fp:
    #     temporal_qna = fp.readlines()[0]
    #     temporal_dicts = ast.literal_eval(temporal_qna)

    # print(len(action_dicts), len(object_dicts), len(scene_dicts), len(temporal_dicts))  # -> 100k
    
    # def get_video_id_from_url(url):
    #     return url.split('/')[-3]
    # video_ids_100k = [get_video_id_from_url(item['file_paths']) for item in action_dicts]

    # def updated_qna(qna_type_dicts, train_example, video_id):
    #     for qna_item in qna_type_dicts:
    #         if video_id == get_video_id_from_url(qna_item["file_paths"]):
    #             ques = qna_item["question"]
    #             ans = qna_item["answer"]
    #             new_dict = train_example.copy()
    #             new_dict["conversations"] = [
    #                 {
    #                     "from": "human",
    #                     "value": f"<image>\n{ques}"
    #                 },
    #                 {
    #                     "from": "gpt",
    #                     "value": f"{ans}"
    #                 }
    #             ]
    #             break

    #     return new_dict


    # with open('webvid_train_45k.json', 'r') as fp:
    #     training_examples = json.load(fp)
    # with open('webvid_val_5k.json', 'r') as fp:
    #     val_examples = json.load(fp)

    # webvid_train_45k_qna_types = []
    # webvid_val_5k_qna_types = []

    # for train_example in tqdm(training_examples):
    #     video_id = train_example["id"]
    #     if video_id in video_ids_100k:
    #         webvid_train_45k_qna_types.append(updated_qna(action_dicts, train_example, video_id))
    #         webvid_train_45k_qna_types.append(updated_qna(object_dicts, train_example, video_id))
    #         webvid_train_45k_qna_types.append(updated_qna(scene_dicts, train_example, video_id))
    #         webvid_train_45k_qna_types.append(updated_qna(temporal_dicts, train_example, video_id))
            
    # for val_example in tqdm(val_examples):
    #     video_id = val_example["id"]
    #     if video_id in video_ids_100k:
    #         webvid_val_5k_qna_types.append(updated_qna(action_dicts, val_example, video_id))
    #         webvid_val_5k_qna_types.append(updated_qna(object_dicts, val_example, video_id))
    #         webvid_val_5k_qna_types.append(updated_qna(scene_dicts, val_example, video_id))
    #         webvid_val_5k_qna_types.append(updated_qna(temporal_dicts, val_example, video_id))

    # # assert len(webvid_train_45k_qna_types) == 4 * 30634, f"check training examples size! {len(webvid_train_45k_qna_types)}" 
    # TRAIN_SAVE_PATH = "webvid_train_45k_qna.json"       
    # with open(TRAIN_SAVE_PATH, 'w') as outfile:
    #     json.dump(webvid_train_45k_qna_types, outfile, indent=2)

    # # assert len(webvid_val_5k_qna_types) == 4 * 3341, f"check val examples size! {len(webvid_val_5k_qna_types)}" 
    # VAL_SAVE_PATH = "webvid_val_5k_qna.json"
    # with open(VAL_SAVE_PATH, 'w') as outfile:
    #     json.dump(webvid_val_5k_qna_types, outfile, indent=2)


    # counter = 0
    # for example in tqdm(training_examples):
    #     vid_id_45k = example["id"]
    #     if vid_id_45k in video_ids_100k:
    #         counter += 1

    # print(f"(train) Total videos also in the 100k set: {counter}/{len(training_examples)}")  # 30634/45912

    # counter = 0
    # for example in tqdm(val_examples):
    #     vid_id_5k = example["id"]
    #     if vid_id_5k in video_ids_100k:
    #         counter += 1

    # print(f"(val) Total videos also in the 100k set: {counter}/{len(val_examples)}")  # 3341/5000
    



    ###### DIVERSIFY PROMPTS IN THE WEBVID DATASET ###################################
    k = 512
    with open('prompt_list.txt', 'r') as fp:
        prompts = fp.readlines()
    sampled_prompts = random.sample(prompts, k-1)
    sampled_prompts = [p.strip().replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'") for p in sampled_prompts]
    # p1 = random.choice(prompts).strip().replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'")
    # print(f"Using the other prompt: {p1}")
    sampled_prompts.append("Describe what is happening in the video.")

    with open('dataset_splits/webvid_train_45k.json', 'r') as fp:
        webvid_train_45k = json.load(fp)

    webvid_train_45k_diff_prompts = []
    for train_datapoint in tqdm(webvid_train_45k):
        prompt = random.choice(sampled_prompts)
        new_datapoint = train_datapoint
        new_datapoint["conversations"][0]["value"] = f"<image>\n{prompt}"
        webvid_train_45k_diff_prompts.append(new_datapoint)

    with open(f'dataset_splits/webvid_train_45k_diff_prompts_k={k}.json', 'w') as outfile:
        json.dump(webvid_train_45k_diff_prompts, outfile, indent=2)


    with open('dataset_splits/webvid_val_5k.json', 'r') as fp:
        webvid_val_5k = json.load(fp)

    webvid_val_5k_diff_prompts = []
    for val_datapoint in tqdm(webvid_val_5k):
        prompt = random.choice(sampled_prompts)
        new_datapoint = val_datapoint
        new_datapoint["conversations"][0]["value"] = f"<image>\n{prompt}"
        webvid_val_5k_diff_prompts.append(new_datapoint)

    with open(f'dataset_splits/webvid_val_5k_diff_prompts_k={k}.json', 'w') as outfile:
        json.dump(webvid_val_5k_diff_prompts, outfile, indent=2)
    

        

    ########## PREPROCESS WEBVID-50K SUBSET ###########################################
    # frames = sample_frames_from_video_path("/simurgh/u/zanedurante/webvid/webvid/videos/0007/4541360.mp4")
    # import pdb; pdb.set_trace()

    # csv_files_path = os.path.join(BASE_PATH, 'partitions')
    # videos_path = os.path.join(BASE_PATH, 'videos')
    # partition_videos_path = os.listdir(videos_path)
    
    # for partition in tqdm(partition_videos_path[1:]):
    #     csv_filepath = os.path.join(csv_files_path, f"{partition}.csv")
    #     df = pd.read_csv(csv_filepath)
    #     video_files_path = os.path.join(videos_path, partition)
    #     for video_file in tqdm(os.listdir(video_files_path)):
    #         video_id = video_file.split('.mp4')[0]
    #         video_file_path = os.path.join(video_files_path, video_file)
    #         # extract 8 frames and save it 
    #         try:
    #             video_frames = sample_frames_from_video_path(video_file_path)
    #             # save these frames at the SAVE_PATH
    #             frame_paths = []
    #             frame_save_path = os.path.join(SAVE_PATH, video_id)
    #             os.makedirs(frame_save_path, exist_ok=True)
    #             for fr in range(video_frames.shape[0]):
    #                 np_frame = video_frames[fr].numpy().astype(np.uint8)
    #                 pil_frame = Image.fromarray(np_frame)
    #                 frame_path = os.path.join(video_id, f"{str(fr).zfill(4)}.png")
    #                 pil_frame.save(os.path.join(SAVE_PATH, frame_path))
    #                 frame_paths.append(frame_path)
                
    #             vid_metadata = df[df['videoid'] == int(video_id)]  # all rows that match video_id (just 1!)

    #             video_data = {
    #                 'id': video_id,
    #                 'frames': frame_paths,
    #                 'conversations': [
    #                     {'from': 'human', 'value': '<image>\nDescribe what is happening in the video.'},
    #                     {'from': 'gpt', 'value': vid_metadata['name'].tolist()[0]}
    #                 ]
    #             }

    #             webvid_metadata.append(video_data)
    #         except Exception:
    #             with open('webvid_50k.json', 'w') as outfile:
    #                 json.dump(webvid_metadata, outfile, indent=2)
    #             continue

    #     with open('webvid_50k.json', 'w') as outfile:
    #         json.dump(webvid_metadata, outfile, indent=2)
    
    # with open('webvid_50k.json', 'w') as outfile:
    #     json.dump(webvid_metadata, outfile, indent=2)
