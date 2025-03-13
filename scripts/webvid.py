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
import glob

import decord
decord.bridge.set_bridge('torch')

BASE_PATH = "/simurgh/u/zanedurante/webvid/webvid"
SAVE_PATH = "/vision/u/silsingh/prismatic-vlms/webvid"
NUM_FRAMES = 4

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
    ####### MANUALLY EXTRACTING THE FRAMES FOR SOME VIDEOS
    # error_in_videos = ['1021460734', '1571302', '2439923', '1013447276', '3502337', '5965364', '1011792236']
    # VIDEO_ORIG_PATH = "/simurgh/u/zanedurante/webvid/webvid/videos/**/*.mp4"
    # orig_video_files = glob.glob(VIDEO_ORIG_PATH, recursive=True)
    # FRAMES_SAVE_PATH = "webvid_num_frames=16"
    # with open("dataset_splits/webvid_train_45k_num_frames=16.json", "r") as fp:
    #     frames16_metadata = json.load(fp)
    # with open("dataset_splits/webvid_train_45k.json", "r") as fp:
    #     orig_metadata = json.load(fp)

    # new_metadata = []
    # for el in tqdm(orig_metadata):
    #     new_el = el
    #     new_el["frames"] = [f"{el['id']}/{str(i).zfill(4)}.png" for i in range(16)]
    #     new_metadata.append(new_el)

    # with open("dataset_splits/webvid_train_45k_num_frames=16.json", "w") as outfile:
    #     json.dump(new_metadata, outfile, indent=2)


    # for vid_id in error_in_videos:
    #     # orig_video_path = list(filter(lambda x: x.endswith(f"{vid_id}.mp4"), orig_video_files))[0] # hopefully just 1 video matching this ID
    #     orig_video_path = f"{vid_id}.mp4"
    #     print(orig_video_path)
    #     try:
    #         frames = sample_frames_from_video_path(orig_video_path, 16)
    #         for fr in range(frames.shape[0]):
    #             np_frame = frames[fr].numpy().astype(np.uint8)
    #             pil_frame = Image.fromarray(np_frame)
    #             frame_path = os.path.join(vid_id, f"{str(fr).zfill(4)}.png")
                
    #             pil_frame.save(os.path.join(FRAMES_SAVE_PATH, frame_path))
    #             # frame_paths.append(frame_path)
    #     except Exception as e:
    #         print(e)
            






    ####### CLEANED CAPTIONS
    # CLEANED_CAPTIONS_PATH = "/simurgh/u/akhatua/long_context_benchmark/webvid_enriched_captions"
    # TRAIN_CAPTIONS_PATH = os.path.join(CLEANED_CAPTIONS_PATH, "webvid_train_45k_cleaned_captions.json")
    # VAL_CAPTIONS_PATH = os.path.join(CLEANED_CAPTIONS_PATH, "webvid_val_5k_cleaned_captions.json")
    
    # with open(TRAIN_CAPTIONS_PATH, 'r') as fp:
    #     TRAIN_CAPTIONS = json.load(fp)
    # with open(VAL_CAPTIONS_PATH, 'r') as fp:
    #     VAL_CAPTIONS = json.load(fp)


    # with open('dataset_splits/webvid_train_45k_cluster_size=4.json', 'r') as fp:
    #     train_metadata = json.load(fp)

    # with open('dataset_splits/webvid_val_5k_cluster_size=4.json', 'r') as fp:
    #     val_metadata = json.load(fp)

    
    # train
    # cleaned_train_metadata = []
    # for data in tqdm(train_metadata):
    #     ids = [entry.strip() for entry in data["id"][:-1].split(',')]
    #     joined_cleaned_captions = ""
    #     for vid_id in ids:
    #         # import pdb; pdb.set_trace()
    #         try:
    #             cleaned_entry = list(filter(lambda x: x["video_id"] == vid_id, list(TRAIN_CAPTIONS.values())))[0]
    #             joined_cleaned_captions += cleaned_entry["cleaned_caption"]+". "
    #         except Exception as e:
    #             print('ids:', ids)
    #             print('vid_id:', vid_id)
    #             print('filter list:', list(filter(lambda x: x["video_id"] == vid_id, list(TRAIN_CAPTIONS.values()))))
        
    #     cleaned_data = data
    #     cleaned_data["conversations"][-1]["value"] = joined_cleaned_captions.strip()
    #     cleaned_train_metadata.append(cleaned_data)

    # with open("dataset_splits/webvid_train_45k_cluster_size=4_cleaned_captions.json", 'w') as outfile:
    #     json.dump(cleaned_train_metadata, outfile, indent=2)


    # # val
    # cleaned_val_metadata = []
    # for data in tqdm(val_metadata):
    #     ids = [entry.strip() for entry in data["id"][:-1].split(',')]
    #     joined_cleaned_captions = ""
    #     for vid_id in ids:
    #         cleaned_entry = list(filter(lambda x: x["video_id"] == vid_id, list(VAL_CAPTIONS.values())))[0]
    #         joined_cleaned_captions += cleaned_entry["cleaned_caption"]+". "
        
    #     cleaned_data = data
    #     cleaned_data["conversations"][-1]["value"] = joined_cleaned_captions.strip()
    #     cleaned_val_metadata.append(cleaned_data)

    # with open("dataset_splits/webvid_val_5k_cluster_size=4_cleaned_captions.json", 'w') as outfile:
    #     json.dump(cleaned_val_metadata, outfile, indent=2)

    


    ##### CLUSTER VIDEOS RANDOMLY - 4 PER CLUSTER
    # FRAMES_SAVE_PATH = "webvid_cluster_size=4_random"
    
    # METADATA = "dataset_splits/webvid_train_45k.json"
    # with open(METADATA, "r") as fp:
    #     train_metadata = json.load(fp)
    
    # TRAIN_EXAMPLES_SIZE = len(train_metadata) # 45912, 5000
    # VIDS_PER_CLUSTER = 4
    # NUM_CLUSTERS = TRAIN_EXAMPLES_SIZE // VIDS_PER_CLUSTER
    # NUM_FRAMES = 4

    # train_video_indices = list(range(TRAIN_EXAMPLES_SIZE))  # [0, 1, ... N-1]
    # random.shuffle(train_video_indices)  # shuffle([0, 1, ... N-1])

    # VIDEO_ORIG_PATH = "/simurgh/u/zanedurante/webvid/webvid/videos/**/*.mp4"
    # orig_video_files = glob.glob(VIDEO_ORIG_PATH, recursive=True)

    # training_samples = []
    # for cluster_idx in tqdm(range(NUM_CLUSTERS)):
    #     videos_in_cluster = train_video_indices[cluster_idx*VIDS_PER_CLUSTER: (cluster_idx+1)*VIDS_PER_CLUSTER]  # chunk of 4
    #     # videos_in_cluster = list(filter(lambda x: x["cluster"] == str(cluster_idx), train_metadata))
    #     # assert len(videos_in_cluster) == VIDS_PER_CLUSTER, f"more videos in cluster than anticipated! {videos_in_cluster}"

    #     frame_paths = []
    #     combined_video_ids = ""
    #     joined_caption = ""
    #     for video_idx in videos_in_cluster:
    #         video = train_metadata[video_idx]
    #         video_id = video["id"]
    #         combined_video_ids += video_id+", "
    #         caption = video["conversations"][-1]["value"]
    #         joined_caption += caption+". "

    #         orig_video_path = list(filter(lambda x: x.endswith(f"{video_id}.mp4"), orig_video_files))[0] # ideally just 1 video matching this ID
    #         frames = sample_frames_from_video_path(orig_video_path, NUM_FRAMES//VIDS_PER_CLUSTER)

    #         for fr in range(frames.shape[0]):
    #             np_frame = frames[fr].numpy().astype(np.uint8)
    #             pil_frame = Image.fromarray(np_frame)
    #             frame_path = os.path.join(video_id, f"{str(fr).zfill(4)}.png")
    #             os.makedirs(os.path.join(FRAMES_SAVE_PATH, video_id), exist_ok=True)
    #             pil_frame.save(os.path.join(FRAMES_SAVE_PATH, frame_path))
    #             frame_paths.append(frame_path)

        
    #     training_samples.append({
    #         "id": combined_video_ids.strip(),
    #         "frames": frame_paths,
    #         "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": "<image>\nDescribe what is happening in the video."
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": joined_caption.strip()
    #             }
    #         ]
    #     })
    
    # JSON_SAVE_PATH = "dataset_splits/webvid_train_45k_cluster_size=4_random.json"
    # with open(JSON_SAVE_PATH, "w") as outfile:
    #     json.dump(training_samples, outfile, indent=2)






    ###### PREPARE WEBVID TRAINING DATA WITH ENRICHED CAPTIONS
    # os.chdir("/simurgh/u/akhatua/long_context_benchmark/webvid_enriched_captions")
    # base_path = "webvid_enriched_captions"
    # train_path = os.path.join(base_path, "train")
    # val_path = os.path.join(base_path, "val")
    
    # train_json_files = glob.glob(f"{train_path}/*")
    # all_train_metadata = []
    # for train_json_file in tqdm(train_json_files):
    #     with open(train_json_file, "r") as fp:
    #         train_content = json.load(fp)
    #         # print(type(train_content), len(train_content))
    #         all_train_metadata += train_content

    # print(f"#train files: {len(all_train_metadata)}")

    # val_json_files = glob.glob(f"{val_path}/*")
    # all_val_metadata = []
    # for val_json_file in tqdm(val_json_files):
    #     with open(val_json_file, "r") as fp:
    #         val_content = json.load(fp)
    #         all_val_metadata += val_content

    # print(f"#val files: {len(all_val_metadata)}")

    # with open("dataset_splits/webvid_train_45k.json", "r") as fp:
    #     train_45k = json.load(fp)

    # enriched_train_45k = []
    # for train_example in tqdm(train_45k):
    #     enriched_metadata = train_example
        
    #     video_id = train_example["id"]
    #     better_caption = list(filter(lambda x: x["video_id"] == video_id, all_train_metadata))[0]["enriched_caption"]
    #     enriched_metadata["conversations"][-1]["value"] = better_caption
    #     enriched_train_45k.append(enriched_metadata)

    # with open("dataset_splits/webvid_train_enriched_caption_45k.json", "w") as outfile:
    #     json.dump(enriched_train_45k, outfile, indent=2)

    ## val
    # with open("dataset_splits/webvid_val_5k.json", "r") as fp:
    #     train_45k = json.load(fp)

    # enriched_val_45k = []
    # for train_example in tqdm(train_45k):
    #     enriched_metadata = train_example
        
    #     video_id = train_example["id"]
    #     better_caption = list(filter(lambda x: x["video_id"] == video_id, all_val_metadata))[0]["enriched_caption"]
    #     enriched_metadata["conversations"][-1]["value"] = better_caption
    #     enriched_val_45k.append(enriched_metadata)

    # with open("dataset_splits/webvid_val_enriched_caption_5k.json", "w") as outfile:
    #     json.dump(enriched_val_45k, outfile, indent=2)







    ##### PREPARE METADATA FOR TRAINING ON WEBVID - 4,8,16 frames per video
    # NUM_FRAMES = 16
    # FRAMES_SAVE_PATH = f"webvid_num_frames={NUM_FRAMES}"
    
    # METADATA = "./dataset_splits/webvid_train_45k.json"
    # with open(METADATA, "r") as fp:
    #     train_metadata = json.load(fp)
    
    # TRAIN_EXAMPLES_SIZE = len(train_metadata) # 45912
    # VIDS_PER_CLUSTER = 1
    # NUM_CLUSTERS = TRAIN_EXAMPLES_SIZE // VIDS_PER_CLUSTER
    

    # VIDEO_ORIG_PATH = "/simurgh/u/zanedurante/webvid/webvid/videos/**/*.mp4"
    # orig_video_files = glob.glob(VIDEO_ORIG_PATH, recursive=True)

    # training_samples = []
    # error_in_videos = []
    # for video in tqdm(train_metadata):
    #     video_id = video["id"]
    #     conversations = video["conversations"]
    #     frame_paths = []

    #     # create video directory to save frames: webvid_../<video id>/0000.png
    #     os.makedirs(os.path.join(FRAMES_SAVE_PATH, video_id), exist_ok=True)
    #     if len(os.listdir(os.path.join(FRAMES_SAVE_PATH, video_id))) >= NUM_FRAMES:
    #         frame_paths = [f"{video_id}/{fname}" for fname in os.listdir(os.path.join(FRAMES_SAVE_PATH, video_id))]
    #         training_samples.append({
    #             "id": video_id,
    #             "frames": frame_paths,
    #             "conversations": conversations
    #         })
    #         continue

    #     orig_video_path = list(filter(lambda x: x.endswith(f"{video_id}.mp4"), orig_video_files))[0] # hopefully just 1 video matching this ID
    #     try:
    #         frames = sample_frames_from_video_path(orig_video_path, NUM_FRAMES//VIDS_PER_CLUSTER)
    #     except:
    #         error_in_videos.append(video_id)
    #         continue

    #     for fr in range(frames.shape[0]):
    #         np_frame = frames[fr].numpy().astype(np.uint8)
    #         pil_frame = Image.fromarray(np_frame)
    #         frame_path = os.path.join(video_id, f"{str(fr).zfill(4)}.png")
            
    #         pil_frame.save(os.path.join(FRAMES_SAVE_PATH, frame_path))
    #         frame_paths.append(frame_path)
    
    # JSON_SAVE_PATH = "dataset_splits/webvid_train_45k_num_frames=16.json"
    # with open(JSON_SAVE_PATH, "w") as outfile:
    #     json.dump(training_samples, outfile, indent=2)

    # print('*'*30)
    # print(error_in_videos)
    # print('*'*30)






    ##### PREPARE METADATA FOR TRAINING ON CLUSTERED VIDEOS AND THEIR CAPTIONS
    # FRAMES_SAVE_PATH = "webvid_cluster_size=4_epoch1"
    
    METADATA = "../clustering/clustering_metadata/webvid_20k_videos_per_cluster=8_total_input_frames=16.json"
    with open(METADATA, "r") as fp:
        train_metadata = json.load(fp)
    
    TRAIN_EXAMPLES_SIZE = len(train_metadata) # 45912
    VIDS_PER_CLUSTER = 8
    NUM_CLUSTERS = TRAIN_EXAMPLES_SIZE // VIDS_PER_CLUSTER
    FRAMES_PER_VIDEO = 2
    # NUM_FRAMES = 4

    # VIDEO_ORIG_PATH = "/simurgh/u/zanedurante/webvid/webvid/videos/**/*.mp4"
    # orig_video_files = glob.glob(VIDEO_ORIG_PATH, recursive=True)

    training_samples = []
    for cluster_idx in tqdm(range(NUM_CLUSTERS)):
        videos_in_cluster = list(filter(lambda x: x["cluster"] == str(cluster_idx), train_metadata))
        assert len(videos_in_cluster) == VIDS_PER_CLUSTER, f"more videos in cluster than anticipated! {videos_in_cluster}"

        frame_paths = []
        combined_video_ids = ""
        joined_caption = ""
        for video in videos_in_cluster:
            video_id = video["video_id"]
            combined_video_ids += video_id+", "
            caption = video["caption"]
            joined_caption += caption+". "

            frame_paths += [f"{video_id}/{str(i).zfill(4)}.png" for i in range(FRAMES_PER_VIDEO)]

            # orig_video_path = list(filter(lambda x: x.endswith(f"{video_id}.mp4"), orig_video_files))[0] # hopefully just 1 video matching this ID
            # frames = sample_frames_from_video_path(orig_video_path, NUM_FRAMES//VIDS_PER_CLUSTER)

            # for fr in range(NUM_FRAMES//VIDS_PER_CLUSTER):  # frames.shape[0]
                # np_frame = frames[fr].numpy().astype(np.uint8)
                # pil_frame = Image.fromarray(np_frame)
                # frame_path = os.path.join(video_id, f"{str(fr).zfill(4)}.png")
                # os.makedirs(os.path.join(FRAMES_SAVE_PATH, video_id), exist_ok=True)
                # pil_frame.save(os.path.join(FRAMES_SAVE_PATH, frame_path))
                # frame_paths.append(frame_path)

        
        training_samples.append({
            "id": combined_video_ids.strip(),
            "frames": frame_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe what is happening in the video."
                },
                {
                    "from": "gpt",
                    "value": joined_caption.strip()
                }
            ]
        })
    
    JSON_SAVE_PATH = f"dataset_splits/webvid_train_20k_videos_per_cluster={VIDS_PER_CLUSTER}_total_input_frames={VIDS_PER_CLUSTER*FRAMES_PER_VIDEO}.json"
    with open(JSON_SAVE_PATH, "w") as outfile:
        json.dump(training_samples, outfile, indent=2)

    





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
    # k = 4
    # with open('prompt_list.txt', 'r') as fp:
    #     prompts = fp.readlines()
    # sampled_prompts = random.sample(prompts, k-1)
    # sampled_prompts = [p.strip().replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'") for p in sampled_prompts]
    # # p1 = random.choice(prompts).strip().replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'")
    # # print(f"Using the other prompt: {p1}")
    # sampled_prompts.append("Describe what is happening in the video.")

    # with open('dataset_splits/webvid_train_45k.json', 'r') as fp:
    #     webvid_train_45k = json.load(fp)

    # webvid_train_45k_diff_prompts = []
    # for train_datapoint in tqdm(webvid_train_45k):
    #     prompt = random.choice(sampled_prompts)
    #     new_datapoint = train_datapoint
    #     new_datapoint["conversations"][0]["value"] = f"<image>\n{prompt}"
    #     webvid_train_45k_diff_prompts.append(new_datapoint)

    # with open(f'dataset_splits/webvid_train_45k_diff_prompts_k={k}.json', 'w') as outfile:
    #     json.dump(webvid_train_45k_diff_prompts, outfile, indent=2)


    # with open('dataset_splits/webvid_val_5k.json', 'r') as fp:
    #     webvid_val_5k = json.load(fp)

    # webvid_val_5k_diff_prompts = []
    # for val_datapoint in tqdm(webvid_val_5k):
    #     prompt = random.choice(sampled_prompts)
    #     new_datapoint = val_datapoint
    #     new_datapoint["conversations"][0]["value"] = f"<image>\n{prompt}"
    #     webvid_val_5k_diff_prompts.append(new_datapoint)

    # with open(f'dataset_splits/webvid_val_5k_diff_prompts_k={k}.json', 'w') as outfile:
    #     json.dump(webvid_val_5k_diff_prompts, outfile, indent=2)
    

        




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
