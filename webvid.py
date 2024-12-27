# CREATING THE WEBVID DATASET

import os
import csv
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
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
        {'from': 'gpt', 'value': 'Aerial shot winter forest'}
    ]
}
'''
# webvid_metadata = []
# with open('webvid_50k.json', 'r') as fp:
#     webvid_metadata = json.load(fp)
# print(f"metadata loaded: {len(webvid_metadata)}")

if __name__=="__main__":
    # to debug why checkpoint not getting saved...
    data = [{
        "id": "10006322",
        "frames": [
        "10006322/0000.png",
        "10006322/0001.png",
        "10006322/0002.png",
        "10006322/0003.png",
        "10006322/0004.png",
        "10006322/0005.png",
        "10006322/0006.png",
        "10006322/0007.png"
        ],
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nDescribe what is happening in the video."
        },
        {
            "from": "gpt",
            "value": "Business middle aged woman smiles - night city (night urban street with cars) river - lamps and headlights - blurred"
        }
        ]
        },
        {
        "id": "1409287",
        "frames": [
        "1409287/0000.png",
        "1409287/0001.png",
        "1409287/0002.png",
        "1409287/0003.png",
        "1409287/0004.png",
        "1409287/0005.png",
        "1409287/0006.png",
        "1409287/0007.png"
        ],
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nDescribe what is happening in the video."
        },
        {
            "from": "gpt",
            "value": "Euros dolly shot isolated on white."
        }
        ]
    },
    {
        "id": "11238242",
        "frames": [
        "11238242/0000.png",
        "11238242/0001.png",
        "11238242/0002.png",
        "11238242/0003.png",
        "11238242/0004.png",
        "11238242/0005.png",
        "11238242/0006.png",
        "11238242/0007.png"
        ],
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nDescribe what is happening in the video."
        },
        {
            "from": "gpt",
            "value": "Fresh juicy beef meat hamburger with dry pepper plate over tablecloth with cutlery 1920x1080 intro motion slow hidef hd"
        }
        ]
    },
    {
        "id": "1007725081",
        "frames": [
        "1007725081/0000.png",
        "1007725081/0001.png",
        "1007725081/0002.png",
        "1007725081/0003.png",
        "1007725081/0004.png",
        "1007725081/0005.png",
        "1007725081/0006.png",
        "1007725081/0007.png"
        ],
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nDescribe what is happening in the video."
        },
        {
            "from": "gpt",
            "value": "Prague, czech republic - september 24, 2017: chinese tourists people walking in city. prague castle and metropolitan cathedral of st vitus, wenceslaus and adalbert on background. unesco world heritage"
        }
        ]
    }]

    with open('single_datapoint.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)




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
