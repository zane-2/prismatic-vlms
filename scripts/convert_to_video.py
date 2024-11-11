# For now, just converts the images into single frame videos. This effectively changes nothing about the data, but allows for easier integration with the rest of the video pipeline.
import json
from tqdm import tqdm
from copy import deepcopy

input_file = "data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"
output_file = "data/download/llava-v1.5-instruct/llava_v1_5_mix665k_videos_1frame.json"


with open(input_file, "r") as f:
    data = json.load(f)

print(data[0])
new_rows = []
for row in tqdm(data):
    if "image" in row:
        row["frames"] = [row["image"]] # single frame video
        del row["image"]
    new_rows.append(row) 

with open(output_file, "w") as f:
    json.dump(new_rows, f)


input_file = "data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"
output_file = "data/download/llava-v1.5-instruct/llava_v1_5_mix665k_videos_2frame.json"

num_frames = 2

with open(input_file, "r") as f:
    data = json.load(f)


print(data[0])
new_rows = []
for idx, row in tqdm(enumerate(data), total=len(data)):
    if "image" not in row or "image" not in data[idx-1]:
        continue
    if idx % num_frames == 1:
        first_row = data[idx-1]
        second_row = row
        first_row_v1 = deepcopy(row)
        second_row_v1 = deepcopy(data[idx-1])
        first_row_v2 = deepcopy(first_row_v1)
        second_row_v2 = deepcopy(second_row_v1)
        first_row_v1["frames"] = [first_row["image"], second_row["image"]]
        second_row["frames"] = [first_row["image"], second_row["image"]]
        first_row_v2["frames"] = [second_row["image"], first_row["image"]]
        second_row_v2["frames"] = [second_row["image"], first_row["image"]]
        del first_row_v1["image"]
        del second_row_v1["image"]
        del first_row_v2["image"]
        del second_row_v2["image"]
        new_rows.append(first_row_v1)
        new_rows.append(second_row_v1)
        new_rows.append(first_row_v2)
        new_rows.append(second_row_v2)

with open(output_file, "w") as f:
    json.dump(new_rows, f)
