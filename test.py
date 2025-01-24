import os
import requests
import torch
from torchvision import transforms

from PIL import Image
from pathlib import Path

from prismatic import load
import cv2

# # For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

def extract_center_frame(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

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


if __name__=="__main__":
    # Example usage
    video_path = "/vision/u/silsingh/NExT-QA/NExTVideo_test_videos/10109006686.mp4"
    # os.system(f"cp {video_path} 10109006686.mp4")
    # save_path = "center_frame.jpg"
    # extract_center_frame(video_path, save_path)

    # # Download an image and specify a prompt
    # image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    # image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = extract_center_frame(video_path)
    # image.save('temp.png')
    # print(image.size)
    # exit()
    user_prompt = "why is there an orange thing at the end of the video"


    # Build prompt
    # prompt_builder = vlm.get_prompt_builder()
    # prompt_builder.add_turn(role="human", message=user_prompt)
    # prompt_text = prompt_builder.get_prompt()

    # Generate!
    pixel_value = transforms.ToTensor()(image)
    generated_text = vlm.generate_batch(
        pixel_value,
        user_prompt,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
        min_length=1
    )
    # generated_text = vlm.generate(
    #     image,
    #     prompt_text,
    #     do_sample=True,
    #     temperature=0.4,
    #     max_new_tokens=512,
    #     min_length=1,
    # )

    print(generated_text)