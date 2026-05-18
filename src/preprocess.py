import argparse
import os
import shutil
import time

import cv2
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input',  default='data/videos')
parser.add_argument('--output', default='data/dataset')
parser.add_argument('--target_size', type=int, default=256)
parser.add_argument('--max_frames', type=int, default=210)
args = parser.parse_args()

input_dir = args.input
output_dir = args.output
target_size = args.target_size
max_frames = args.max_frames

if os.path.exists(output_dir):
    # Delete the output directory if it already exists
    print(f"Deleting existing output directory: {output_dir}")
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=False)

start_time = time.time()

# Loop through the classes (skip hidden dirs like .claude)
for i, class_name in enumerate(sorted(d for d in os.listdir(input_dir) if not d.startswith('.'))):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Create the class folder in the output directory
    output_class_dir = os.path.join(output_dir, f"class_{i:02d}_{class_name}")
    os.makedirs(output_class_dir, exist_ok=True)

    print(f"Processing class {class_name} ({i + 1}/{len(os.listdir(input_dir))})")

    # Loop through the videos in the class folder
    video_count = 0
    for video_name in tqdm(sorted(os.listdir(class_path)), total=len(os.listdir(class_path))):
        video_path = os.path.join(class_path, video_name)
        video_extension = os.path.splitext(video_name)[1].lower()

        # Check if the file is a supported video format
        if video_extension in ['.mp4', '.avi']:
            # Open the video and extract frames
            cap = cv2.VideoCapture(video_path)

            # Check if length of video > max_frames
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if length > max_frames:
                print(f"Skipping {class_name}, {video_name}, (would have been written to) video_{video_count:03d}, length {length} > {max_frames}")
                cap.release()
                continue

            # Create a folder for the video in the output class directory
            output_video_dir = os.path.join(output_class_dir, f"video_{video_count:03d}")
            os.makedirs(output_video_dir, exist_ok=True)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                height, width, _ = frame.shape

                # To grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Crop
                start_x = (width - height) // 2
                frame = frame[:, start_x:start_x + height]
                # Resize
                frame = cv2.resize(frame, (target_size, target_size))

                # Save the frame as an image
                frame_path = os.path.join(output_video_dir, f'frame_{frame_count:04d}.jpg')
                Image.fromarray(frame).save(frame_path)
                frame_count += 1

            cap.release()
        
        video_count += 1

print(f'Preprocessing complete in {time.time() - start_time:.2f} seconds. Output saved in {output_dir}.')

# Check for corrupted frames
print("Checking for corrupted frames...")
corrupted_frames = []
for dirpath, _, filenames in os.walk(output_dir):
    for filename in filenames:
        frame_path = os.path.join(dirpath, filename)
        try:
            Image.open(frame_path).verify()
        except (IOError, SyntaxError):
            corrupted_frames.append(frame_path)
            print(f"Frame {frame_path} is corrupted")

print(f"Found {len(corrupted_frames)} corrupted frames")