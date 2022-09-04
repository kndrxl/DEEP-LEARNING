import os
import cv2
import argparse
from os.path import join, dirname
from pathlib import Path
from logging import Logger
from utilities.inference import YoloInference

logger = Logger('Yolo Module')

def detect(
    video_path: str,
    video_save_path: str,
    frames_path: str = "output/",
    instance: YoloInference = YoloInference()
):
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    
    instance.load_model('yolov5s')
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames = []
    results = []
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps =  video_capture.get(cv2.CAP_PROP_FPS)

    while video_capture.isOpened():
        frame_is_read, frame = video_capture.read()  

        if frame_is_read:
            if not os.path.exists(f"{frames_path}/frames/raw/"):
                os.makedirs(f"{frames_path}/frames/raw/")
            cv2.imwrite(f"{frames_path}/frames/raw/frame_{str(frame_idx)}.jpg", frame)
            frames.append(f"{frames_path}/frames/raw/frame_{str(frame_idx)}.jpg")
            frame_idx += 1

            results.append(instance.frame_inference(
                frame, 
                frames_path,
                f"{Path(video_path).stem}_frame_{str(frame_idx)}.jpg"
            ))

        else:
            logger.info("Could not read the frame.")
            break

    video_capture.release()

    # print(f"Saving video to: {video_save_path}")
    # save frames to output video
    output_video = cv2.VideoWriter(filename=video_save_path, fourcc=cv2.VideoWriter_fourcc(*'avc1'), fps=fps, frameSize=(int(width), int(height)))
    for frame_num in range(frame_idx):
        image = cv2.imread(f"{frames_path}/frames/detected/{Path(video_path).stem}_frame_{str(frame_num+1)}.jpg")
        output_video.write(image)

    output_video.release()

    return results


video_path = "video/us.mp4"
video_save_path = "output/videos/us_labeled.mp4"
detect(video_path, video_save_path)