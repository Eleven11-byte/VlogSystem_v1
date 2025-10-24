# utils.py
import os
import subprocess
import numpy as np
import cv2

def reencode_video(input_path, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(cmd)

def save_features_np(save_path, embeddings):
    # 保存 numpy 特征
    np.save(save_path, embeddings)

def extract_frames_from_video(video_path, timestamps, out_folder, view_position, face_extractor, features_folder):
    """
    从视频中按时间点提取帧并保存图像与特征（调用外部 face_extractor）
    timestamps: list of seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_name = os.path.basename(video_path).split(".")[0]
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(out_folder, f"{video_name}_frame_{t}s.jpg")
            cv2.imwrite(img_path, frame)
            feature_path = os.path.join(features_folder, view_position, f"{video_name}_frame_{t}s.npy")
            embeddings = face_extractor.extract_features(frame)
            np.save(feature_path, embeddings)
    cap.release()
