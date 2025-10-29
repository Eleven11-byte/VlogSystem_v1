# utils.py
import os
import subprocess
import numpy as np
import cv2
import re

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

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

from datetime import datetime

def generate_file_name(base_timestamp, file_type, seq, sub_seq=None):
    """
    生成文件名
    :param base_timestamp: 视频起始时间戳（字符串，如"20251029_163025_789"）
    :param file_type: 文件类型（"video"/"frame"/"facefeat"）
    :param seq: 主序号（视频固定000，帧/特征按顺序递增）
    :param sub_seq: 人脸特征二级序号（一帧多个人脸时使用，如01/02）
    :return: 完整文件名
    """
    # 序号补零（确保3位，支持001~999）
    seq_str = f"{seq:03d}"
    # 拼接二级序号（若有）
    if sub_seq is not None:
        seq_str += f"_{sub_seq:02d}"
    # 后缀映射
    ext_map = {"video": "mp4", "frame": "jpg", "facefeat": "npy"}
    return f"{base_timestamp}_{file_type}_{seq_str}.{ext_map[file_type]}"

def extract_base_timestamp_regex(file_name):
    # 正则匹配基础时间戳（格式：8位日期_6位时间_3位毫秒）
    pattern = r"^(\d{8}_\d{6}_\d{3})"
    # 搜索文件名（先移除后缀）
    name_without_ext = file_name.split(".")[0]
    match = re.search(pattern, name_without_ext)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"文件名格式错误，无法提取时间戳：{file_name}")

def find_corresponding_video(face_feat_file, video_ext="mp4"):
    # 1. 提取基础时间戳
    base_ts = extract_base_timestamp_regex(face_feat_file)
    # 2. 拼接视频文件名
    video_file = f"{base_ts}_video_000.{video_ext}"
    # 3. 假设视频与特征文件在同一目录，直接返回路径（实际中可拼接目录）
    return video_file

if __name__ == "__main__":
    # 示例：生成某视频的关联文件

    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 基础时间戳
    video_name = generate_file_name(base_ts, "video", 0)  # 视频文件
    frame1_name = generate_file_name(base_ts, "frame", 1)  # 第1帧
    face_feat_name = generate_file_name(base_ts, "facefeat", 1)  # 第3帧第1个人脸特征

    print(video_name)       # 20251029_163025_789_video_000.mp4
    print(frame1_name)      # 20251029_163025_789_frame_001.jpg
    print(face_feat_name)   # 20251029_163025_789_facefeat_003_01.npy

    base_ts_find = extract_base_timestamp_regex(face_feat_name)
    video_name = find_corresponding_video(face_feat_name)
    print(base_ts_find)
    print(video_name)

