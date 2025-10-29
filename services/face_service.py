# services/face_service.py
import os
import time
from flask import jsonify
from config import  UPLOAD_FOLDER, FEATURES_FOLDER, FACE_FOLDER, FACE_FEATURE_FOLDER, VIEW_POSITIONS
from utils_app import reencode_video, extract_frames_from_video
from face_extractor_singleton import face_extractor  # 你原来的单例
import cv2

# timestamps used in your original code
TIMESTAMPS = [5, 7.5, 10]

# 加入摄像头设备后已弃用
"""
def handle_upload_video(request):
    # 接收 multipart/form-data 视频文件并存储（保持原逻辑）
    if 'video' not in request.files:
        return 'No video part', 400

    video = request.files['video']
    view_position = request.form.get('viewPosition', VIEW_POSITIONS[0])

    temp_video_path = os.path.join(TEMP_FOLDER, view_position, f"temp_{int(time.time())}_{view_position}.mp4")
    out_video_path = os.path.join(UPLOAD_FOLDER, view_position, f"{int(time.time())}_{view_position}.mp4")
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    video.save(temp_video_path)

    try:
        reencode_video(temp_video_path, out_video_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 抽帧并保存特征
    try:
        extract_frames_from_video(out_video_path, TIMESTAMPS, THREEFRAMES_FOLDER, view_position, face_extractor, FEATURES_FOLDER)
    except Exception as e:
        # 记录错误但不阻断上传
        print("extract frames error:", e)

    return 'Video uploaded successfully', 200
"""

def handle_upload_facepic(request):
    """
    接收单张人脸图片，保存并提取特征（保持原接口行为）
    """
    if 'facePic' not in request.files:
        return jsonify('失败-没有人脸图像被上传'), 400

    user_id = request.form.get("userId")
    if not user_id:
        return jsonify('失败-缺少 userId 参数'), 400

    facePic = request.files['facePic']
    if facePic.filename == '':
        return jsonify('失败-没有选择人脸图像'), 400

    if not facePic.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify('失败-只支持 JPG,JPEG,PNG 格式'), 400

    os.makedirs(FACE_FOLDER, exist_ok=True)
    os.makedirs(FACE_FEATURE_FOLDER, exist_ok=True)

    facePic_path = os.path.join(FACE_FOLDER, f"{user_id}.jpg")
    facePic.save(facePic_path)

    img = cv2.imread(facePic_path)
    embeddings = face_extractor.extract_features(img)
    faceFeature_path = os.path.join(FACE_FEATURE_FOLDER, f"{user_id}.npy")
    # 保存 np 特征
    import numpy as np
    np.save(faceFeature_path, embeddings)

    return jsonify({'message': '成功-人脸图像上传成功', 'userId': user_id})

