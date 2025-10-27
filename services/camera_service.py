# services/camera_service.py
import base64
import time
import numpy as np
import cv2
from flask import jsonify, request
# from config import FRAMES_FOLDER, VIEW_POSITIONS, DETECT_CONDITION, CAMERAS
from config import CAMERA_FRAME_FOLDER, CAMERA_VIDEO_FOLDER, FPS, CAMERAS
from record import DetectVisitor
from utils_app import ensure_dir
import os
import threading
"""
# visitor_detector 与原始逻辑一致（使用你的 record.py）
visitor_detector = DetectVisitor(DETECT_CONDITION)
# 单摄像头简单全局状态（原代码使用全局 is_recording）
is_recording = False

def get_is_recording():
    global is_recording
    return jsonify({'isRecording': is_recording})

def handle_upload_frames(request):
    
    # 接收 base64 图像（JSON），检测是否触发录制（与原代码一致）

    global is_recording
    try:
        if not is_recording:
            data = request.json
            image_data = data['image']
            view_position = data.get('viewPosition', VIEW_POSITIONS[0])

            image_data = image_data.split(",")[1] if "," in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            is_recording = visitor_detector.detect(img)
            if is_recording:
                # 模拟你原来 sleep 15s 的行为（触发录制后等待）
                # 这里保持原样：阻塞 sleep（可改为线程任务）
                time.sleep(15)
                is_recording = False

        return ('upload frames', 200)
    except Exception as e:
        return (f"error: {e}", 500)
"""

class CameraThread(threading.Thread):
    """单个摄像头采集线程"""
    def __init__(self, cam_id, rtsp_url, save_dir):
        super().__init__()
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.save_dir = save_dir
        self.cap = None
        self.running = False
        self.frame = None

    def run(self):
        self.running = True
        os.makedirs(self.save_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print(f"Camera {self.cam_id} failed to open RTSP stream.")
            return

        print(f"Camera {self.cam_id} stream started.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.cam_id} failed to read frame, retrying...")
                time.sleep(1)
                continue

            self.frame = frame
            # 保存最近帧
            frame_path = os.path.join(self.save_dir, f"{self.cam_id}_latest.jpg")
            cv2.imwrite(frame_path, frame)

            time.sleep(0.1)  # 控制帧率（10fps）

        self.cap.release()
        print(f"Camera {self.cam_id} stopped.")

    def stop(self):
        self.running = False

class CameraManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'cameras'):
            self.cameras = {}  # {camera_id: {'url':..., 'thread':..., 'running':...}}

    def start_all(self):
        for cam_cfg in CAMERAS:
            if not cam_cfg["enabled"]:
                continue
            cam_id = cam_cfg["id"]
            rtsp_url = cam_cfg["rtsp_url"]
            save_dir = os.path.join(CAMERA_FRAME_FOLDER, cam_id)
            cam_thread = CameraThread(cam_id, rtsp_url, save_dir)
            cam_thread.start()
            self.cameras[cam_id] = cam_thread
            print("start all cameras")

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()

    def get_status(self):
        """返回摄像头状态列表"""
        status = []
        for cam_id, cam in self.cameras.items():
            status.append({
                "cam_id": cam_id,
                "running": cam.running,
                "has_frame": cam.frame is not None
            })
        return status

    def get_frame_path(self, cam_id):
        """获取最新帧路径"""
        if cam_id not in self.cameras:
            return None
        return os.path.join(CAMERA_FRAME_FOLDER, cam_id, f"{cam_id}_latest.jpg")

    # 初始化全局摄像头管理器
camera_manager = CameraManager()


