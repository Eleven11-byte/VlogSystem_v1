# services/camera_service.py
import base64
import time
import numpy as np
import cv2
from flask import jsonify, request
from datetime import datetime

from torch.nn.functional import embedding

# from config import FRAMES_FOLDER, VIEW_POSITIONS, DETECT_CONDITION, CAMERAS
from config import FPS
from record import DetectVisitor
from utils_app import ensure_dir
import os
import threading
from threading import Thread, Event
from AIDetector_pytorch import Detector
from face_recognize import FaceExtractor

class CameraThread(threading.Thread):
    """单个摄像头采集线程"""
    def __init__(self, cam_id, rtsp_url, save_dir, record_duration):
        super().__init__()
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        
        # 存储路径
        self.frames_dir = os.path.join(save_dir, "frames", cam_id)    
        self.records_dir = os.path.join(save_dir, "records", cam_id)
        self.face_features_dir = os.path.join(save_dir, "featuresfromvideo", cam_id)
        self._ensure_dirs()

        # 状态控制
        self.running = False
        self.is_capturing = False
        self.is_recording = False

        # 摄像头资源
        self.cap = None
        self.fps = FPS
        # self.frame_size = (1920, 1080)

        self.detector = Detector()
        self.bboxes = None
        self.record_duration = record_duration

        self.condition = []

        self.face_extractor = FaceExtractor()
        
    def _ensure_dirs(self):
        """确保frames和records文件夹存在"""
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)
        if not os.path.exists(self.face_features_dir):
            os.makedirs(self.face_features_dir)
        

    def _connect_camera(self):
        """连接摄像头"""
        print(f"Connecting to camera {self.cam_id} at {self.rtsp_url}...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20
            self.frame_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            frame_width, frame_height = self.frame_size

            # 1. 目标范围：仅限制x轴和y轴上边界（去掉y轴下边界）
            x_min = int(frame_width * 0.15)  # （画面15%处）
            x_max = int(frame_width * 0.85)  # （画面85%处）
            y_max = int(frame_height * 0.95)  # (画面95%以下)

            # 2. 目标大小：足够清晰的阈值
            min_width = int(frame_width * 0.3)  # 最小宽度（画面30%）
            min_height = int(frame_height * 0.4)  # 最小高度(画面40%）

            self.condition = [x_min, x_max, y_max, min_width, min_height]

            print("目标范围和大小阈值:", self.condition)

            self.is_connected = True
            print(f"Camera {self.cam_id} connected: FPS={self.fps}, Size={self.frame_size}")
            return True
        return False

    def _save_frame(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_path = os.path.join(self.frames_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _check_record_condition(self):

        if self.bboxes is None or len(self.bboxes) == 0:
            print("No targets detected.")
            return False  # 无目标时不录制

        # 遍历所有检测到的目标
        for bbox in self.bboxes:
            x1, y1, x2, y2, cls, _ = bbox
            if cls != 'person':
                continue  # 只关注人体目标

            # 计算目标宽高
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            print(f"{x1},{x2},{y2},{bbox_width},{bbox_height}")

            # 检查条件：在x轴范围内 + y轴上边界之下 + 宽高足够（无y轴下边界限制）
            in_x_range = (x1 > self.condition[0]) and (x2 < self.condition[1])
            in_y_upper_range = (y2 < self.condition[2])  # 仅限制上边界，不限制下边界
            large_enough = (bbox_width > self.condition[3]) and (bbox_height > self.condition[4])

            if in_x_range and in_y_upper_range and large_enough:
                print("Recording condition met.")
                return True  # 任一目标满足即触发录制

        return False  # 所有目标均不满足

    def _record_video(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.records_dir, f"{self.cam_id}_{timestamp}.mp4")

        total_frames_to_save = 8  # 总共保存8帧用于特征提取
        interval = self.record_duration / (total_frames_to_save - 1)


        # 帧保存路径（使用与视频相同的时间戳前缀，便于关联）
        frame_prefix = f"{timestamp}"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, self.frame_size)
        if not out.isOpened():
            return

        start_time = time.time()
        last_save_time = start_time
        frame_count = 0

        while (time.time() - start_time) < self.record_duration and self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            out.write(frame)
            current_time = time.time()

            need_save = (current_time - last_save_time >= interval) or \
                        (current_time - start_time >= self.record_duration - 1)
            if need_save and frame_count < total_frames_to_save:
                frame_path = os.path.join(self.frames_dir, f"{frame_prefix}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                feature_path = os.path.join(self.face_features_dir, f"{frame_prefix}_{frame_count}.npy")
                print(feature_path)
                np.save(feature_path, self.face_extractor.extract_features(frame))
                last_save_time = current_time
                frame_count += 1 #每次存储后计数
        out.release()

    def run(self):
        if not self._connect_camera():
            return
        
        self.running = True
        
        while self.running:
            # print(f"is_recording: {self.is_recording}")
            if self.is_recording:
                self._record_video()
                self.is_recording = False
            else:
                ret, frame = self.cap.read()
                if ret:
                    # self._save_frame(frame)
                    result = self.detector.feedCap(frame) # 对每一帧进行检测
                    self.bboxes = result['bboxes2draw']
                    if self._check_record_condition():
                        self.is_recording = True
                else:
                    time.sleep(1)
            
            time.sleep(0.01)
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.is_recording = False

class CameraManager:
    def __init__(self, camera_configs):
        self.camera_threads = {}
        for config in camera_configs:
            thread = CameraThread(
                cam_id=config["cam_id"],
                rtsp_url=config["rtsp_url"],
                save_dir=config["save_root"],
                record_duration=config["record_duration"],
            )
            self.camera_threads[config["cam_id"]] = thread

    def start_all(self):
        for cam_id, thread in self.camera_threads.items():
            if not thread.is_alive():
                thread.start()
        print("All camera threads started.")

    def stop_all(self):
        for thread in self.camera_threads.values():
            thread.stop()
            thread.join()
        print("All camera threads stopped.")

    def get_status(self, cam_id):
        thread = self.camera_threads.get(cam_id)
        if not thread:
            return {"cam_id": cam_id, "status": "invalid"}
        return {
            "cam_id": cam_id,
            "connected": thread.is_connected,
            "running": thread.running,
            "recording": thread.is_recording
        }

"""
if __name__ == "__main__":
    # 摄像头配置列表

    configs = [
        {
            "cam_id": "cam1",
            "rtsp_url": "rtsp://admin:bupt1021@192.168.1.5:554/Streaming/Channels/101",
            "save_root": "camera",
            "record_duration": RECORD_DURATION,
        },
    ]

    # 初始化管理器并启动所有摄像头
    manager = CameraManager(configs)
    manager.start_all()

    # 运行3分钟后停止
    try:
        time.sleep(180)
    finally:
        manager.stop_all()
"""
