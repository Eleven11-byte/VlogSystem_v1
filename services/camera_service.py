# services/camera_service.py
import base64
import time
import numpy as np
import cv2
from flask import jsonify
from config import FRAMES_FOLDER, VIEW_POSITIONS, DETECT_CONDITION
from record import DetectVisitor

# visitor_detector 与原始逻辑一致（使用你的 record.py）
visitor_detector = DetectVisitor(DETECT_CONDITION)
# 单摄像头简单全局状态（原代码使用全局 is_recording）
is_recording = False

def get_is_recording():
    global is_recording
    return jsonify({'isRecording': is_recording})

def handle_upload_frames(request):
    """
    接收 base64 图像（JSON），检测是否触发录制（与原代码一致）
    """
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
