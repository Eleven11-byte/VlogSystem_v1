# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# core folders (same names as your original script)
FRAMES_FOLDER = os.path.join(BASE_DIR, 'frames') + os.sep
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads') + os.sep
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp') + os.sep
THREEFRAMES_FOLDER = os.path.join(BASE_DIR, 'threeframes') + os.sep
FEATURES_FOLDER = os.path.join(BASE_DIR, 'featuresfromvideo') + os.sep

FACE_FOLDER = os.path.join(BASE_DIR, 'faces') + os.sep
FACE_FEATURE_FOLDER = os.path.join(BASE_DIR, 'facefeature') + os.sep
BACKGROUNDMUSIC_FOLDER = os.path.join(BASE_DIR, 'audio') + os.sep
PREPARED_FOLDER = os.path.join(BASE_DIR, 'prepareds') + os.sep
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs') + os.sep

# 上传或保存目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_SAVE_DIR = os.path.join(BASE_DIR, "data/camera_frames")

CAMERA_VIDEO_FOLDER = os.path.join(BASE_DIR, "camera_videos")
CAMERA_FRAME_FOLDER = os.path.join(BASE_DIR, "camera_frames")

# 摄像头配置
CAMERAS = [
    {"id": "cam1", "rtsp_url": "rtsp://admin:bupt1021@192.168.1.5:554/Streaming/Channels/101", "enabled": True},
    # {"id": "cam2", "rtsp_url": "rtsp://admin:password@192.168.1.102:554/Streaming/Channels/1", "enabled": True},
    # 可继续添加摄像头
]

# ensure view folders exist
VIEW_POSITIONS = ["view1", "view2", "view3"]

for view in VIEW_POSITIONS:
    os.makedirs(os.path.join(FRAMES_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(TEMP_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(THREEFRAMES_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(FEATURES_FOLDER, view), exist_ok=True)

# make base dirs
for p in [FRAMES_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER, THREEFRAMES_FOLDER, FEATURES_FOLDER,
          FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER, PREPARED_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(p, exist_ok=True)

# parameters
THRESHOLD = 1.0
# detection condition example (same as your original)
DETECT_CONDITION = [20, 800, 200, 100]
# ffmpeg path is set via moviepy config in app.py

FFMPEG_PATH = "/usr/bin/ffmpeg"
FPS = 25