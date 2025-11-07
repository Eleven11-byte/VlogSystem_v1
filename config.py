# config.py
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'DATA')

FRAMES_FOLDER = os.path.join(DATA_DIR, 'frames') + os.sep
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'records') + os.sep
FEATURES_FOLDER = os.path.join(DATA_DIR, 'featuresfromvideo') + os.sep

FACE_FOLDER = os.path.join(DATA_DIR, 'faces') + os.sep
FACE_FEATURE_FOLDER = os.path.join(DATA_DIR, 'facefeature') + os.sep
BACKGROUNDMUSIC_FOLDER = os.path.join(DATA_DIR, 'audio') + os.sep
PREPARED_FOLDER = os.path.join(DATA_DIR, 'prepareds') + os.sep
OUTPUT_FOLDER = os.path.join(DATA_DIR, 'outputs') + os.sep
OUTPUT_WATERMARK_FOLDER = os.path.join(DATA_DIR, 'output_watermarks') + os.sep

# ensure view folders exist
VIEW_POSITIONS = ["view1", "view2", "view3"]
# TODO:摄像头和景点之间的映射关系
CAMERA_VIEW = {"cam1":"view1", "cam2":"view2"}

for view in VIEW_POSITIONS:
    os.makedirs(os.path.join(FRAMES_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, view), exist_ok=True)
    # os.makedirs(os.path.join(TEMP_FOLDER, view), exist_ok=True)
    # os.makedirs(os.path.join(THREEFRAMES_FOLDER, view), exist_ok=True)
    os.makedirs(os.path.join(FEATURES_FOLDER, view), exist_ok=True)

# make base dirs
for p in [FRAMES_FOLDER, UPLOAD_FOLDER, FEATURES_FOLDER,
          FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER, PREPARED_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(p, exist_ok=True)

# parameters
THRESHOLD = 1.0
# detection condition example (same as your original)
DETECT_CONDITION = [20, 800, 200, 100]
# ffmpeg path is set via moviepy config in app.py



# 查找 ffmpeg 的路径
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise FileNotFoundError("未找到 ffmpeg，请确认已安装并加入系统环境变量。")

print(f"找到 ffmpeg 路径：{ffmpeg_path}")

FFMPEG_PATH = ffmpeg_path #TODO: 在不同设备上需要更改此处ffmpeg位置
# FFMPEG_PATH = "D:/Document/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"
FPS = 25
RECORD_DURATION = 10

CAMERAS = [
        {
            "cam_id": "cam1",
            "rtsp_url": "rtsp://admin:bupt1021@192.168.1.5:554/Streaming/Channels/101",
            "save_root": DATA_DIR,
            "record_duration": RECORD_DURATION,
        },
    ]


