import os
from datetime import datetime

# FFmpeg 路径配置
FFMPEG_PATH = "D:/Document/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"

# 人脸识别阈值
FACE_RECOGNITION_THRESHOLD = 1.0

# 录制条件参数
DETECT_CONDITION = [20, 800, 200, 100]

# 景点列表
VIEW_POSITIONS = ["view1", "view2", "view3"]

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础文件夹路径
FRAMES_FOLDER = os.path.join(BASE_DIR, 'frames/')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads/')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp/')
THREEFRAMES_FOLDER = os.path.join(BASE_DIR, 'threeframes/')
FEATURES_FOLDER = os.path.join(BASE_DIR, 'featuresfromvideo/')
FACE_FOLDER = os.path.join(BASE_DIR, 'faces/')
FACE_FEATURE_FOLDER = os.path.join(BASE_DIR, 'facefeature/')
BACKGROUNDMUSIC_FOLDER = os.path.join(BASE_DIR, 'audio/')
PREPARED_FOLDER = os.path.join(BASE_DIR, 'prepareds/')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs/')


# 创建所有目录
def init_directories():
    # 创建基础目录
    for dir_path in [
        FRAMES_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER,
        THREEFRAMES_FOLDER, FEATURES_FOLDER, FACE_FOLDER,
        FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER,
        PREPARED_FOLDER, OUTPUT_FOLDER
    ]:
        os.makedirs(dir_path, exist_ok=True)

    # 创建景点子目录
    for view in VIEW_POSITIONS:
        for base_dir in [FRAMES_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER, THREEFRAMES_FOLDER, FEATURES_FOLDER]:
            os.makedirs(os.path.join(base_dir, view), exist_ok=True)


# 初始化目录
init_directories()