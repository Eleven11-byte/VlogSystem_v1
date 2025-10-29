# app.py
import moviepy.config as mpy_config
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

from config import (
    FRAMES_FOLDER, UPLOAD_FOLDER,
    FEATURES_FOLDER, FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER,
    PREPARED_FOLDER, OUTPUT_FOLDER, CAMERAS, FFMPEG_PATH
)

# from services.camera_service import handle_upload_frames, get_is_recording

from services.face_service import handle_upload_facepic
from services.video_service import handle_get_video, download_video_file, preview_output
from services.camera_service_modified import CameraManager

# make sure moviepy uses your ffmpeg
mpy_config.change_settings({"FFMPEG_BINARY": FFMPEG_PATH})

app = Flask(__name__)
CORS(app)

# create folders at startup (redundant safe)
for p in [FRAMES_FOLDER, UPLOAD_FOLDER, FEATURES_FOLDER,
          FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER, PREPARED_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(p, exist_ok=True)
mpy_config.change_settings({"FFMPEG_BINARY": "D:/Document/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"})


# 初始化摄像头
camera_manager = CameraManager(CAMERAS)
camera_manager.start_all()

# 之前采用前端摄像头录制视频，已弃用
"""
@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    return handle_upload_video(request)
"""

@app.route('/uploadFacePic', methods=['POST'])
def upload_facepic():
    return handle_upload_facepic(request)

@app.route('/getVideo', methods=['POST'])
def get_video():
    return handle_get_video(request)

@app.route('/outputs/<path:filename>', methods=['GET'])
def preview(filename):
    return preview_output(filename)

@app.route('/downloadVideo', methods=['GET'])
def download_video():
    return download_video_file(request)


"""
摄像头管理
"""

@app.route("/camera/status", methods=["GET"])
def camera_status():
    """查看所有摄像头状态"""
    return jsonify(camera_manager.get_status())


@app.route("/camera/frame/<cam_id>", methods=["GET"])
def get_latest_frame(cam_id):
    """获取某个摄像头的最新帧"""
    frame_path = camera_manager.get_frame_path(cam_id)
    if frame_path and os.path.exists(frame_path):
        return send_file(frame_path, mimetype="image/jpeg")
    else:
        return jsonify({"status": "error", "msg": "No frame available"}), 404

@app.route("/")
def index():
    return jsonify({"msg": "Camera backend running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

