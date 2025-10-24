# app.py
import moviepy.config as mpy_config
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os

# make sure moviepy uses your ffmpeg
mpy_config.change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})

from config import (
    FRAMES_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER, THREEFRAMES_FOLDER,
    FEATURES_FOLDER, FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER,
    PREPARED_FOLDER, OUTPUT_FOLDER, VIEW_POSITIONS, THRESHOLD
)

from services.camera_service import handle_upload_frames, get_is_recording
from services.face_service import handle_upload_facepic, handle_upload_video
from services.video_service import handle_get_video, download_video_file, preview_output

app = Flask(__name__)
CORS(app)

# create folders at startup (redundant safe)
for p in [FRAMES_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER, THREEFRAMES_FOLDER, FEATURES_FOLDER,
          FACE_FOLDER, FACE_FEATURE_FOLDER, BACKGROUNDMUSIC_FOLDER, PREPARED_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(p, exist_ok=True)

# Routes (kept same endpoints as your original)
@app.route('/isRecording', methods=['GET'])
def is_recording():
    return get_is_recording()

@app.route('/uploadFrames', methods=['POST'])
def upload_frames():
    return handle_upload_frames(request)

@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    return handle_upload_video(request)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

