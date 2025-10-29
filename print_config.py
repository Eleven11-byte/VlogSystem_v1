import config

def print_all_settings():
    print("=== 配置路径及参数检查 ===")
    print(f"BASE_DIR: {config.BASE_DIR}")
    print("\n=== 核心文件夹路径 ===")
    print(f"FRAMES_FOLDER: {config.FRAMES_FOLDER}")
    print(f"UPLOAD_FOLDER: {config.UPLOAD_FOLDER}")
    print(f"TEMP_FOLDER: {config.TEMP_FOLDER}")
    print(f"THREEFRAMES_FOLDER: {config.THREEFRAMES_FOLDER}")
    print(f"FEATURES_FOLDER: {config.FEATURES_FOLDER}")
    print(f"FACE_FOLDER: {config.FACE_FOLDER}")
    print(f"FACE_FEATURE_FOLDER: {config.FACE_FEATURE_FOLDER}")
    print(f"BACKGROUNDMUSIC_FOLDER: {config.BACKGROUNDMUSIC_FOLDER}")
    print(f"PREPARED_FOLDER: {config.PREPARED_FOLDER}")
    print(f"OUTPUT_FOLDER: {config.OUTPUT_FOLDER}")

    print("\n=== 上传/保存目录 ===")
    print(f"FRAME_SAVE_DIR: {config.FRAME_SAVE_DIR}")
    print(f"CAMERA_VIDEO_FOLDER: {config.CAMERA_VIDEO_FOLDER}")
    print(f"CAMERA_FRAME_FOLDER: {config.CAMERA_FRAME_FOLDER}")

    print("\n=== 视图位置 ===")
    print(f"VIEW_POSITIONS: {config.VIEW_POSITIONS}")

    print("\n=== 参数设置 ===")
    print(f"THRESHOLD: {config.THRESHOLD}")
    print(f"DETECT_CONDITION: {config.DETECT_CONDITION}")
    print(f"FFMPEG_PATH: {config.FFMPEG_PATH}")
    print(f"FPS: {config.FPS}")
    print(f"RECORD_DURATION: {config.RECORD_DURATION}")

    print("\n=== 相机配置 ===")
    for i, cam in enumerate(config.CAMERAS, 1):
        print(f"相机 {i}:")
        print(f"  cam_id: {cam['cam_id']}")
        print(f"  rtsp_url: {cam['rtsp_url']}")
        print(f"  save_root: {cam['save_root']}")
        print(f"  record_duration: {cam['record_duration']}")


if __name__ == "__main__":
    print_all_settings()