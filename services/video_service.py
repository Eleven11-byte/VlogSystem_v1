# services/video_service.py
import os
import numpy as np
from flask import jsonify, send_from_directory, send_file
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from config import (FACE_FEATURE_FOLDER, FEATURES_FOLDER, UPLOAD_FOLDER, PREPARED_FOLDER,
                    BACKGROUNDMUSIC_FOLDER, OUTPUT_FOLDER, VIEW_POSITIONS, THRESHOLD)
import face_recognize  # 保持原有接口
from utils_app import reencode_video  # 可能用不到，但保留
from utils_app import reencode_video

def find_video_indices(userVideoList, keyword):
    indices = [index for index, video in enumerate(userVideoList) if keyword in video]
    return indices[0]

def find_similar(file_path, target_embedding, threshold):
    video_list = []
    files = os.listdir(file_path)
    for file in files:
        faces_embeddings = np.load(os.path.join(file_path, file), allow_pickle=True)
        for embedding in faces_embeddings:
            if face_recognize.face_compare(embedding, target_embedding, threshold):
                video_name_ele = file.split('_')
                video_name = video_name_ele[0] + "_" + video_name_ele[1]
                if video_name not in video_list:
                    video_list.append(video_name)
    return video_list

def getVideoList(view_position_list, target_embedding, threshold):
    video_list_all = []
    video_list_result = []

    for view_position in view_position_list:
        feature_path = os.path.join(FEATURES_FOLDER, view_position)
        # use face_recognize.find_similar if you have it
        video_list = face_recognize.find_similar(feature_path, target_embedding, threshold) \
                    if hasattr(face_recognize, 'find_similar') else find_similar(feature_path, target_embedding, threshold)
        if len(video_list) == 0:
            video_list_result.append(None)
        else:
            video_list_result.append(video_list[-1])
        video_list_all.append(video_list)

    return video_list_result, video_list_all

def handle_get_video(request):
    music_type = request.form.get('musicType')
    user_id = request.form.get("userId")

    if not user_id:
        return jsonify({"error": "缺少 userId 参数"}), 400

    faceFeature_path = os.path.join(FACE_FEATURE_FOLDER, f"{user_id}.npy")
    if not os.path.exists(faceFeature_path):
        return {"error": f"未找到 {user_id} 的人脸特征，请先上传"}

    target_embeddings = np.load(faceFeature_path, allow_pickle=True)

    # 默认使用第一个 embedding（跟你原代码一致）
    videoOfUser, _ = getVideoList(VIEW_POSITIONS, target_embeddings[0], THRESHOLD)

    # 把对应的视频加入 clips（与原代码逻辑保持一致）
    clips = []
    # 处理每个 view (view1, view2, view3)
    for v in VIEW_POSITIONS:
        # find index in videoOfUser
        try:
            idx = find_video_indices(videoOfUser, v)
            user_video_name = videoOfUser[idx]
            user_video_path = os.path.join(UPLOAD_FOLDER, v, f"{user_video_name}.mp4")
            if not os.path.exists(user_video_path):
                return jsonify({"error": f"缺少人像视频文件: {user_video_path}"}), 404
            clip = VideoFileClip(user_video_path).set_fps(30).resize((3840, 2160))
            clips.append(clip)
        except Exception as e:
            return jsonify({"error": f"视频列表中未找到{v}或索引错误: {e}"}), 400

    # 插入预制景区视频（按你原始文件名）
    prepared_names = [
        '20240424_C1720.MP4',
        '20240424_C1640.MP4',
        '20240424_C1672.MP4',
        '20240424_C1694.MP4'
    ]
    prepared_paths = [os.path.join(PREPARED_FOLDER, n) for n in prepared_names]
    # 插入到特定位置以匹配原脚本（0,2,4,6）
    for i, p in enumerate(prepared_paths):
        if not os.path.exists(p):
            # 如果预设视频不存在，跳过或返回错误（这里跳过）
            print(f"prepared video missing: {p}, skipping")
            continue
        insert_idx = i * 2  # 0,2,4,6
        clips.insert(insert_idx, VideoFileClip(p).set_fps(30).resize((3840, 2160)))

    # 合并
    final_clip = concatenate_videoclips(clips, method='compose')

    # 背景音乐叠加
    music_path = os.path.join(BACKGROUNDMUSIC_FOLDER, f"{music_type}.mp3")
    if not os.path.exists(music_path):
        return jsonify({"error": f"音乐文件未找到: {music_path}"}), 404

    background_music = AudioFileClip(music_path)
    original_audio = final_clip.audio

    if original_audio is not None:
        final_audio = CompositeAudioClip([original_audio.volumex(0.5), background_music.volumex(0.8)])
        final_clip = final_clip.set_audio(final_audio)
    else:
        final_clip = final_clip.set_audio(background_music)

    final_clip = final_clip.set_fps(30)

    output_video = f"{user_id}_output.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_video)
    # write file (blocking)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=30)

    # close resources
    for clip in clips:
        clip.close()
    background_music.close()
    final_clip.close()

    return jsonify({'message': 'Videos processed successfully', 'output': f'/outputs/{output_video}'})

def preview_output(filename):
    # send_from_directory returns a response
    return send_from_directory(OUTPUT_FOLDER, filename)

def download_video_file(request):
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "缺少 userId 参数"}), 400

    output_video = f"{user_id}_output.mp4"
    video_path = os.path.join(OUTPUT_FOLDER, output_video)
    if not os.path.exists(video_path):
        return jsonify({"error": f"未找到用户 {user_id} 的视频文件"}), 404

    return send_file(video_path, as_attachment=True)
