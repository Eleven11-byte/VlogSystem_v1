# services/video_service.py
import os
import numpy as np
from flask import jsonify, send_from_directory, send_file
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip, ImageClip, CompositeVideoClip
from config import (FACE_FEATURE_FOLDER, FEATURES_FOLDER, UPLOAD_FOLDER, PREPARED_FOLDER,
                    BACKGROUNDMUSIC_FOLDER, OUTPUT_FOLDER, VIEW_POSITIONS, THRESHOLD, OUTPUT_WATERMARK_FOLDER, FACE_FOLDER)
import face_recognize  # 保持原有接口
from utils_app import reencode_video  # 可能用不到，但保留

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
        # 在每个景点的视频目录下调用
        feature_path = os.path.join(FEATURES_FOLDER, view_position)
        # use face_recognize.find_similar if you have it
        video_list = face_recognize.find_similar_face(feature_path, target_embedding, threshold)
        if len(video_list) == 0:
            video_list_result.append(None)
        else:
            video_list_result.append(video_list[-1])
        video_list_all.append(video_list)

    return video_list_result, video_list_all

def add_watermark_ffmpeg(input_video_path, output_video_path, watermark_text="预览水印"):
    """
    使用FFmpeg直接添加水印 - 最快方案

    :param input_video_path: 输入视频路径
    :param output_video_path: 输出视频路径
    :param watermark_text: 水印文本
    """
    import subprocess
    import time

    start_time = time.time()

    # FFmpeg命令添加文字水印
    cmd = [
        'ffmpeg', '-i', input_video_path,
        '-vf', f"drawtext=text='{watermark_text}':fontfile=util/msyh.ttc:fontsize=36:fontcolor=white@0.3:x=(w-text_w)/2:y=(h-text_h)/2",
        '-c:a', 'copy',  # 音频直接复制，不重新编码
        '-preset', 'ultrafast',
        '-crf', '23',
        '-y',  # 覆盖输出文件
        output_video_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        end_time = time.time()
        print(f"FFmpeg水印添加完成，耗时：{end_time - start_time:.2f}秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg处理失败：{e}")
        return False

def add_watermark(input_video_path, output_video_path, watermark_text="预览水印 - 付费解锁高清无水印"):
    """
    给视频添加水印

    :param input_video_path: 输入视频路径
    :param output_video_path: 输出视频路径
    :param watermark_text: 水印文本
    """
    # 加载输入视频
    video = VideoFileClip(input_video_path)
    w, h = video.size  # 获取视频宽高

    # 创建水印图像
    from PIL import Image, ImageDraw, ImageFont
    import math

    font = ImageFont.truetype("watermark/msyh.ttc", 36)  # 黑体

    # 计算水印文字大小
    bbox = font.getbbox(watermark_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 创建单个水印图像（带透明背景）
    watermark_img = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_img)
    draw.text((10, 10), watermark_text, fill=(255, 255, 255, 80), font=font)  # 半透明白色

    # 旋转水印
    watermark_img = watermark_img.rotate(30, expand=True)
    watermark_path = "watermark/watermark.png"
    watermark_img.save(watermark_path)

    # 创建多个水印实例，斜向铺满屏幕
    watermarks = []

    # 获取旋转后水印的实际尺寸
    rotated_img = Image.open(watermark_path)
    rotated_width, rotated_height = rotated_img.size

    # 设置更大的间距，避免重叠
    spacing_x = rotated_width + 15  # 水印间距（水印宽度 + 额外间距）
    spacing_y = rotated_height + 15  # 水印间距（水印高度 + 额外间距）

    # 计算需要的水印数量，确保铺满屏幕
    rows = (h // spacing_y) + 2
    cols = (w // spacing_x) + 2

    for row in range(rows):
        for col in range(cols):
            # 计算每个水印的位置
            x = col * spacing_x - spacing_x // 2
            y = row * spacing_y - spacing_y // 2

            # 交错排列，形成更好的视觉效果
            if row % 2 == 1:
                x += spacing_x // 2

            # 确保水印不会超出屏幕边界太多
            if x < w + rotated_width and y < h + rotated_height and x > -rotated_width and y > -rotated_height:
                watermark = ImageClip(watermark_path).set_duration(video.duration).set_position((x, y))
                watermarks.append(watermark)

    # 合成视频（原视频 + 所有水印）
    watermarked_video = CompositeVideoClip([video] + watermarks)
    watermarked_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # 释放资源
    video.close()
    # 先释放所有水印实例
    for watermark in watermarks:
        watermark.close()
    watermarked_video.close()

    # 不删除临时水印文件，避免文件占用问题
    print(f"水印添加完成，保留临时文件： {watermark_path}")

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
    print(videoOfUser)

    # 把对应的视频加入 clips（与原代码逻辑保持一致）
    clips = []
    # 处理每个 view (view1, view2, view3)


    for v in VIEW_POSITIONS:
        # find index in videoOfUser
        try:
            # TODO：取出对应的video的idx
            idx = find_video_indices(videoOfUser, v)
            user_video_name = videoOfUser[idx]
            user_video_path = os.path.join(UPLOAD_FOLDER, v, f"{user_video_name}")
            print(f"对应视频文件：{user_video_path}")
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

    # 添加水印
    output_watermark_path = os.path.join(OUTPUT_WATERMARK_FOLDER, output_video)
    add_watermark_ffmpeg(output_path, output_watermark_path, "智旅VLOG")
    print("添加水印完成，输出路径：", output_watermark_path)

    # close resources
    for clip in clips:
        clip.close()
    background_music.close()
    final_clip.close()

    return jsonify({'message': 'Videos processed successfully', 'output': f'/outputs/{output_video}'})


def preview_output(filename):
    # send_from_directory returns a response
    return send_from_directory(OUTPUT_WATERMARK_FOLDER, filename)

def download_video_file(request):
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "缺少 userId 参数"}), 400

    output_video = f"{user_id}_output.mp4"
    video_path = os.path.join(OUTPUT_FOLDER, output_video)
    if not os.path.exists(video_path):
        return jsonify({"error": f"未找到用户 {user_id} 的视频文件"}), 404

    return send_file(video_path, as_attachment=True)
