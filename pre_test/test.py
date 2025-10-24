## 视频转场效果测试

# from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
#
# # 加载两个视频
#
# video1 = VideoFileClip("./example_video/tiantan.mp4")
# video2 = VideoFileClip("./example_video/man.mp4")
#
# # 为视频1添加淡出效果，为视频2添加淡入效果
# """
# video1 = video1.crossfadeout(1)  # 持续1秒的淡出
# video2 = video2.crossfadein(1)  # 持续1秒的淡入
# """
#
# # 拼接两个视频，并保持转场效果
# # final_video = concatenate_videoclips([video1, video2], method="compose")
#
# # 导出最终视频
# # final_video.write_videofile("./example_video/output_with_transition.mp4", codec="libx264", fps=24)
#
#
#
# def zoom_transition(clip1, clip2, duration=1):
#     """
#     实现缩放转场效果
#     :param clip1: 第一个视频片段
#     :param clip2: 第二个视频片段
#     :param duration: 转场持续时间（秒）
#     :return: 添加缩放转场效果的 CompositeVideoClip
#     """
#     zoom_out = clip1.resize(lambda t: max(0.01, 1 - t / duration)).set_duration(duration)
#     zoom_in = clip2.resize(lambda t: max(0.01, t / duration)).set_duration(duration).set_start(duration)
#
#     # 组合两个效果
#     transition = CompositeVideoClip([zoom_out, zoom_in])
#
#     return concatenate_videoclips([clip1.set_end(clip1.duration - duration), transition, clip2.set_start(duration)], method="compose")
#
#
# # 应用缩放转场
# final_video = zoom_transition(video1, video2, duration=1)
# final_video.write_videofile("./example_video/output_zoom_transition.mp4", codec="libx264", fps=24)

import os
from moviepy.editor import VideoFileClip

# 输入文件夹路径
input_dir = "../transition_video/transition_output"
# 输出文件夹路径
output_dir = "../transition_video/transition_output/result"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有视频文件
for filename in os.listdir(input_dir):
    if filename.endswith((".mp4", ".avi", ".mov")):
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"silent_{filename}")

        with VideoFileClip(file_path) as video:
            # 静音处理
            video = video.set_audio(None)
            # 导出视频
            video.write_videofile(output_path, codec="libx264", audio_codec="mp3")
            print(f"已处理：{filename} -> {output_path}")