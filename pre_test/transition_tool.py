import cv2
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, VideoClip, CompositeAudioClip, AudioFileClip
import numpy as np


def wave_transition(clip1, clip2, duration=1):
    """
    实现水波转场效果
    """
    def wave_effect(get_frame, t):
        """
        应用于每帧的波浪效果
        :param get_frame: 获取当前帧的函数
        :param t: 当前时间
        :return: 添加波浪效果后的帧
        """
        frame = get_frame(t)
# 获取当前帧
        h, w, _ = frame.shape
# 获取帧的高度和宽度
        wave = np.sin(2 * np.pi * (np.arange(h)[:, None] / h + t / duration))  # 波浪形状
        displacement = (wave * 20).astype(int)  # 波浪幅度
        return np.array([np.roll(frame[y], displacement[y], axis=0) for y in range(h)])

    # 对第一个视频应用波浪效果
    wave_clip1 = clip1.fl(wave_effect).set_duration(duration)
    # 对第二个视频应用波浪效果
    wave_clip2 = clip2.fl(wave_effect).set_duration(duration).set_start(duration)

    # 组合转场
    transition = CompositeVideoClip([wave_clip1, wave_clip2], size=clip1.size)

    return transition

def rotate_transition(clip1, clip2, duration=1):
    """
    实现旋转转场效果
    """
    # 第一个视频旋转消失
    rotate_out = clip1.rotate(lambda t: -360 * t / duration).set_duration(duration)

    # 第二个视频旋转进入
    rotate_in = clip2.rotate(lambda t: 360 - 360 * t / duration).set_duration(duration).set_start(duration)

    # 组合转场
    transition = CompositeVideoClip([rotate_out, rotate_in])
    return concatenate_videoclips([clip1.set_end(clip1.duration - duration), transition, clip2.set_start(duration)], method="compose")

def sliding_with_blur_transition(clip1, clip2, duration=1):
    """
    实现滑动转场，并添加动态模糊效果
    :param clip1: 第一个视频片段
    :param clip2: 第二个视频片段
    :param duration: 转场持续时间（秒）
    :return: 添加动态模糊的转场视频
    """
    def slide_with_blur(get_frame, t):
        """
        滑动转场的帧处理函数，添加动态模糊
        :param get_frame: 获取帧的函数
        :param t: 当前时间
        :return: 带模糊的帧
        """
        frame1 = clip1.get_frame(t) if t < duration else np.zeros_like(clip1.get_frame(0))
        frame2 = clip2.get_frame(t - duration) if t >= 0 else np.zeros_like(clip2.get_frame(0))

        # 计算滑动位置
        w, h = clip1.size
        offset = int(w * t / duration) if t < duration else w
        position1 = max(-w, -offset)  # clip1 从左向右移动
        position2 = position1 + w  # clip2 从右向左移动

        # 创建空白帧
        combined_frame = np.zeros_like(frame1)

        # 合并滑动位置帧
        if position1 + w > 0:  # clip1 可见部分
            combined_frame[:, :max(0, position1 + w)] = frame1[:, max(0, -position1):w]
        if position2 < w:  # clip2 可见部分
            combined_frame[:, max(0, position2):w] = frame2[:, :max(0, w - position2)]

        # 动态模糊计算 (简单线性叠加模拟模糊效果)
        alpha = abs(position1 / w)
        blurred_frame = (frame1 * (1 - alpha) + frame2 * alpha).astype("uint8")
        return blurred_frame

    # 创建动态模糊滑动效果
    sliding_clip = clip1.fl(slide_with_blur).set_duration(duration)

    # 合并视频
    final_clip = concatenate_videoclips([clip1.set_end(clip1.duration - duration), sliding_clip, clip2.set_start(duration)], method="compose")
    return final_clip

"""
实现动态模糊的函数
"""

def apply_blur(frame, blur_strength):
    """
    对帧应用动态模糊。
    :param frame: 视频帧 (numpy array)
    :param blur_strength: 模糊强度
    :return: 模糊处理后的帧
    """
    if blur_strength <= 0:
        return frame
    ksize = int(blur_strength) * 2 + 1  # 确保核大小为奇数
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def blur_effect(clip, max_blur=10):
    """
    为视频剪辑添加动态模糊效果。
    :param clip: 视频剪辑
    :param max_blur: 最大模糊强度
    :return: 应用模糊效果的视频剪辑
    """

    def fl(gf, t):
        frame = gf(t)  # 获取当前帧
        blur_strength = max_blur * (t / clip.duration)  # 模糊强度随时间变化
        return apply_blur(frame, blur_strength)

    return clip.fl(fl)

def zoom_with_blur_transition(clip1, clip2, duration=2, max_blur=10):
    # 视频1：从正常尺寸逐渐缩小并加入动态模糊
    zoom_out = blur_effect(
        clip1.resize(lambda t: 1 - 0.5 * (t / duration)).set_duration(duration),
        max_blur=max_blur
    )

    # 视频2：从小尺寸逐渐放大到正常尺寸并加入动态模糊
    zoom_in = blur_effect(
        clip2.resize(lambda t: 0.5 + 0.5 * (t / duration)).set_duration(duration),
        max_blur=max_blur
    )

    # 合成转场视频
    transition = CompositeVideoClip([zoom_out.crossfadeout(duration), zoom_in.crossfadein(duration)])
    return transition

def slide_with_blur_transition(clip1, clip2, duration=2, max_blur=10):
    """
    创建滑动转场并加入动态模糊效果。
    :param clip1: 视频剪辑1
    :param clip2: 视频剪辑2
    :param duration: 转场持续时间
    :param max_blur: 最大模糊强度
    :return: 滑动转场剪辑
    """
    w, h = clip1.size

    # 视频1：从正常位置向左滑动并加入动态模糊
    slide_out = blur_effect(
        clip1.set_position(lambda t: (-w * (t / duration), 0)).set_duration(duration),
        max_blur=max_blur
    )

    # 视频2：从右侧滑入到正常位置并加入动态模糊
    slide_in = blur_effect(
        clip2.set_position(lambda t: (w * (1 - t / duration), 0)).set_duration(duration),
        max_blur=max_blur
    )

    # 合成转场视频
    transition = CompositeVideoClip([slide_out, slide_in], size=(w, h))
    return transition

# 淡入淡出
def fading_transition(clip1, clip2, duration=2):
    """
    实现画面融合（交叉叠化）转场效果。
    :param clip1: 视频剪辑1
    :param clip2: 视频剪辑2
    :param duration: 转场持续时间
    :return: 转场剪辑
    """
    # 视频1：透明度逐渐减小
    fade_out = clip1.set_duration(duration).crossfadeout(duration)

    # 视频2：透明度逐渐增大
    fade_in = clip2.set_duration(duration).crossfadein(duration)

    # 混合两段视频
    transition = CompositeVideoClip([fade_out, fade_in])
    return transition


# 用于实现切割转场
def percent_func_gen(a, b, time, n, mode):
    """
    高次多项式计算函数生成器
    :param a: 起始百分比（如：0.25）
    :param b: 结束百分比
    :param time: 动画持续时间
    :param n: 多项式次数
    :param mode: faster（越来越快）、slower（越来越慢）
    :return: 每个时刻到达百分比的计算函数
    """
    if mode == "slower":
        a, b = b, a
    delta = abs(a - b)
    sgn = 1 if b - a > 0 else (-1 if b - a < 0 else 0)

    def percent_calc(ti):
        if mode == "slower":
            ti = time - ti
        return sgn * delta / (time ** n) * (ti ** n) + a

    return percent_calc


def switch_transition_frame(t, img1, img2, duration, percent_func):
    rows, cols = img1.shape[:2]
    percent = percent_func(t)
    x = int(percent * cols)

    # 拼接画布
    img_u = np.hstack([img1, img2])
    img_d = np.hstack([img2, img1])

    # 上部分
    M1 = np.float32([[1, 0, -x], [0, 1, 0]])
    res1 = cv2.warpAffine(img_u, M1, (cols * 2, rows))[:rows // 2, :cols]

    # 下部分
    M2 = np.float32([[1, 0, x - cols], [0, 1, 0]])
    res2 = cv2.warpAffine(img_d, M2, (cols * 2, rows))[rows // 2:, :cols]

    # 合成
    res = np.vstack([res1, res2])
    # return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)  # 转为RGB格式，适配moviepy
    return res

def switch_transition(clip1, clip2, duration=0.4):
    percent_func = percent_func_gen(a=0, b=1, time=duration, n=3, mode="slower")
    img1 = clip1.get_frame(0)
    img2 = clip2.get_frame(0)

    def make_frame(t):
        return switch_transition_frame(t, img1, img2, duration, percent_func)

    return VideoClip(make_frame, duration=duration)

def sliding_transition(clip1, clip2, duration=2, direction="left"):
    """
    实现滑动转场效果。

    :param clip1: 第一个视频剪辑
    :param clip2: 第二个视频剪辑
    :param duration: 转场持续时间（秒）
    :param direction: 滑动方向（"left", "right", "up", "down"）
    :return: 添加滑动转场的视频剪辑
    """
    w, h = clip1.size

    # 确定滑动方向
    if direction == "left":
        start_pos1 = (0, 0)
        end_pos1 = (-w, 0)
        start_pos2 = (w, 0)
        end_pos2 = (0, 0)
    elif direction == "right":
        start_pos1 = (0, 0)
        end_pos1 = (w, 0)
        start_pos2 = (-w, 0)
        end_pos2 = (0, 0)
    elif direction == "up":
        start_pos1 = (0, 0)
        end_pos1 = (0, -h)
        start_pos2 = (0, h)
        end_pos2 = (0, 0)
    elif direction == "down":
        start_pos1 = (0, 0)
        end_pos1 = (0, h)
        start_pos2 = (0, -h)
        end_pos2 = (0, 0)
    else:
        raise ValueError("Invalid direction. Choose from 'left', 'right', 'up', or 'down'.")

    # 第一个视频滑出
    sliding_out = clip1.set_position(lambda t: (
        start_pos1[0] + (end_pos1[0] - start_pos1[0]) * (t / duration),
        start_pos1[1] + (end_pos1[1] - start_pos1[1]) * (t / duration)
    )).set_duration(duration)

    # 第二个视频滑入
    sliding_in = clip2.set_position(lambda t: (
        start_pos2[0] + (end_pos2[0] - start_pos2[0]) * (t / duration),
        start_pos2[1] + (end_pos2[1] - start_pos2[1]) * (t / duration)
    )).set_duration(duration).set_start(duration)

    # 合并转场
    transition = CompositeVideoClip([sliding_out, sliding_in], size=(w, h))
    return transition

def curtain_transition(get_frame1, get_frame2, duration, fps):
    """
    横向拉幕转场效果
    :param get_frame1: 第一个视频帧的获取函数
    :param get_frame2: 第二个视频帧的获取函数
    :param duration: 转场持续时间（秒）
    :param fps: 视频帧率
    :return: 转场效果的函数
    """
    percent_func = percent_func_gen(0, 1, duration, n=1, mode="null")

    def make_frame(t):
        percent = percent_func(t)
        frame1 = get_frame1(t)
        frame2 = get_frame2(t)

        frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
        frame2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)

        rows, cols = frame1.shape[:2]
        width = int(percent * cols)

        # 替换 frame1 的部分区域
        frame1[:, :width] = frame2[:, :width]

        return cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    return make_frame

def center_expand_transition(get_frame1, get_frame2, duration, fps):
    """
    从中心向两边展开的转场效果
    :param get_frame1: 第一个视频帧的获取函数
    :param get_frame2: 第二个视频帧的获取函数
    :param duration: 转场持续时间（秒）
    :param fps: 视频帧率
    :return: 转场效果的函数
    """
    percent_func = percent_func_gen(0, 1, duration, n=1, mode="null")

    def make_frame(t):
        percent = percent_func(t)
        frame1 = get_frame1(t)
        frame2 = get_frame2(t)

        frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
        frame2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)

        rows, cols = frame1.shape[:2]
        center = cols // 2
        width = int(percent * center)

        # 创建新的帧
        result = frame1.copy()
        result[:, center - width:center + width] = frame2[:, center - width:center + width]

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return make_frame

def make_transition(type, clip1, clip2, duration=2):
    if type == "fading":
        return fading_transition(clip1, clip2, duration=duration)
    elif type == "wave":
        return wave_transition(clip1, clip2, duration=duration)
    elif type == "rotate":
        return rotate_transition(clip1, clip2, duration=duration)
    elif type == "sliding":
        return VideoClip(curtain_transition(clip1.get_frame, clip2.get_frame, duration, fps=28),duration=duration,)
    elif type == "center_expand":
        return VideoClip(center_expand_transition(clip1.get_frame, clip2.get_frame, duration, fps=28),
                         duration=duration, )
    elif type == "switch":
        return switch_transition(clip1, clip2, duration=duration)
    elif type == "sliding_with_blur":
        return sliding_with_blur_transition(clip1, clip2, duration=duration)
    elif type == "zoom_with_blur":
        return zoom_with_blur_transition(clip1, clip2, duration=duration)
    else:
        raise ValueError(f"Unsupported transition type: {type}")



if __name__ == '__main__':

    save_path = "../transition_video/output/"
    # # 转场效果时间
    # transition_duration = 0.4
    # clip1 = VideoFileClip("./example_video/tiantan.mp4")
    # clip2 = VideoFileClip("./example_video/man.mp4")
    #
    # type = "switch"
    # transition_clip = make_transition(type, clip1, clip2, duration=transition_duration)
    # final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - transition_duration), transition_clip, clip2.subclip(transition_duration)])
    #
    # transition_clip.write_videofile(save_path + type + "_transition.mp4", fps=24)
    # final_clip.write_videofile(save_path + type + ".mp4", fps=24)

    # 人像
    man_clip1 = VideoFileClip("../transition_video/人像/1.mp4").subclip(2, 6)
    man_clip2 = VideoFileClip("../transition_video/人像/2.mp4").subclip(2, 6)
    man_clip3 = VideoFileClip("../transition_video/人像/3.mp4").subclip(2, 6)

    # view
    view_clip0 = VideoFileClip("../transition_video/竖屏/tiantan (4).mp4").subclip(1, 6)
    view_clip1 = VideoFileClip("../transition_video/竖屏/tiantan (1).mp4").subclip(1, 6)
    view_clip2 = VideoFileClip("../transition_video/竖屏/tiantan (2).mp4").subclip(1, 6)
    view_clip3 = VideoFileClip("../transition_video/竖屏/tiantan (3).mp4").subclip(1, 6)

    clip1 = view_clip0
    clip2 = man_clip1
    transition_clip = make_transition("fading", clip1, clip2, duration=0.6)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.6), transition_clip, clip2.subclip(0.6)])

    clip1 = final_video
    clip2 = view_clip1
    transition_clip = make_transition("fading", clip1, clip2, duration=0.6)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.6), transition_clip, clip2.subclip(0.6)])

    clip1 = final_video
    clip2 = man_clip2
    transition_clip = make_transition("switch", clip1, clip2, duration=0.8)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.8), transition_clip, clip2.subclip(0.8)])

    # transition_clip.write_videofile(save_path + "wavetest.mp4", codec='libx264', audio_codec='aac', fps=30)

    clip1 = final_video
    clip2 = view_clip2
    transition_clip = make_transition("sliding", clip1, clip2, duration=0.6)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.6), transition_clip, clip2.subclip(0.6)])

    clip1 = final_video
    clip2 = man_clip3
    transition_clip = make_transition("center_expand", clip1, clip2, duration=0.6)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.6), transition_clip, clip2.subclip(0.6)])

    clip1 = final_video
    clip2 = view_clip3
    transition_clip = make_transition("switch", clip1, clip2, duration=0.6)
    final_video = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.6), transition_clip, clip2.subclip(0.6)])

    final_clip = final_video


    original_audio = final_clip.audio
    background_music = AudioFileClip("../audio/bgm01_gufeng.mp3")

    # 音频处理
    if original_audio is not None:
        background_music = background_music.subclip(0, final_clip.duration)
        final_audio = CompositeAudioClip([original_audio.volumex(0.1), background_music.volumex(0.8)])
        final_clip = final_clip.set_audio(final_audio)
    else:
        print("Original audio is None, setting only background music.")
        final_clip = final_clip.set_audio(background_music)


    output_path = save_path + "output_with_music.mp4"
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=28)