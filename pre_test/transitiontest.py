import cv2
from moviepy.editor import VideoClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips
import numpy as np
import moviepy.video.fx.all as vfx


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
    final_clip = concatenate_videoclips(
        [clip1.set_end(clip1.duration - duration), transition, clip2.set_start(duration)],
        method="compose")
    return final_clip

# 加载视频并应用波浪转场
# final_video = wave_transition(video1, video2, duration=1)
# final_video.write_videofile("./example_video/output_wave_transition.mp4", codec="libx264", fps=24)

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

# 示例视频加载与导出
video1 = VideoFileClip("./example_video/tiantan.mp4")
video2 = VideoFileClip("./example_video/man.mp4")

# final_video = sliding_with_blur_transition(video1, video2, duration=1)
# final_video.write_videofile("./example_video/output_sliding_blur.mp4", codec="libx264", fps=24)


# 加载视频并应用转场
# final_video = rotate_transition(video1, video2, duration=1)
# final_video.write_videofile("./example_video/output_rotate_transition.mp4", codec="libx264", fps=24)

clip1 = VideoFileClip("./example_video/tiantan.mp4")
clip2 = VideoFileClip("./example_video/man.mp4")

# 自定义模糊效果函数
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


# 自定义动态模糊滤镜
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

# 定义推近转场并加入动态模糊
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


"""
# 创建推近转场并加入动态模糊
transition_clip = zoom_with_blur_transition(clip1, clip2, duration=2, max_blur=15)

# 合并视频
final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - 2), transition_clip, clip2.subclip(2)])

# 导出视频
final_clip.write_videofile("./example_video/output_zoom_blur_transition.mp4", fps=24)
"""


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

"""
transition_clip = slide_with_blur_transition(clip1, clip2, duration=2, max_blur=15)

# 合并视频
final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - 2), transition_clip, clip2.subclip(2)])

# 导出视频
final_clip.write_videofile("./example_video/output_slide_blur_transition.mp4", fps=24)
"""


def cross_dissolve_transition(clip1, clip2, duration=2):
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

"""
# 创建画面融合转场
transition_clip = cross_dissolve_transition(clip1, clip2, duration=2)

# 合并视频
final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - 2), transition_clip, clip2.subclip(2)])

# 导出视频
final_clip.write_videofile("./example_video/output_cross_dissolve_transition.mp4", fps=24)
"""

"""
切割转场
"""

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


def transition_frame(t, img1, img2, duration, percent_func):
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

def create_transition_clip(clip1, clip2, duration=0.4):
    percent_func = percent_func_gen(a=0, b=1, time=duration, n=3, mode="slower")
    img1 = clip1.get_frame(0)
    img2 = clip2.get_frame(0)

    def make_frame(t):
        return transition_frame(t, img1, img2, duration, percent_func)

    return VideoClip(make_frame, duration=duration)

def percent_func_gen(a, b, time, n, mode):
    """
    高次多项式计算函数生成器
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

def add_padding(img, padding_factor=2):
    """
    给图像添加边框，防止旋转时出现黑边
    :param img: 输入图像
    :param padding_factor: 边框的比例，旋转角度越大，padding_factor 应越大
    :return: 添加边框后的图像
    """
    h, w = img.shape[:2]
    pad_h = int(h * padding_factor)
    pad_w = int(w * padding_factor)
    padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

def prepare_extended_image(img, scale_factor=2):
    """
    准备扩展图像以避免旋转时的黑边，并控制图像大小
    :param img: 输入图像
    :param scale_factor: 扩展比例（越大越能避免黑边，但需要控制尺寸）
    :return: 扩展后的图像
    """
    h, w = img.shape[:2]
    # 控制扩展图像的大小，确保其尺寸不超过 OpenCV 限制
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 重新调整图像大小
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建扩展图像
    img_ru = cv2.flip(img_resized, 1)
    img_ld = cv2.flip(img_resized, 0)
    img_rd = cv2.flip(img_resized, -1)
    img_u = np.hstack([img_resized, img_ru])
    img_d = np.hstack([img_ld, img_rd])
    img_res_tmp = np.vstack([img_u, img_d])
    img_res = np.hstack([img_res_tmp] * 2)  # 控制堆叠次数
    return img_res


def crop_to_center(img, center, target_size):
    """
    从图像中裁剪指定大小的区域
    :param img: 输入图像
    :param center: 中心点 (x, y)
    :param target_size: 裁剪大小 (width, height)
    :return: 裁剪后的图像
    """
    h, w = img.shape[:2]
    x, y = center
    tw, th = target_size
    x1, y1 = max(x - tw // 2, 0), max(y - th // 2, 0)
    x2, y2 = min(x + tw // 2, w), min(y + th // 2, h)
    return img[y1:y2, x1:x2]


def rotate_frame(t, img1, img2, duration, percent_func1, percent_func2, rows, cols):
    angle_all = 150  # 最大旋转角度
    res_rows, res_cols = img1.shape[:2]

    if t <= duration / 2:
        percent = percent_func1(t)
        angle = percent * angle_all
        center = (cols * 3, rows * 3)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(img1, M, (res_cols, res_rows))
    else:
        percent = percent_func2(t - duration / 2)
        angle = -percent * angle_all
        center = (cols * 4, rows * 3)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(img2, M, (res_cols, res_rows))

    # 裁剪到目标大小
    rotated = crop_to_center(rotated, center=(cols * 2, rows * 2), target_size=(cols, rows))
    return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)


def create_transition_clip(clip1, clip2, duration=0.4):
    """
    创建旋转转场的moviepy剪辑
    """
    img1 = clip1.get_frame(0)
    img2 = clip2.get_frame(0)
    rows, cols = img1.shape[:2]

    img1_res = prepare_extended_image(img1)
    img2_res = prepare_extended_image(img2)

    percent_func1 = percent_func_gen(a=0, b=1, time=duration / 2, n=4, mode="faster")
    percent_func2 = percent_func_gen(a=1, b=0, time=duration / 2, n=4, mode="slower")

    def make_frame(t):
        return rotate_frame(t, img1_res, img2_res, duration, percent_func1, percent_func2, rows, cols)

    return VideoClip(make_frame, duration=duration)



# # 创建左右切割转场效果
# transition = create_transition_clip(clip1, clip2, duration=0.4)
#
# # 合并剪辑
# final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.4), transition, clip2])
#
# # 导出结果
# final_clip.write_videofile(output_path, fps=24)

"""
# 创建旋转转场效果
transition = create_transition_clip(clip1, clip2, duration=1.2)
transition.write_videofile("./transition_video/transition_1.mp4", fps=24)
# 合并剪辑
final_clip = concatenate_videoclips([clip1.subclip(0, clip1.duration - 0.8), transition, clip2])
# 导出结果
final_clip.write_videofile(output_path, fps=24)
"""

def erasing_transition(get_frame1, get_frame2, duration, fps):
    """
    实现擦除转场效果
    :param get_frame1: 第一个视频帧的获取函数
    :param get_frame2: 第二个视频帧的获取函数
    :param duration: 转场持续时间（秒）
    :param fps: 视频帧率
    :return: 转场效果的函数
    """
    percent_func = percent_func_gen(0, 1, duration, n=1, mode="null")
    frame_count = int(duration * fps)

    def make_frame(t):
        percent = percent_func(t)
        frame1 = get_frame1(t)
        frame2 = get_frame2(t)

        frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
        frame2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)

        rows, cols = frame1.shape[:2]
        height = int(percent * rows)

        # 替换 frame1 的部分区域
        frame1[:height, :] = frame2[:height, :]

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




# 主程序
# output_path = "./transition_video/output_rotation_transition_1.mp4"

clip1 = VideoFileClip("./example_video/tiantan.mp4")
clip2 = VideoFileClip("./example_video/man.mp4")

# 转场持续时间
transition_duration = 1  # 转场持续时间（秒）
fps = video1.fps

"""
# 创建转场效果
transition_clip = VideoClip(
    erasing_transition(video1.get_frame, video2.get_frame, transition_duration, fps),
    duration=transition_duration,
)

# 拼接视频
final_video = concatenate_videoclips([
    video1.subclip(0, video1.duration - transition_duration / 2),
    transition_clip,
    video2.subclip(transition_duration / 2, video2.duration)
])

# 保存结果
final_video.write_videofile("./transition_video/output_with_erasing_transition.mp4", fps=fps)
"""

"""
curtain效果
# 创建转场效果
transition_clip = VideoClip(
    curtain_transition(video1.get_frame, video2.get_frame, transition_duration, fps),
    duration=transition_duration,
)

# 拼接视频
final_video = concatenate_videoclips([
    video1.subclip(0, video1.duration - transition_duration / 2),
    transition_clip,
    video2.subclip(transition_duration / 2, video2.duration)
])

# 保存结果
final_video.write_videofile("./transition_video/output_with_curtain_transition.mp4", fps=fps)

"""



# 创建转场效果
transition_clip = VideoClip(
    center_expand_transition(video1.get_frame, video2.get_frame, transition_duration, fps),
    duration=transition_duration,
)

# 拼接视频
final_video = concatenate_videoclips([
    video1.subclip(0, video1.duration - transition_duration / 2),
    transition_clip,
    video2.subclip(transition_duration / 2, video2.duration)
])

# 保存结果
final_video.write_videofile("./transition_video/output_with_center_expand_transition.mp4", fps=fps)


