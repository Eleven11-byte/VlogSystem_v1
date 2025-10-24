import numpy as np
import cv2
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageSequenceClip, concatenate_videoclips

def make_custom_transition(clip1, clip2, transition_type, duration, fps=24):
    frames = []
    total_frames = int(duration * fps)
    w, h = clip1.size

    for i in range(total_frames):
        t = i / total_frames
        f1 = cv2.cvtColor(clip1.get_frame(i / fps), cv2.COLOR_RGB2BGR)
        f2 = cv2.cvtColor(clip2.get_frame(i / fps), cv2.COLOR_RGB2BGR)

        if transition_type == "shutter":  # 横向百叶窗
            stripes = 10
            step = int(h / stripes)
            frame = f1.copy()
            for j in range(stripes):
                if t > j / stripes:
                    frame[j * step:(j + 1) * step, :] = f2[j * step:(j + 1) * step, :]

        elif transition_type == "circle":
            radius = int(np.sqrt(w ** 2 + h ** 2) * t)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
            frame = np.where(mask[:, :, None] == 255, f2, f1)

        elif transition_type == "pixelate":
            k = max(1, int((1 - t) * 50))
            temp = cv2.resize(f1, (w // k, h // k), interpolation=cv2.INTER_LINEAR)
            temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            frame = np.where(t > 0.5, f2, temp)

        elif transition_type == "block":
            grid_size = 10
            block_w, block_h = w // grid_size, h // grid_size
            frame = f1.copy()
            np.random.seed(42)  # 固定随机性
            indices = np.arange(grid_size * grid_size)
            np.random.shuffle(indices)
            num_blocks = int(t * grid_size * grid_size)
            for idx in indices[:num_blocks]:
                x = (idx % grid_size) * block_w
                y = (idx // grid_size) * block_h
                frame[y:y + block_h, x:x + block_w] = f2[y:y + block_h, x:x + block_w]

        else:
            raise ValueError(f"Unsupported complex transition: {transition_type}")

        frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return ImageSequenceClip(frames, fps=fps)


def apply_transition(clip1, clip2, transition_type='fade', duration=1):
    """
    Apply transition between two clips.
    Supported transitions:
    - fade
    - crossfade
    - slide
    - shutter
    - circle
    - pixelate
    - block
    """
    fps = clip1.fps if hasattr(clip1, 'fps') else 24
    transition_type = transition_type.lower()

    if transition_type == 'fade':
        return [clip1.fadeout(duration), clip2.fadein(duration)]

    elif transition_type == 'crossfade':
        clip1_mod = clip1.crossfadeout(duration)
        clip2_mod = clip2.crossfadein(duration).set_start(clip1.duration - duration)
        return [clip1_mod.set_end(clip1.duration), clip2_mod]

    elif transition_type == 'slide':
        def slide_transition(get_frame, t):
            if t < duration:
                dx = int(clip1.w * t / duration)
                f1 = clip1.get_frame(t)
                f2 = clip2.get_frame(t)
                f = f1.copy()
                f[:, dx:] = f2[:, :clip1.w - dx]
                return f
            else:
                return clip2.get_frame(t - duration)

        transition_clip = clip1.set_duration(duration).set_make_frame(slide_transition)
        clip2 = clip2.set_start(duration)
        return [transition_clip, clip2]


    elif transition_type in ['shutter', 'circle', 'pixelate', 'block']:
        transition_clip = make_custom_transition(clip1, clip2, transition_type, duration, fps=fps)
        clip2 = clip2.set_start(duration)
        return [transition_clip, clip2]

    else:
        raise ValueError(f"Unsupported transition type: {transition_type}")

import numpy as np
import cv2
from moviepy.editor import VideoClip

def apply_transition(clip1, clip2, transition_type="fade", duration=1.0):
    fps = 24
    total_frames = int(duration * fps)
    w, h = clip1.w, clip1.h
    frames = []

    def fisheye_distort(img, t_strength):
        h, w = img.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
        D = np.array([t_strength, 0, 0, 0], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def ripple(img, strength):
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = 10 * np.sin(2 * np.pi * y_indices / 128.0 + strength * 5)
        dy = 10 * np.sin(2 * np.pi * x_indices / 128.0 + strength * 5)
        map_x = (x_indices + dx).clip(0, w - 1)
        map_y = (y_indices + dy).clip(0, h - 1)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    for i in range(total_frames):
        t = i / total_frames
        f1 = cv2.cvtColor(clip1.get_frame(i / fps), cv2.COLOR_RGB2BGR)
        f2 = cv2.cvtColor(clip2.get_frame(i / fps), cv2.COLOR_RGB2BGR)

        if transition_type == "fade":
            blended = cv2.addWeighted(f1, 1 - t, f2, t, 0)

        elif transition_type == "crossfade":
            blended = cv2.addWeighted(f1, 1 - t, f2, t, 0)

        elif transition_type == "slide":
            offset = int(w * t)
            blended = np.zeros_like(f1)
            blended[:, :w - offset] = f1[:, offset:]
            blended[:, w - offset:] = f2[:, :offset]

        elif transition_type == "circle":
            mask = np.zeros((h, w), dtype=np.uint8)
            radius = int(np.hypot(w, h) * t)
            cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
            mask = cv2.merge([mask] * 3)
            blended = np.where(mask == 255, f2, f1)

        elif transition_type == "shutter":
            lines = 10
            step = h // lines
            blended = f1.copy()
            for j in range(lines):
                y_start = j * step
                y_end = y_start + int(step * t)
                blended[y_start:y_end, :] = f2[y_start:y_end, :]

        elif transition_type == "pixelate":
            scale = int(50 * (1 - t)) + 1
            f1_small = cv2.resize(f1, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
            f1_pix = cv2.resize(f1_small, (w, h), interpolation=cv2.INTER_NEAREST)
            blended = cv2.addWeighted(f1_pix, 1 - t, f2, t, 0)

        elif transition_type == "block":
            blocks = 10
            bw, bh = w // blocks, h // blocks
            blended = f1.copy()
            for y in range(blocks):
                for x in range(blocks):
                    if np.random.rand() < t:
                        blended[y * bh:(y + 1) * bh, x * bw:(x + 1) * bw] = f2[y * bh:(y + 1) * bh, x * bw:(x + 1) * bw]

        elif transition_type == "fisheye":
            strength = 0.5 * (1 - t)
            f1_distort = fisheye_distort(f1, strength)
            f2_distort = fisheye_distort(f2, -strength)
            blended = cv2.addWeighted(f1_distort, 1 - t, f2_distort, t, 0)

        elif transition_type == "zoom":
            zoom_factor = 1 + 0.5 * (1 - t)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
            f1_zoom = cv2.warpAffine(f1, M, (w, h))
            blended = cv2.addWeighted(f1_zoom, 1 - t, f2, t, 0)

        elif transition_type == "rotate":
            angle = 180 * t
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            f1_rot = cv2.warpAffine(f1, M, (w, h))
            blended = cv2.addWeighted(f1_rot, 1 - t, f2, t, 0)

        elif transition_type == "flip":
            direction = 1 if i % 2 == 0 else 0
            f1_flip = cv2.flip(f1, direction)
            blended = cv2.addWeighted(f1_flip, 1 - t, f2, t, 0)

        elif transition_type == "ripple":
            f1_ripple = ripple(f1, t)
            f2_ripple = ripple(f2, 1 - t)
            blended = cv2.addWeighted(f1_ripple, 1 - t, f2_ripple, t, 0)

        else:
            blended = cv2.addWeighted(f1, 1 - t, f2, t, 0)

        frame_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    transition_clip = VideoClip(lambda t: frames[int(t * fps)], duration=duration)
    clip1_part = clip1.subclip(0, clip1.duration - duration)
    clip2_part = clip2.subclip(duration, clip2.duration) if clip2.duration > duration else clip2.subclip(0, 0)

    return [clip1_part, transition_clip, clip2_part]



# 加载两个视频并统一分辨率
clip1 = VideoFileClip("../prepareds/20240424_C1640.MP4").resize((1280, 720))
clip2 = VideoFileClip("../prepareds/20240424_C1672.MP4").resize((1280, 720))

# 应用一个转场，例如 "circle"（你也可以替换为 "fade"、"block"、"shutter"、"pixelate"、"crossfade"、"slide"）
transitioned_clips = apply_transition(clip1, clip2, transition_type="ripple", duration=1.5)

# 拼接视频
# final_video = concatenate_videoclips(transitioned_clips, method="compose")

# 导出结果
# final_video.write_videofile("./transition_video/transition_output/fisheye.mp4", fps=24)
transitioned_clips[1].write_videofile("./transition_video/transition_output/ripple_transition.mp4", fps=24)