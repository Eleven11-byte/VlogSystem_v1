"""
测试算法剪辑录制视频效果
"""

import os
import numpy as np
import cv2

from AIDetector_pytorch import Detector
import imutils
from moviepy.editor import *


video_path = './example_video/scene/Clip0053.mp4'
video_name = video_path.split('/')[-1].split('.')[0]

cap = cv2.VideoCapture(video_path)
det = Detector()

fps = int(cap.get(5))
print('fps:', fps)
t = int(1000 / fps)
duration = 15

frame_id = 0
detected = False
appear_frame = 0


# 用于输入视频剪辑
while True:
    # try:
    _, im = cap.read()  # 读取每一帧
    frame_id += 1
    if im is None:
        break

    condition = [180, 3660, 480, 1200] # 45
    # condition = [180, 3660, 300, 900]  # 46/53
    # condition = [180, 3660, 300, 900]  # 47
    # condition = [180, 3660, 1200, 1600] # 48/49/50

    while True:

        # try:
        _, im = cap.read()  # 读取每一帧
        frame_id += 1
        if im is None:
            break

        result = det.feedCap(im)

        bboxes = result['bboxes2draw']

        # 输入视频剪辑
        if not detected:
            if len(bboxes) > 0:
                for bbox in bboxes:
                    x1, y1, x2, y2, _, _ = bbox
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    # temp = [x1, x2, width, height]
                    # if x1 > condition[0]:
                    #    print(temp)
                    if x1 > condition[0] and x2 < condition[1] and width > condition[2] \
                            and height > condition[3]:  # 判断条件，根据实际需求修改
                        detected = True
                        appear_frame = frame_id
                        # disappear_frame = frame_num
                        # disappear_flag = True
                        # debug print appear_frame
                        print("appear_frame: " + str(appear_frame))
                        print(bbox)
                        cv2.imwrite(f"./example_frames/cut/{video_name}_start_frame_1.jpg", im)
                        break

        else:
            break

    cap.release()
    print(appear_frame)



# 输入视频剪辑
start_time = appear_frame / fps
end_time = start_time + duration

clip = VideoFileClip(video_path)
cut_clip = clip.subclip(start_time, end_time)
cut_clip.write_videofile(f'./example_video_result/{video_name}_cut_video_2.mp4')
