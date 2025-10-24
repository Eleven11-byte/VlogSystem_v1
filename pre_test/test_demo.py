"""
测试算法剪辑录制视频效果
"""

from AIDetector_pytorch import Detector
import imutils
import cv2
from moviepy.editor import *


def main():
    name = 'demo'

    video_path = 'D:/Document/School/2024fall/VlogSystem/sucai_video/Clip0045.mp4'

    det = Detector()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000 / fps)
    duration = 15

    # 用于输入视频剪辑
    frame_id = 0
    detected = False
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    appear_frame = 0


    condition = [180, 3660, 420, 1200]

    while True:

        # try:
        _, im = cap.read()  # 读取每一帧
        frame_id += 1
        if im is None:
            break

        result = det.feedCap(im)

        bboxes = result['bboxes2draw']

        cv2.imwrite("./example_frames/test0045.jpg", im)

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
                            and height > condition[3]: # 判断条件，根据实际需求修改
                        detected = True
                        appear_frame = frame_id
                        # disappear_frame = frame_num
                        # disappear_flag = True
                        # debug print appear_frame
                        print("appear_frame: " + str(appear_frame))
                        print(bbox)
                        cv2.imwrite("./example_frames/Clip0045_start_frame.jpg", im)
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
    cut_clip.write_videofile('result/Clip0046_cut_video_1.mp4')


if __name__ == '__main__':
    # video_path = './example_video/Clip0045.mp4'
    #
    # start_time = 195/25
    # end_time = start_time + 15
    #
    # clip = VideoFileClip(video_path)
    # cut_clip = clip.subclip(start_time, end_time)
    # cut_clip.write_videofile('result/Clip0046_cut_video_1.mp4')
    main()