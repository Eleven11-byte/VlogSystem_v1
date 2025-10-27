"""
实现识别某帧视频中的行人及位置，限制开始条件
触发条件开始录制往后延续15s

"""

from AIDetector_pytorch import Detector
import imutils
import cv2
from moviepy.editor import *
import time
import keyboard

class DetectVisitor:
    """
    用于检测是否有行人出现在画面中央的类
    """
    def __init__(self, condition):
        self.condition = condition  # condition = [左边缘，右边缘，宽，高]
        self.detector = Detector()

    def detect_frame(self, img):
        result = self.detector.feedCap(img)
        bboxes = result['bboxes2draw']

        #if len(bboxes) > 0:
        '''
        if keyboard.is_pressed("space"):
            return True
        else:  # 没有检测到人
            return False
        '''
        return result

    def detect(self, img):
        """
        返回布尔值，是否满足检测条件
        """
        result = self.detector.feedCap(img)
        bboxes = result['bboxes2draw']
        # print(bboxes)

        if len(bboxes) > 0:  # 屏幕中检测到了不止一个人
            for bbox in bboxes:
                x1, y1, x2, y2, _, _ = bbox
                print(bbox)
                # TODO: 完善判断条件
                # 判断条件：有一个人物进入画面主要区域，且宽、高都大于设置值
                if x1 > self.condition[0] and x2 < self.condition[1] and abs(x1 - x2) > self.condition[2] and abs(y1 - y2) > self.condition[3]:
                    return True
                else:
                    return False
        else:
            return False
