"""
用于提取图片中人脸面部特征，比对人脸特征进行视频检索
"""

import numpy as np
from insightface.app import FaceAnalysis
from sklearn import preprocessing
import pickle
import os


class FaceExtractor:
    """
    提取图片中人脸的面部特征并存储
    """
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_sc')
        # 进行设置，resize后的大小
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract_features(self, image):
        faces = self.app.get(image)
        faces_embedding = []
        # 对画面中识别到的所有人脸提取特征并存储
        for i in range(len(faces)):
            embedding = faces[i]["embedding"]
            # 正则化
            embedding = np.array(embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            faces_embedding.append(embedding)
        faces_embedding = np.array(faces_embedding)
        # TODO: 异常处理，从图像中未能提取到人脸特征的情况
        return faces_embedding

def face_compare(feature1, feature2, threshold):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    # TODO:移除此处打印
    # print("dist:" + str(dist))
    if dist < threshold:
        return True
    else:
        return False

def find_similar(file_path, target_embedding, threshold):
    """
    根据threshold检索相似的视频，返回列表
    file_path：存储特征文件的位置
    特征文件的命名格式：1728891454_view1_frame_5s
    视频文件的命名格式：1728891454_view1.mp4
    """
    video_list = []

    files = os.listdir(file_path)
    for file in files:
        # FIXME: 修改文件名生成方式
        # print(file)
        faces_embeddings = np.load(file_path + '/' + file)
        for embedding in faces_embeddings:
            if face_compare(embedding, target_embedding, threshold):
                # TODO：根据具体情况更新文件名
                video_name_ele = file.split('_')
                video_name = video_name_ele[0] + "_" + video_name_ele[1]
                if video_name not in video_list:
                    video_list.append(video_name)
    return video_list



def get_video(featurefile_path, target_embedding, threshold):
    """
    检索视频
    file_path:存储视频对应帧特征的视频
    target_embedding:提取的上传的人脸照片的特征
    threshold:认定为同一人的阈值
    """
    video_list = []

    files = os.listdir(featurefile_path)
    for file in files:
        print(file)
        faces_embeddings = np.load(featurefile_path + file)
        for embedding in faces_embeddings:
            # FIXME: 存特征的时候是否进行了正则化
            # embedding = np.array(embedding).reshape((1, -1))
            # embedding = preprocessing.normalize(embedding)
            if face_compare(embedding, target_embedding, threshold):
                # TODO：根据具体情况更新文件名
                video_name_ele = file.split('_')
                video_name = video_name_ele[0] + "_" + video_name_ele[1] # + ".mp4"
                # FIXME：暂时采取去重
                if video_name not in video_list:
                    video_list.append(video_name)
                # TODO: 一旦检测到一个人脸即可选择该视频，break跳出循环
                # break

    return video_list

# if __name__ == "__main__":
#     face_extractor = FaceExtractor()
#     threshold = 1.8
#     # 保证最终获得的目标图像只有一张清晰人脸
#     target_image = None
#     file_path = "存放特征的文件路径"

#     target_embedding = face_extractor.extract(target_image)[0].embedding
#     target_embedding = np.array(target_embedding).reshape((1, -1))
#     target_embedding = preprocessing.normalize(target_embedding)

#     video_list = get_all_video(file_path, target_embedding, threshold)
