"""
用于检测后端demo功能
"""

import face_recognize
import numpy as np
from sklearn import preprocessing
from record import DetectVisitor
import cv2
from moviepy.editor import *
from face_recognize import FaceExtractor
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip,CompositeAudioClip

# NOTE: 用于检测录制条件相关的全局变量
condition = [20, 800, 200, 100]
visitor_detector = DetectVisitor(condition)
face_extractor = FaceExtractor()
threshold = 1.0  # 人脸识别的阈值
target_embeddings = None  # 每次上传新的人脸图片时提取特征并进行更新


# NOTE: 各种存储路径
FRAMES_FOLDER='frames/'  # 帧图片
UPLOAD_FOLDER='uploads/'  # 封装后人像视频
TEMP_FOLDER='temp/'  # 封装前人像视频
FACE_FOLDER='faces/'  # 录入的人脸图像
THREEFRAMES_FOLDER='threeframes/'  # 从视频中截取三帧图片存储
BACKGROUNDMUSIC_FOLDER='audio/'  # 背景音乐
PREPARED_FOLDER='prepareds/'  # 景区视频
OUTPUT_FOLDER='outputs/'  # 输出视频文件夹
FACE_FEATURE_FOLDER = 'facefeature/'  # 上传的人脸特征

# TODO: 更新输出视频的名字
OUTPUT_VIDEO = 'outputs/output_video.mp4'

# FEATURES_FOLDER = 'features/'
FEATURES_FOLDER = 'featuresfromvideo/'


os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(THREEFRAMES_FOLDER, exist_ok=True)
os.makedirs(BACKGROUNDMUSIC_FOLDER, exist_ok=True)
os.makedirs(PREPARED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FACE_FEATURE_FOLDER, exist_ok=True)



def cut(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    duration = 15
    frame_id = 0
    appear_frame = 0
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 目标识别剪辑视频
    condition = [200, 0, 200, 100]  # condition = [左边缘，上边缘，宽，高]
    visitorDetector = DetectVisitor(condition)

    while True:
        _, im = cap.read()
        frame_id += 1
        if im is None:
            break
        detected = visitorDetector.detect(im)
        print(detected)

        if detected:
            appear_frame = frame_id
            break

    cap.release()

    start_time = appear_frame / fps
    end_time = start_time + duration

    clip = VideoFileClip(video_path)
    cut_clip = clip.subclip(start_time, end_time)
    cut_clip.write_videofile('example_video_result/test1_cut_video.mp4')


# def save_features(image_path, image_id, save_path):
#     image = cv2.imread(image_path)
#     face_extractor = FaceExtractor()
#     face_extractor.save_embedding(image, image_id, save_path)

def save_features(image, face_extractor, save_path):
    """
    提取并存储人脸图像的特征，并返回特征
    """
    embeddings = face_extractor.extract_features(image)
    np.save(save_path, embeddings)
    return embeddings


def find_video(target_image, file_path, threshold):
    face_extractor = face_recognize.FaceExtractor()
    # threshold = 1.8
    #
    # # TODO:保证最终获得的目标图像只有一张清晰人脸

    # file_path = "存放特征的文件路径"
    #
    target_embedding = face_extractor.extract(target_image)[0].embedding
    target_embedding = np.array(target_embedding).reshape((1, -1))
    target_embedding = preprocessing.normalize(target_embedding)
    #
    video_list = face_recognize.get_video(file_path, target_embedding, threshold)
    return video_list


def find_video_indices(userVideoList,keyword):
    # 使用 enumerate() 获取索引和视频名
    indices = [index for index, video in enumerate(userVideoList) if keyword in video]
    return indices[0]


def edit(videoOfUser):
    clips = []

    index_view1 = find_video_indices(videoOfUser, 'view1')
    userVideoPath_view1 = os.path.join(UPLOAD_FOLDER, f'{videoOfUser[int(index_view1)]}.mp4')
    userVideo_view1 = VideoFileClip(userVideoPath_view1).set_fps(30).resize((640, 480))
    clips.append(userVideo_view1)

    # 景点2（view2）处的人像视频
    index_view2 = find_video_indices(videoOfUser, 'view2')
    userVideoPath_view2 = os.path.join(UPLOAD_FOLDER, f'{videoOfUser[int(index_view2)]}.mp4')
    userVideo_view2 = VideoFileClip(userVideoPath_view2).set_fps(30).resize((640, 480))
    clips.append(userVideo_view2)

    # 景点3（view3）处的人像视频
    index_view3 = find_video_indices(videoOfUser, 'view3')
    userVideoPath_view3 = os.path.join(UPLOAD_FOLDER, f'{videoOfUser[int(index_view3)]}.mp4')
    userVideo_view3 = VideoFileClip(userVideoPath_view3).set_fps(30).resize((640, 480))
    clips.append(userVideo_view3)

    # 将景区风景视频插入clips列表
    prepared_clip1_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video1.mp4')
    prepared_clip1 = VideoFileClip(prepared_clip1_path).set_fps(30).resize((640, 480))
    clips.insert(0, prepared_clip1)  # 预设视频1放在合成视频列表的第一个

    prepared_clip2_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video2.mp4')
    prepared_clip2 = VideoFileClip(prepared_clip2_path).set_fps(30).resize((640, 480))
    clips.insert(2, prepared_clip2)  # 预设视频1放在合成视频列表的第三个

    prepared_clip3_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video3.mp4')
    prepared_clip3 = VideoFileClip(prepared_clip3_path).set_fps(30).resize((640, 480))
    clips.insert(4, prepared_clip3)  # 预设视频1放在合成视频列表的第五个

    prepared_clip4_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video4.mp4')
    prepared_clip4 = VideoFileClip(prepared_clip4_path).set_fps(30).resize((640, 480))
    clips.insert(6, prepared_clip4)  # 预设视频1放在合成视频列表的第七个
    # clips列表完成
    # 检查每个剪辑的 fps
    for clip in clips:
        print("Clip FPS:", clip.fps)

    # 合并所有裁剪过的视频
    final_clip = concatenate_videoclips(clips, method='compose')

    music_type = "bgm01_menghuan"
    # 添加背景音乐
    music_path = os.path.join(BACKGROUNDMUSIC_FOLDER, f"{music_type}.mp3")
    background_music = AudioFileClip(music_path)

    # 获取原视频的音频
    original_audio = final_clip.audio

    # 音频处理
    if original_audio is not None:
        final_audio = CompositeAudioClip([original_audio.volumex(0.5), background_music.volumex(0.8)])
        final_clip = final_clip.set_audio(final_audio)
    else:
        print("Original audio is None, setting only background music.")
        final_clip = final_clip.set_audio(background_music)

    # 输出调试信息
    print("Final clip size:", final_clip.size)
    print("Final clip duration:", final_clip.duration)
    print("Final clip FPS:", final_clip.fps)

    # NOTE: 存储处理好的视频
    final_clip = final_clip.set_fps(30)  # 强制设置为 30 fps
    final_clip.write_videofile(OUTPUT_VIDEO, codec='libx264', audio_codec='aac', fps=30, threads=4)
    # final_clip.write_videofile(OUTPUT_VIDEO, codec='mpeg4', audio_codec='aac', fps=30)

    # 释放资源
    for clip in clips:
        clip.close()
    background_music.close()
    final_clip.close()

    final_clip = concatenate_videoclips(clips, method='compose')

    return final_clip

def extract_frames(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取文件名
    video_name = os.path.basename(video_path).split(".")[0] + "_view1"

    for t in timestamps:
        # 设置视频捕捉的位置（以毫秒为单位）
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

        # 读取帧
        ret, frame = cap.read()
        if ret:
            # FIXME:暂时保存帧为图片用于debug，之后可以不用存
            threeFrames_path = os.path.join(THREEFRAMES_FOLDER, f'{video_name}_frame_{t}s.jpg')
            cv2.imwrite(threeFrames_path, frame)
            print(f'Saved frame at {t}s to {threeFrames_path}')
            # TODO:存帧图片
            feature_path = FEATURES_FOLDER + video_name.split('.')[0] + "_frame_" + str(t) + "s" + ".npy"
            save_features(frame, face_extractor, feature_path)
            print(f'Saved framefeature at {t}s {feature_path}')
        else:
            print(f'Error: Could not read frame at {t}s.')

    # 释放视频捕捉对象

    cap.release()
    return


if __name__ == "__main__":
    # face_extractor = FaceExtractor()
    # image_path = "./example_frames/extract_example/Clip0046 - frame at 0m31s.jpg"
    # image_id = "Clip0046_1"
    # save_path = "./features/test.npy"
    #
    # #
    # image = cv2.imread(image_path)
    # save_features(image, face_extractor, save_path)

    '''
    path = "./example_frames/extract_example"
    save_path = "./features"
    files = os.listdir(path)
    # for file in files:
    #     image_id = file.split(".")[0]
    #     save_features(path + "/" + file, image_id, save_path)
    file = files[2]
    image_id = file.split(".")[0]
    save_features(path + "/" + file, image_id, save_path)
    '''


    # file_path = "./features/"
    # threshold = 1.8
    #
    # for i in range(target_embeddings.shape[0]):
    #     print(i)
    #     target_embedding = target_embeddings[i]
    #     target_embedding = np.array(target_embedding).reshape((1, -1))
    #     target_embedding = preprocessing.normalize(target_embedding)
    #     video_list = face_recognize.get_video(file_path, target_embedding, threshold)
    #     print(video_list)

    """
    FEATURES_FOLDER = "featuresfromvideo/"
    face_extractor = FaceExtractor()
    threshold = 1.0

    target_image = cv2.imread('./faces/2024_10_12_2.jpg')

    target_embeddings = face_extractor.extract_features(target_image)

    for i in range(target_embeddings.shape[0]):
        videoOfUser = face_recognize.get_video(FEATURES_FOLDER, target_embeddings[i], threshold)
        print(videoOfUser)
    """

    # videoOfUser = ['1728699528_view1', '1728700557_view3', '1728700573_view2']


    # 提取视频并保存特征
    video_path = "./example_video/scene/A040_01211633_C029.mp4"
    timestamps = [5, 7.5, 10]
    extract_frames(video_path, timestamps)


    # 比较人脸相似度

    """
    target_embeddings = np.load("./facefeature/2024_10_13_1.jpg.npy")

    files = os.listdir("./featuresfromvideo/")
    faces_embeddings = np.load("./featuresfromvideo/" + files[-1])

    for i in range(target_embeddings.shape[0]):
        for embedding in faces_embeddings:
            if face_recognize.face_compare(embedding, target_embeddings[i], threshold):
                print(files[0])
    """
