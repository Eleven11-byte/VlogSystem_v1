import base64
import time
from datetime import datetime, timedelta
from threading import Lock, Thread
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import moviepy.config as mpy_config
import subprocess
import numpy as np
from record import DetectVisitor
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip,CompositeAudioClip
# from face_recognize import FaceExtractor
from face_extractor_singleton import face_extractor
import face_recognize


# 一直GET isRecording（GET和POST）
# TODO：替换ffmpeg路径
# mpy_config.change_settings({"FFMPEG_BINARY": "D:/Document/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"})  # 替换为你的 ffmpeg 路径
mpy_config.change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})

# 创建一个 Flask 应用实例
app = Flask(__name__)
# 并允许来自所有域的请求
CORS(app)

is_recording = False  # 录制状态


countPerDay = 0  # 每日（每次启动）的录入人脸的人数



# NOTE: 用于检测录制条件相关的全局变量
condition = [20, 800, 200, 100]
visitor_detector = DetectVisitor(condition)
# face_extractor = FaceExtractor()
threshold = 1.0  # 人脸识别的阈值
# TODO:需要加入的限制，在未上传人脸图像时不能进行合成旅行vlog
# target_embeddings = None  # 每次上传新的人脸图片时提取特征并进行更新
# output_video = 'ERROR_NoVideo.mp4'


# NOTE: 各种存储路径


"""
根据景区名区分存储位置
"""
FRAMES_FOLDER = 'frames/'#帧图片
UPLOAD_FOLDER = 'uploads/'#封装后人像视频
TEMP_FOLDER = 'temp/'#封装前人像视频
THREEFRAMES_FOLDER = 'threeframes/'#从视频中截取三帧图片存储
FEATURES_FOLDER = 'featuresfromvideo/'  # 存储三帧对应提取的人脸特征
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(THREEFRAMES_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)


view_position_list = ["view1", "view2", "view3"]
# 根据景区初始化文件夹

for view_position in view_position_list:
    os.makedirs(FRAMES_FOLDER + view_position + "/", exist_ok=True)
    os.makedirs(UPLOAD_FOLDER + view_position + "/", exist_ok=True)
    os.makedirs(TEMP_FOLDER + view_position + "/", exist_ok=True)
    os.makedirs(THREEFRAMES_FOLDER + view_position + "/", exist_ok=True)
    os.makedirs(FEATURES_FOLDER + view_position + "/", exist_ok=True)


FACE_FOLDER='faces/'#录入的人脸图像
FACE_FEATURE_FOLDER = 'facefeature/'  # 上传的人脸特征
BACKGROUNDMUSIC_FOLDER='audio/'#背景音乐
PREPARED_FOLDER='prepareds/'#景区视频
OUTPUT_FOLDER='outputs/'#输出视频文件夹
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(BACKGROUNDMUSIC_FOLDER, exist_ok=True)
os.makedirs(PREPARED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FACE_FEATURE_FOLDER, exist_ok=True)

# 封装视频
def reencode_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',  # 自动覆盖文件
        '-i', input_path,  # 输入文件
        '-c:v', 'libx264',  # 使用 H.264 编解码器
        '-c:a', 'aac',  # 使用 AAC 音频编解码器
        output_path  # 输出文件
    ]

    # 执行命令
    subprocess.run(command)

# 从视频中提取帧
def extract_frames(video_path, timestamps, view_position):
    """
    从视频中存3帧
    """
    print("extract_frames")
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取文件名
    video_name = os.path.basename(video_path).split(".")[0]

    for t in timestamps:
        # 设置视频捕捉的位置（以毫秒为单位）
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

        # 读取帧
        ret, frame = cap.read()
        if ret:
            # 保存帧为图片
            folder = "/" + view_position + THREEFRAMES_FOLDER
            threeFrames_path = os.path.join(THREEFRAMES_FOLDER + view_position + "/", f'{video_name}_frame_{t}s.jpg')
            cv2.imwrite(threeFrames_path, frame)
            print(f'Saved frame at {t}s to {threeFrames_path}')
            # 存特征
            feature_path = FEATURES_FOLDER + view_position + "/" + video_name.split('.')[0] + "_frame_" + str(t) + "s" + ".npy"
            save_features(frame, face_extractor, feature_path)
            print(f'Saved framefeature at {t}s {feature_path}')

        else:
            print(f'Error: Could not read frame at {t}s.')

    # 释放视频捕捉对象

    cap.release()
    return


def save_features(image, face_extractor, save_path):
    """
    提取并存储人脸图像的特征，并返回特征
    """
    embeddings = face_extractor.extract_features(image)
    np.save(save_path, embeddings)
    return embeddings


def find_video(target_image, file_path, threshold):
    """
    检索视频，返回检索出的视频list
    """
    global target_embeddings

    # FIXME:加入限制，保证最终获得的目标图像只有一张清晰人脸，现在暂时以有多张人脸的特征进行逐一检索
    for i in range(target_embeddings.shape[0]):
        video_list = face_recognize.get_video(file_path, target_embeddings[i], threshold)
        break
    return video_list


@app.route('/isRecording', methods=['GET'])
def getIsRecording():
    # FIXME：之后需要控制多个摄像头，需要获取多个摄像头的录制状态
    global is_recording
    return jsonify({'isRecording': is_recording})


@app.route('/uploadFrames', methods=['POST'])
def uploadFrames():
    """
    未触发录制时，传帧并检测是否开始录制
    """
    # FIXME：多个摄像头后需要改变global变量is_recording，如何传递该状态
    global is_recording

    if not is_recording:
        data = request.json
        image_data = data['image']
        view_position = data['viewPosition']  # 景点位置

        # 从 base64 数据中提取实际的图像数据
        image_data = image_data.split(",")[1]
        image_data = base64.b64decode(image_data)
        # 不进行存储，实时识别是否开始录制
        nparr = np.frombuffer(image_data, np.uint8)  # 转换为numpy数组
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 将numpy数组转换为opencv格式的图像
        # 检测是否触发录制条件
        is_recording = visitor_detector.detect(img)

        if is_recording:
            print("start sleep")
            time.sleep(15)
            is_recording = False
            print("录制结束")

    # 保存图像
    # filename = f"frame_{int(time.time())}.jpg"
    # with open(os.path.join(FRAMES_FOLDER,filename), 'wb') as f:
    #     f.write(image_data)

    return 'upload frames', 200


# 依据关键字查找列表中的对应的索引
def find_video_indices(userVideoList,keyword):
    # 使用 enumerate() 获取索引和视频名
    indices = [index for index, video in enumerate(userVideoList) if keyword in video]
    return indices[0]


# 上传并保存录制视频
@app.route('/uploadVideo', methods=['POST'])
def uploadVideo():
    if 'video' not in request.files:
        return 'No video part', 400

    video = request.files['video']  # 视频
    # view = request.form.get('view')  # 景点代号
    view_position = request.form.get('viewPosition')  # 景点代号

    temp_video_path = os.path.join(TEMP_FOLDER + view_position + "/", f"temp_{int(time.time())}_{view_position}.mp4")
    video_path = os.path.join(UPLOAD_FOLDER + view_position + "/", f"{int(time.time())}_{view_position}.mp4")

    # temp_video_path = os.path.join(TEMP_FOLDER, f"temp_{int(time.time())}_{view}.mp4")  # time.time()时间戳，单位秒，自1970.1.1的0：00经过的秒数

    video.save(temp_video_path)

    # video_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{view}.mp4")
    # video_path=os.path.join(UPLOAD_FOLDER, 'people_video.mp4')
    try:
        reencode_video(temp_video_path, video_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    timestamps = [5, 7.5, 10]
    extract_frames(video_path, timestamps, view_position)

    return 'Video uploaded successfully', 200

@app.route('/uploadFacePic', methods=['POST'])
def uploadFacePic():
    if 'facePic' not in request.files:
        return jsonify('失败-没有人脸图像被上传'), 400

    # Note: 上传时通过userID区分
    user_id = request.form.get("userId")
    print("userid:" + str(user_id))

    if not user_id:
        return jsonify('失败-缺少 userId 参数'), 400

    facePic = request.files['facePic']
    if facePic.filename == '':
        return jsonify('失败-没有选择人脸图像'), 400

    # 检查文件扩展名
    if not facePic.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify('失败-只支持 JPG,JPEG,PNG 格式'), 400

    # 保存原图（可选）
    facePic_path = os.path.join(FACE_FOLDER, f"{user_id}.jpg")
    facePic.save(facePic_path)

    # 提取并保存人脸特征
    img = cv2.imread(facePic_path)
    faceFeature_path = os.path.join(FACE_FEATURE_FOLDER, f"{user_id}.npy")
    embeddings = save_features(img, face_extractor, faceFeature_path)

    return jsonify({'message': '成功-人脸图像上传成功', 'userId': user_id})


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
        print(file)
        faces_embeddings = np.load(file_path + file)
        for embedding in faces_embeddings:
            if face_recognize.face_compare(embedding, target_embedding, threshold):
                # TODO：根据具体情况更新文件名
                video_name_ele = file.split('_')
                video_name = video_name_ele[0] + "_" + video_name_ele[1]
                if video_name not in video_list:
                    video_list.append(video_name)
    return video_list


def getVideoList(view_position_list, target_embedding, threshold):
    video_list_all = []
    video_list_result = []

    for view_position in view_position_list:
        feature_path = FEATURES_FOLDER + view_position + "/"
        video_list = face_recognize.find_similar(feature_path, target_embedding, threshold)
        video_list_result.append(video_list[-1])
        video_list_all.append(video_list)

    return video_list_result, video_list_all


#上传
@app.route('/getVideo', methods=['POST'])
def getVideo():
    # global target_embeddings  # 目标人物的特征
    global threshold
    global view_position_list
    # global output_video

    # 获取背景音乐
    music_type = request.form.get('musicType')

    print(music_type)

    user_id = request.form.get("userId")

    print("getvideo userid" + str(user_id))

    faceFeature_path = os.path.join(FACE_FEATURE_FOLDER, f"{user_id}.npy")
    if not os.path.exists(faceFeature_path):
        return {"error": f"未找到 {user_id} 的人脸特征，请先上传"}

    target_embeddings = np.load(faceFeature_path, allow_pickle=True)

    # TODO: 将videoOfUser改为检索得到
    #获取对应id的视频列表
    # NOTE:此处由于需要对拍摄位置检索，所以暂时手动改名view1.2.3
    # videoOfUser=['1727766554_view1','1727335235_view2','1727335222_view3']
    # Note: 在此打印视频列表用于debug
    # FIXME: 默认target_embeddings中只存储了一张人脸特征
    # for i in range(target_embeddings.shape[0]):
    #     videoOfUser, _ = face_recognize.get_video(FEATURES_FOLDER, target_embeddings[i], threshold)
    videoOfUser, _ = getVideoList(view_position_list, target_embeddings[0], threshold)
    print("videolist")
    print(videoOfUser)

    #人像视频以及风景视频列表
    clips = []

    # TODO：可改成循环，以便之后灵活增减景点数
    #在给出的id视频列表中查找对应顺序的人像视频，并加入clips列表
    #景点1（view1）处的人像视频
    index_view1=find_video_indices(videoOfUser, 'view1')
    userVideoPath_view1=os.path.join(UPLOAD_FOLDER + "view1" + "/", f'{videoOfUser[int(index_view1)]}.mp4')
    userVideo_view1=VideoFileClip(userVideoPath_view1).set_fps(30).resize((3840, 2160))
    clips.append(userVideo_view1)

    # 景点2（view2）处的人像视频
    index_view2 = find_video_indices(videoOfUser, 'view2')
    userVideoPath_view2 = os.path.join(UPLOAD_FOLDER + "view2" + "/", f'{videoOfUser[int(index_view2)]}.mp4')
    userVideo_view2 = VideoFileClip(userVideoPath_view2).set_fps(30).resize((3840, 2160))
    clips.append(userVideo_view2)

    # 景点3（view3）处的人像视频
    index_view3 = find_video_indices(videoOfUser, 'view3')
    userVideoPath_view3 = os.path.join(UPLOAD_FOLDER + "view3" + "/", f'{videoOfUser[int(index_view3)]}.mp4')
    userVideo_view3 = VideoFileClip(userVideoPath_view3).set_fps(30).resize((3840, 2160))
    clips.append(userVideo_view3)

    #将景区风景视频插入clips列表
    # prepared_clip1_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video1.mp4')
    prepared_clip1_path = os.path.join(PREPARED_FOLDER, '20240424_C1720.MP4')
    prepared_clip1 = VideoFileClip(prepared_clip1_path).set_fps(30).resize((3840, 2160))
    clips.insert(0, prepared_clip1)  # 预设视频1放在合成视频列表的第一个

    # prepared_clip2_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video2.mp4')
    prepared_clip2_path = os.path.join(PREPARED_FOLDER, '20240424_C1640.MP4')
    prepared_clip2 = VideoFileClip(prepared_clip2_path).set_fps(30).resize((3840, 2160))
    clips.insert(2, prepared_clip2)  # 预设视频1放在合成视频列表的第三个

    # prepared_clip3_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video3.mp4')
    prepared_clip3_path = os.path.join(PREPARED_FOLDER, '20240424_C1672.MP4')
    prepared_clip3 = VideoFileClip(prepared_clip3_path).set_fps(30).resize((3840, 2160))
    clips.insert(4, prepared_clip3)  # 预设视频1放在合成视频列表的第五个

    # prepared_clip4_path = os.path.join(PREPARED_FOLDER, 'scenicSpot_video4.mp4')
    prepared_clip4_path = os.path.join(PREPARED_FOLDER, '20240424_C1694.MP4')
    prepared_clip4 = VideoFileClip(prepared_clip4_path).set_fps(30).resize((3840, 2160))
    clips.insert(6, prepared_clip4)  # 预设视频1放在合成视频列表的第七个

    # clips列表完成
    # 检查每个剪辑的 fps
    for clip in clips:
        print("Clip FPS:", clip.fps)

    # 合并所有裁剪过的视频
    final_clip = concatenate_videoclips(clips, method='compose')

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

    final_clip = final_clip.set_fps(30)  # 强制设置为 30 fps

    # output_video = f"{int(time.time())}_output.mp4"
    output_video = f"{user_id}_output.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_video)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=30)

    # 释放资源
    for clip in clips:
        clip.close()
    background_music.close()
    final_clip.close()

    return jsonify({'message': 'Videos processed successfully', 'output': f'/outputs/{output_video}'})

# 预览
@app.route('/outputs/<path:filename>', methods=['GET'])
def lookVideo(filename):
    response = send_from_directory(OUTPUT_FOLDER, filename)
    response.headers['Content-Type'] = 'video/mp4'  # 设置 MIME 类型
    return response


# 下载
@app.route('/downloadVideo', methods=['GET'])
def downloadVideo():
    # 关键：GET 请求通过 request.args 获取 URL 中的 query 参数
    user_id = request.args.get("userId")  # 替换 request.form.get

    if not user_id:
        return jsonify({"error": "缺少 userId 参数"}), 400

    # 拼接视频文件名（需与视频生成时的命名规则一致）
    # 生成视频时的文件名是 f"{int(time.time())}_output_{user_id}.mp4"
    # 这里需要根据实际存储的文件名规则查询，以下是示例：
    output_video = f"{user_id}_output.mp4"  # 需与生成视频时的命名匹配
    video_path = os.path.join(OUTPUT_FOLDER, output_video)

    # 检查文件是否存在
    if not os.path.exists(video_path):
        return jsonify({"error": f"未找到用户 {user_id} 的视频文件"}), 404

    # 返回文件并触发下载
    return send_file(video_path, as_attachment=True)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
