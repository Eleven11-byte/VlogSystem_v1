- 自动录制+生成旅游视频系统

代码说明：
- app.py：实际需要运行的后端代码文件
- record.py：VisitorDetector类，用于实现检测录制开始条件
- face_recognize：人脸特征提取，人脸识别对比是否是同一人
- tracker.py：视频中的目标跟踪
- AIDetector_pytorch：视频中的目标检测
- video：所有前端文件
- demo：用于测试后端功能的文件



- database：存储录制的视频（暂定命名方式：摄像头序号/位置 + 年-月-日-小时-分-秒）
- frames：提取的帧
- feature：存储每个视频对应的几帧画面中提取的人脸特征
- faces：存储上传永远检索的目标人物图片



```python
FACE_FOLDER='faces/'#录入的人脸图像
FACE_FEATURE_FOLDER = 'facefeature/'  # 上传的人脸特征
BACKGROUNDMUSIC_FOLDER='audio/'#背景音乐
PREPARED_FOLDER='prepareds/'#景区视频
OUTPUT_FOLDER='outputs/'#输出视频文件夹
FRAMES_FOLDER = 'frames/'#帧图片，从录制视频中存储多帧图片存储
UPLOAD_FOLDER = 'uploads/'#封装后人像视频
TEMP_FOLDER = 'temp/'#封装前人像视频
THREEFRAMES_FOLDER = 'threeframes/'#从视频中截取三帧图片存储
FEATURES_FOLDER = 'featuresfromvideo/'  # 存储三帧对应提取的人脸特征
```

