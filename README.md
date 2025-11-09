# 自动录制+生成旅游视频系统



## 主要功能

1. 用户人脸图像上传与识别
2. 摄像头实时帧捕获与上传
3. 视频录制与上传
4. Vlog 生成与下载

最终部署到服务器上，通过前端网页的形式进行交互，前端网页提供上传网页、视频预览、下载视频的功能

### 功能对应进度和问题

#### 1. 用户人脸图像上传与识别

- ✅ 已实现，前端上传图片，后端提取人脸特征并匹配视频

#### 2. 条件触发摄像头录制

- 录制画面尺寸1920x1080，触发范围设置：
-  定义为画面中间 70% 的核心区域（排除边缘干扰）：
  - x 轴：左边界 > 画面宽度的 15%，右边界 < 画面宽度的 85%。
  - y 轴：上边界 < 画面高度的 95%（过滤画面顶部无效区域，保留底部主要区域）。
  - 为了保证能提取到人脸特征，从每段视频中截取8帧存储并存储人脸特征

#### 3. 摄像头管理，视频录制与上传

- 单个摄像机目前通过网线直接连接电脑，调整IP设置，通过RTSP连接可以实现连接摄像头、获取画面和录制视频
- 跨设备不在一个网段无法直接传输，实际使用多个时可能需要连录像机设备，暂时无法测试和验证

#### 4. Vlog 生成与下载

- ✅ 支持上传人脸图像，选择音乐合成，合成后预览和下载
- 异步任务管理celery + redis并发实现多个视频生成未实现
- 下载前加入付费接口还未添加


## 功能模块设计和对应实现代码

### 前端功能模块

- **录制与上传视频**

- **人脸管理**
  - 用户人脸选择、预览、上传
  - 每个用户在上传图片时创建userid，根据userid进行后续生成视频的标识
- **任务管理**
  - 异步任务状态轮询
  - 视频生成进度显示
- **视频合成与预览**
  - 用户选择背景音乐类型
- **下载功能**
  - 生成视频后付费下载

### 后端功能模块

- **摄像头管理**（service/camera_service_modified.py）
  - 目标检测判断触发开启/关闭摄像头
  - 帧抓取与实时上传
- **人脸管理**（service/face_service.py）
  - 存储人脸图像和人脸图像特征
- **视频合成**（service/video_service.py）
  - 根据用户选择音乐和上传图像检索的视频结果合成视频
- **API 接口**
  - `/uploadFacePic`：人脸图像上传
  - `/getVideo`：生成 Vlog 视频
  - ~~`/taskStatus`：任务状态查询~~
  - ~~`/isRecording`：录制状态检查~~
- **GPU** **加速处理**
  - 人脸识别（face_extractor.py face_extractor_singleton.py）
  - 目标检测（AIDetector_pytorch.py）
- **异步任务管理**
  - ==Celery + Redis 任务队列==
  - ==任务状态管理与轮询==
- **数据存储**
  - 本地文件存储视频与帧

### 其他代码说明

#### config.py

> 用于初始化一些配置，主要包括

- ffmpeg路径
- 人脸识别阈值
- 录制条件参数
- 景点列表
- 路径配置
- 存储目录

#### app_modified.py

> 实现了摄像头线程管理的主程序

### 项目结构
```
├── AIDetector_pytorch.py  #实现行人检测
├── DATA #所有数据存储
│   ├── audio
│   ├── facefeature #用户上传人脸提取的特征
│   ├── faces #用户上传人脸特征
│   ├── featuresfromvideo #自动录制过程中提取的特征
│   ├── frames #自动录制时存的视频帧
│   ├── outputs #合成输出的视频
│   ├── prepareds #预先准备好的素材视频
│   └── records #自动录制的视频
├── README.md 
├── ~~app.py~~  #未拆分前的代码，已弃用
├── app_modified.py #目前的主程序代码
├── celery_app.py #试图实现并发但是没有实现的代码
├── config.py #实现配置管理
├── deep_sort # 实现目标检测
│   ├── configs
│   ├── deep_sort
│   └── utils
├── demo.py
├── environment.yml 
├── face_extractor_singleton.py
├── face_recognize.py #实现人脸检测功能
├── models #模型文件
│   ├── `__init__`.py (0B)
│   ├── common.py (16.9KB)
│   ├── experimental.py (5.3KB)
│   ├── export.py (7.1KB)
│   ├── hub
│   ├── yolo.py (13.6KB)
│   ├── yolov5l.yaml (1.4KB)
│   ├── yolov5m.yaml (1.4KB)
│   ├── yolov5s.yaml (1.4KB)
│   └── yolov5x.yaml (1.4KB)
├── outputs
│   ├── user_pcxul2j6u_output.mp4 (60.0MB)
│   ├── user_snu7xx196_output.mp4 (124.1MB)
│   └── user_uu030eelg_output.mp4 (124.1MB)
├── prepareds
│   ├── 20240424_C1640.MP4 (64.6MB)
│   ├── 20240424_C1672.MP4 (64.6MB)
│   ├── 20240424_C1694.MP4 (64.6MB)
│   ├── 20240424_C1720.MP4 (64.6MB)
│   ├── scenicSpot_video1.mp4 (90.6MB)
│   ├── scenicSpot_video2.mp4 (67.8MB)
│   ├── scenicSpot_video3.mp4 (62.3MB)
│   └── scenicSpot_video4.mp4 (58.1MB)
├── requirements.txt 
├── services
│   ├── camera_service.py 
│   ├── camera_service_modified.py 
│   ├── camera_singleton.py 
│   ├── face_service.py 
│   └── video_service.py 
├── tracker.py # 实现目标跟踪
├── utils
├── utils_app.py #主程序需要的工具函数 
├── weights
│   └── yolov5s.pt 
└── 开发文档.md 
```
### 统一文件命名方式

- 用户上传人脸图片：
  - 图片：/faces/{user_id}.jpg
  - 特征：/facefeature/{user_id}.npy
  - 注意：user_id在前端生成，每次上传图片会生成新的id`'user_' + Math.random().toString(36).substr(2, 9);`
  - 与最终生成视频命名匹配
- 生成视频命名
  - outputs/{user_id}_output.mp4
- 录制过程中的文件
  - 实时上传的帧和特征（目前存帧出于便于debug的目的，实际只要存特征就可以）
  - frames/view1
  - featuresfromvideo/view1（检索对应视频时会用到）
  - records/view1

## 部署方案

> 一种可能的部署方案

### 前端部署

- 云服务器Nginx 静态资源
- HTTPS 配置 + 域名绑定

### 后端部署

- 本地 GPU 服务器
- Flask/Gunicorn 提供 API
- VPN 或反向代理实现云端访问

### 数据流说明

1. 用户在浏览器操作 → 前端上传视频/API 请求
2. 云端前端通过 VPN/反向代理访问本地 GPU 后端
3. 后端生成任务 → Celery 异步处理 GPU 密集任务
4. 前端轮询任务状态 → 获取生成的视频下载链接