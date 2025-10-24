<template>
    <div>
    <!-- header section start -->
    <!--纵栏-->
    <div class="header_section">
         <div class="header_main">
            <div class="mobile_menu">
               <nav class="navbar navbar-expand-lg navbar-light bg-light">
                  <div class="logo_mobile"><router-link to="/"><img src="../images/logo.png"></router-link></div>
                  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                  <span class="navbar-toggler-icon"></span>
                  </button>
                  <div class="collapse navbar-collapse" id="navbarNav">
                     <ul class="navbar-nav">
                        <li class="nav-item">
                           <router-link to="/" class="nav-link">首页</router-link>
                        </li>
                        <li class="nav-item">
                           <router-link to="/about" class="nav-link">关于景区</router-link>
                        </li>
                        <li class="nav-item">
                           <router-link to="/services" class="nav-link">实时录制</router-link>
                        </li>
                        <li class="nav-item">
                           <router-link to="/blog" class="nav-link">旅行vlog</router-link>
                        </li>
                        <li class="nav-item">
                           <router-link to="/contact" class="nav-link">联系我们</router-link>
                        </li>
                     </ul>
                  </div>
               </nav>
            </div>
             <!--横栏-->
            <div class="container-fluid">
               <div class="logo"><router-link to="/"><img src="../images/logo.png"></router-link></div>
               <div class="menu_main">
                  <ul>
                     <li class="active"><router-link to="/" class="nav-link">首页</router-link></li>
                     <li><router-link to="/about" class="nav-link">关于景区</router-link></li>
                     <li><router-link to="/services" class="nav-link">实时录制</router-link></li>
                     <li><router-link to="/blog" class="nav-link">旅行vlog</router-link></li>
                     <li><router-link to="/contact" class="nav-link">联系我们</router-link></li>
                  </ul>
               </div>
            </div>
         </div>
         <!-- banner section start -->
         <div class="banner_section layout_padding">
            <div id="carouselExampleSlidesOnly" class="carousel slide" data-ride="carousel">
               <div class="carousel-inner">
                  <div class="carousel-item active">
                     <div class="container">
                        <h1 class="banner_taital">Travel with U</h1>
                        <p class="banner_text">Get Your Own Travel Vlog.</p>
                     </div>
                  </div>
                  <div class="carousel-item">
                     <div class="container">
                        <h1 class="banner_taital">沿途邮你</h1>
                        <p class="banner_text">获取专属于你的旅行VLOG。</p>
                     </div>
                  </div>
               </div>
            </div>
         </div>
         <!-- banner section end -->
      </div>
      <!-- services section end -->
      <!-- about section start -->
      <div class="about_section layout_padding">
         <div class="container-fluid">
            <div class="row">
               <div class="col-md-6">
                  <div class="about_taital_main">
 
                      <h1 class="about_taital">上传人脸图像</h1>
                      <p class="about_text">上传人脸图像，您将得到专属于您的景区旅行视频。</p>
                      
                      <!--字体-->
                      <p class="about_text">
 
                      <!-- 人脸图片上传部分 -->
  
  
                      <div>
                        <p><label for="file-upload" class="upload-label">选择人脸图像
                            <input type="file" id="file-upload" @change="selectFacePic" accept="image/jpeg, image/png" />
                        </label>

                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        
                        <el-button type="primary" @click="uploadFacePic" :disabled="facePic==null" round size="large" color="#2b2278" style="width: 100px;">
                        上传
                        </el-button></p>
                        <div v-if="imageUrl">
                            <p>预览人脸图像：</p>
                            <img :src="imageUrl" alt="预览" width="300" />
                        </div>
                        <div v-if="facePicInfo">
                            <p>{{ facePicInfo }}</p>
                        </div>
                      </div>

                      <p></p> 
    
 
                      <p>
                         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                      </p>

                      <!--选择背景音乐类型-->
                     <div>
                     <p>
                        <p>请选择背景音乐：</p>
                        <el-select v-model="selectedMusicType" placeholder="选择背景音乐" style="width: 200px;" class="musicSelect">
                           <el-option label="古风" value="bgm01_gufeng"></el-option>
                           <el-option label="气势" value="bgm02_qishi"></el-option>
                           <el-option label="送别" value="bgm03_songbie"></el-option>
                           <el-option label="激昂" value="bgm04_jiang"></el-option>
                           <el-option label="哀伤" value="bgm05_aishang"></el-option>
                           <el-option label="希望" value="bgm06_xiwang"></el-option>
                        </el-select>
                     </p>
                     </div>


                      <!-- 获取Vlog按钮 -->
                     <p><el-button type="primary" :loading="loading" :disabled="loading" @click="GetVideos" round size="large" color="#2b2278" style="width: 300px;">
                        {{ loading ? '旅行VLOG加载中...' : '获取旅行VLOG' }}
                     </el-button></p>

                     <div v-if="taskStatus" class="status-container">
                     <p>任务状态: {{ taskStatusText }}</p>
                     <el-progress :percentage="progress" stroke-width="6" v-if="progress > 0"></el-progress>
                     <p v-if="taskStatus === 'failed'" class="error-message">
                     任务失败，请重试
                     </p>
                     </div>
 
                    </p>
 
                  </div>
               </div>
 
                
               <div class="col-md-6 padding_right_0">
                   <div>
                      <!--图片-->
                     <div v-if="!videoUrl">
                        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                        <img src="../images/about-img.png" class="about_img">
                     </div>
                      
 
                      <!-- vlog预览 -->
                     <div v-if="videoUrl" class="about_taital_main">
                        <h1 class="about_taital">旅行VLOG预览</h1>
                        <p class="about_text">上传或录制人像视频，您将得到专属于您的景区旅行视频。</p>

                        <!--下载vlog-->
                        <p class="about_text">
                        <p><el-button type="success" @click="downloadVideo" round size="large" color="#2b2278" style="width: 300px;">下载视频</el-button></p>
                        <video controls :src="videoUrl" width="600"></video>
                        </p>                          
                     </div>
                                                                  
                   </div>
               </div>
            </div>
         </div>
      </div>
      <!-- header section end -->
      
      <!-- blog section start -->
      <div class="blog_section layout_padding">
         <div class="container">
            <h1 class="blog_taital">旅行VLOG</h1>
            <p class="blog_text">您可以获取为您拍摄的专属于您的旅行vlog</p>
            <div class="play_icon_main">
               <div class="play_icon"><router-link to="/blog"><img src="../images/play-icon.png"></router-link></div>
            </div>
         </div>
      </div>
      <!-- blog section end -->
     
      <!-- footer section start -->
      <div class="footer_section layout_padding">
         <div class="container">
            <div class="input_btn_main">
               <input type="text" class="mail_text" placeholder="输入您的邮箱" name="Enter your email">
               <div class="subscribe_bt"><router-link to="/a">提交</router-link></div>
            </div>
            <div class="location_main">
               <div class="call_text"><img src="../images/call-icon.png"></div>
               <div class="call_text"><router-link to="/a">电话：+86 123 1234 1234</router-link></div>
               <div class="call_text"><img src="../images/mail-icon.png"></div>
               <div class="call_text"><router-link to="/a">邮箱：123 123 123@.com</router-link></div>
            </div>
            <div class="social_icon">
               <ul>
                  <li><a href="#"><img src="../images/fb-icon.png"></a></li>
                  <li><a href="#"><img src="../images/twitter-icon.png"></a></li>
                  <li><a href="#"><img src="../images/linkedin-icon.png"></a></li>
                  <li><a href="#"><img src="../images/instagram-icon.png"></a></li>
               </ul>
            </div>
         </div>
      </div>
      <!-- footer section end -->
      <!-- copyright section start -->
      <div class="copyright_section">
         <div class="container">
            <p class="copyright_text">版权所有 &copy; <a target="_blank" href="https://sdmda.bupt.edu.cn/">北邮数媒院</a></p>
         </div>
      </div>
    </div>
 </template>
 
 <script>
  import { BASE_URL } from '../utils/apiConfig'; 

  export default {
    data() {
      return {
        stream: null,//摄像头的媒体流
        intervalIdFrame: null,//时间间隔
        isCamera:false,//摄像头是否开启

        intervalIdRecord:null,//录制状态检查时间间隔
        mediaRecorder: null,//MediaRecorder对象
        recordedChunks: [],//存储录制视频的片段
        recording: false,//录制状态(前端)

        facePic:null,//人脸图片
        imageUrl:null,//人脸图像URL方便预览
        facePicInfo:null,//人脸上传失败的具体原因

        selectedMusicType: 'bgm01_gufeng',// 音乐：默认选择类型1

        //测试使用——方便切换景点录制视频
        selectedViewPosition:'view1',//景点位置：默认选择景点1

        loading: false, // 加载状态

        videoUrl: '', // 最终合成的视频URL

        taskId: null, // 后端任务ID
        taskStatus: null, // 任务状态
        statusPolling: null, // 轮询状态ID
        progeress: 0, // 进度(0-100)
      };
    },
    
  methods: {

      // 启动状态轮询
      startStatusPolling() {
        // 清除可能存在的旧定时器
        if (this.statusPolling) {
          clearInterval(this.statusPolling);
        }

        // 每3秒查询一次状态
        this.statusPolling = setInterval(async () => {
          await this.checkTaskStatus();
        }, 3000);
      },

      // 查询任务状态
      async checkTaskStatus() {
        if (!this.taskId) return;

        try {
          const response = await fetch(`${BASE_URL}/taskStatus?taskId=${this.taskId}`);
          const status = await response.json();

          this.taskStatus = status.status;
          this.progress = status.progress || 0; // 更新进度

          // 任务完成或失败时停止轮询
          if (status.status === 'completed') {
            this.videoUrl = `${BASE_URL}${status.outputPath}`; // 使用后端返回的视频路径
            this.stopStatusPolling();
            this.loading = false;
          } else if (status.status === 'failed') {
            alert(`任务失败: ${status.message || '未知错误'}`);
            this.stopStatusPolling();
            this.loading = false;
          }
        } catch (error) {
          console.error('Error checking task status:', error);
        }
      },

      // 停止轮询
      stopStatusPolling() {
        if (this.statusPolling) {
          clearInterval(this.statusPolling);
          this.statusPolling = null;
        }
      },

      //开启摄像头
      async startCamera() {
        this.isCamera=true;
        try {
          //获取摄像头和麦克风流
          this.stream = await navigator.mediaDevices.getUserMedia({ video: true,audio: true});
          this.$refs.CameraVideo.srcObject = this.stream;
          // 每秒获取 8 次状态
          this.intervalIdRecord = setInterval(this.checkRecordingStatus, 125);
          // 每秒获取 8 帧
          this.intervalIdFrame = setInterval(this.captureFrame, 125);
        } catch (error) {
          console.error("Error accessing camera: ", error);
        }
    },
      //关闭摄像头
      stopCamera() {
        if (this.stream) {
          //视频流
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
          //抓帧时间间隔
          clearInterval(this.intervalIdFrame);
          clearInterval(this.intervalIdRecord);
          this.intervalIdFrame = null;
          this.intervalIdRecord = null;
          //摄像头开启状态
          this.isCamera=false;
        }
    },


    //抓帧
      captureFrame() {
        const canvas = document.createElement('canvas');
        canvas.width = 640; // 与视频宽度一致
        canvas.height = 480; // 与视频高度一致
  
        const context = canvas.getContext('2d');
        context.drawImage(this.$refs.CameraVideo, 0, 0, canvas.width, canvas.height);
  
        // 将图像数据转换为 base64 URL
        const frameData = canvas.toDataURL('image/jpeg');
  
        // 发送数据到后端
        this.sendFrameToBackend(frameData);
    },
    //发送帧
      async sendFrameToBackend(frameData) {
        try {
          await fetch(`${BASE_URL}/uploadFrames`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
               image: frameData,
               viewPosition: this.selectedViewPosition//传输景点位置
             }),
          });
        } catch (error) {
          console.error("Error sending frame to backend: ", error);
        }
    },


      //检查录制状态
      async checkRecordingStatus() {
        const response = await fetch(`${BASE_URL}/isRecording`);
        const data = await response.json();

        if (data.isRecording && !this.recording) {//录制开始
          this.startRecording();    
          //开始录制时停止发送帧 
          clearInterval(this.intervalIdFrame);    
        } else if (!data.isRecording && this.recording) {//录制结束
          this.stopRecording();
          //结束录制时恢复发送帧
          this.intervalIdFrame = setInterval(this.captureFrame, 125);
        }  
    },
    //开始录制
      startRecording() {
      //初始化
      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: 'video/webm' });
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };
      this.mediaRecorder.onstop=this.uploadVideo;
      this.mediaRecorder.start();
      this.recording = true;
      console.log("开始录制");
    },
    //停止录制
    stopRecording() {
        if (this.mediaRecorder) {
        this.mediaRecorder.stop();
        this.recording = false;
        console.log("停止录制");
      }
    },
    //上传录制视频
    async uploadVideo() {
      const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
      const mp4File = await this.convertWebmToMp4(blob, 'peopleVideo.mp4');
      const formData = new FormData();
      formData.append('video', mp4File);

      //测试使用——方便切换景点录制视频
      formData.append('viewPosition', this.selectedViewPosition); // 将选择的景点位置添加到 FormData 中

      await fetch(`${BASE_URL}/uploadVideo`, {
        method: 'POST',
        body: formData,
      });

      this.recordedChunks = [];
      console.log("视频上传成功");
    },

    
    //选择人脸图像
    selectFacePic(event) {
      this.facePic = event.target.files[0];
      if (this.facePic) {
        this.imageUrl = URL.createObjectURL(this.facePic); // 预览图像
      }
    },

    // 浏览器本地保存 userId，确保同一用户多次使用时 ID 一致
    getUserId() {
      let userId = localStorage.getItem("userId");
      if (!userId) {
        userId = crypto.randomUUID();  // 生成 UUID
        localStorage.setItem("userId", userId);
        console.log("生成新的 userId:", userId);
      } else {
        console.log("已存在 userId:", userId);
      }
      return userId;
    },

    //上传人脸图像
    async uploadFacePic() {
      if (!this.facePic) {
        alert("请先选择一张图片！");
        return;
      }

      // 获取或生成 userId
      const userId = this.getUserId();
      const formData = new FormData();
      formData.append('userId', userId); // 将 userId 添加到 FormData 中
      formData.append('facePic', this.facePic);

      try {
        const response = await fetch(`${BASE_URL}/uploadFacePic`, {
        //const response = await fetch('http://10.21.140.30:5000/uploadFacePic', {
          method: 'POST',
          body: formData,
        });
        this.facePicInfo = await response.json();
        console.log(this.facePicInfo);
      } catch (error) {
        console.error('上传失败:', error);
      }
    },

    //获取旅行vlog
    async GetVideos(){
        const userId = this.getUserId();
    
        // 检查用户ID是否存在（确保已上传人脸）
        if (!userId) {
            alert('请先上传人脸图像以生成用户信息');
            this.loading = false;
            return;
        }
    
        const formData = new FormData();
        formData.append('musicType', this.selectedMusicType);
        formData.append('userId', userId); // 传递用户ID到后端

        this.loading = true; // 开始加载
        this.taskStatus = null; // 重置任务状态
        this.progeress = 0; // 重置进度

        try {
          const response = await fetch(`${BASE_URL}/getVideo`, {
            method: 'POST',
            body: formData
          });
          const result = await response.json();
          if(result.taskId){
            this.taskId = result.taskId;
            this.startStatusPolling(); // 启动状态轮询
            if (result.output) {
              console.log('result output Video URL:', result.output); // 添加日志，方便调试
              // this.videoUrl = `${BASE_URL}` + result.output; // 设定视频的 URL
              console.log('userId:', userId); // 添加日志，方便调试
              this.videoUrl = `${BASE_URL}/video/${userId}_output.mp4` ; // 设定视频的 URL
              console.log('Full Video URL:', this.videoUrl); // 添加日志，方便调试
            } else {
              console.error(result.error);
            }
          }
        } catch (error) {
          console.error('Error uploading videos:', error);
        } finally {
          this.loading = false; // 结束加载
        }
    },
    // 下载合成的视频
    downloadVideo() {
        const userId = this.getUserId();
        if (!userId) {
            alert('请先上传人脸图像以生成用户信息');
            return;
        }
        
        //window.open(`${BASE_URL}/downloadVideo`); // 下载合成视频
        window.open(`${BASE_URL}/downloadVideo?userId=${userId}`); // 下载合成视频，传递 userId 参数
    },

    // 将webm格式转换为mp4格式
    convertWebmToMp4(blob, fileName) {
        return new Promise((resolve) => {
          const file = new File([blob], fileName, { type: 'video/mp4' });
          resolve(file);
        });
    },
},
    //生命周期钩子用于在组件销毁之前执行一些清理操作
    beforeDestroy() {
      clearInterval(this.intervalIdFrame);
      clearInterval(this.intervalIdRecord);
      this.stopCamera(); // 组件销毁时停止摄像头
},

computed: {
  taskStatusText() {
    const statusMap = {
      'pending': '等待中',
      'processing': '处理中',
      'completed': '已完成',
      'failed': '失败'
    };
    return statusMap[this.taskStatus] || this.taskStatus;
  }
}
  };
 </script>
 
 <style scoped>
 
 @import url("../css/bootstrap.min.css");
 
 @import url("../css/style.css");
 
 @import url("../css/responsive.css");
 
 @import url("../css/jquery.mCustomScrollbar.min.css");
 
 @import url("../css/font-awesome.css");
 
 @import url("../css/owl.carousel.min.css");
 @import url("../css/jquery.fancybox.min.css");
 
   /* 上传按钮 */
   .upload-label {
     display: inline-block;
     padding: 10px 20px;
     background-color: #2b2278;
     color: white;
     border-radius: 4px;
     cursor: pointer;
     margin-bottom: 10px;
   }
   .upload-label input {
     display: none; /* 隐藏原始文件输入 */
   }
   /* 视频预览 */
   .video-preview {
     margin-top: 20px;
     display: flex; /* 使用 Flexbox */
     flex-direction: column; /* 垂直排列 */
     align-items: center; /* 水平居中 */
   }
   .video-preview video {
     max-width: 70%; /* 确保视频不会超出容器 */
     height: auto; /* 保持视频比例 */
   }
   /* 录制预览 */
   .preview-container {
     margin-top: 20px;
   }
   .preview-container video {
     border: 1px solid #dcdfe6;
     border-radius: 4px;
   }
 </style>