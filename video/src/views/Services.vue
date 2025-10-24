<template>
   <!--相关服务-->
    <div>
        <!-- header section start -->
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
            <div class="container-fluid">
               <div class="logo"><router-link to="/"><img src="../images/logo.png"></router-link></div>
               <div class="menu_main">
                  <ul>
                     <li class="active"><router-link to="/" class="nav-link">首页</router-link></li>
                     <li><router-link to="/about" class="nav-link">关于景区</router-link></li>
                     <li> <router-link to="/services" class="nav-link">实时录制</router-link></li>
                     <li><router-link to="/blog" class="nav-link">旅行vlog</router-link></li>
                     <li><router-link to="/contact" class="nav-link">联系我们</router-link></li>
                  </ul>
               </div>
            </div>
         </div>
      </div>
      <!-- header section end -->
      <!-- services section start -->
      <div class="services_section layout_padding">
         <div class="container">
            <h1 class="services_taital">实时录制人像视频</h1>
            <p class="services_text">录制人像视频，同于制作专属于您的旅行VLOG。</p>
            <div class="services_section_2">
               <div class="row text-center">
                  <div class="col-md-6 d-flex flex-column align-items-center">
                     <p></p>
                     <!-- 摄像头开启关闭部分 -->
                     <p>
                         <el-button-group>
                            <el-button type="primary" @click="startCamera" :disabled="isCamera" round size="large" style="width: 200px;" color="#2b2278" plain class="startRecording">开启摄像头</el-button>
                            <el-button type="primary" @click="stopCamera" :disabled="!isCamera" round size="large" style="width: 200px;" color="#2b2278">关闭摄像头</el-button>
                         </el-button-group>                       
                      </p>

                      <p></p>

                      <!--选择景点位置-->
                     <p>
                        <p>请选择景点位置：</p>
                        <el-select v-model="selectedViewPosition" placeholder="选择景点位置" style="width: 200px;" class="viewPosition">
                           <el-option label="景点1" value="view1"></el-option>
                           <el-option label="景点2" value="view2"></el-option>
                           <el-option label="景点3" value="view3"></el-option>
                        </el-select>
                     </p>                     
                  </div>

                  <div class="col-md-6 padding_right_0">
                     <!--摄像头替换图像-->
                     <div v-if="!isCamera"><img src="../images/img-1.png" class="services_img"></div>
                     <!-- 摄像头画面预览 -->
                     <div v-if="isCamera" class="preview-container">
                         <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                         <p class="about_text" style="font-size: 20px; font-weight: bold;">录制中......</p>
                         <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                         <video ref="CameraVideo" width="600" autoplay playsinline></video>
                      </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
      <!-- services section end -->
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
      };
    },
    methods: {
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
          //await fetch('http://10.21.140.30:5000/uploadFrames', {  
          //await fetch('http://localhost:5000/uploadFrames', {
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
    //上传人脸图像
    async uploadFacePic() {
      if (!this.facePic) {
        alert("请先选择一张图片！");
        return;
      }

      const formData = new FormData();
      formData.append('facePic', this.facePic);

      try {
        const response = await fetch(`${BASE_URL}/uploadFacePic`, {
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
        const formData = new FormData();
        formData.append('musicType', this.selectedMusicType); // 将选择的音乐类型添加到 FormData 中

         this.loading = true; // 开始加载
  
        try {
          const response = await fetch(`${BASE_URL}/getVideo`, {
            method: 'POST',
            body: formData
          });
          const result = await response.json();
          if (result.output) {
            console.log('Video URL:', result.output); // 添加日志，方便调试
            this.videoUrl = `${BASE_URL}` + result.output; // 设定视频的 URL
          } else {
            console.error(result.error);
          }
        } catch (error) {
          console.error('Error uploading videos:', error);
        } finally {
          this.loading = false; // 结束加载
        }
    },
    // 下载合成的视频
    downloadVideo() {
        window.open(`${BASE_URL}/downloadVideo`); // 下载合成视频
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

</style>