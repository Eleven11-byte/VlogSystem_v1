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
                          <router-link to="/services" class="nav-link">相关服务</router-link>
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
                    <li><router-link to="/services" class="nav-link">相关服务</router-link></li>
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
                       <div class="read_bt"><router-link to="/about">Start</router-link></div>
                    </div>
                 </div>
                 <div class="carousel-item">
                    <div class="container">
                       <h1 class="banner_taital">沿途邮你</h1>
                       <p class="banner_text">获取专属于你的旅行VLOG。</p>
                       <div class="read_bt"><router-link to="/about">开始</router-link></div>
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

                     <h1 class="about_taital">上传人像视频</h1>
                     <p class="about_text">上传或录制人像视频，您将得到专属于您的景区旅行视频。</p>
                     
                     <!--字体-->
                     <p class="about_text">

                     <!-- 文件上传部分 -->
                      <p>
                     <label for="file-upload" class="upload-label">选择视频文件
                     <input type="file" id="file-upload" multiple @change="handleFileUpload" accept="video/*" ref="fileInput" />
                     </label>
                     </p>
  
                     <!-- 摄像头录制部分 -->
                     <p>
                        <el-button-group>
                           <el-button type="primary" @click="startRecording" :disabled="isRecording" round size="large" style="width: 200px;" color="#2b2278" plain class="startRecording">开始录制</el-button>
                           <el-button type="primary" @click="stopRecording" :disabled="!isRecording" round size="large" style="width: 200px;" color="#2b2278">停止录制</el-button>
                        </el-button-group>                       
                     </p>

                     <!-- 已选择的视频文件列表 -->
                     <div v-if="files.length > 0">
                        <p>
                        <h4 class="about_text">已选择文件:</h4>
                        <ul>
                           <li v-for="(file, index) in files" :key="index">
                              视频 {{ index + 1 }} : {{ file.name }}
                           </li>
                         </ul>
                        </p>
                     </div>

                     <p>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                     </p>

                     <!--选择背景音乐类型-->
                     <div>
                     <p>
                        <p>请选择背景音乐：</p>
                        <el-select v-model="selectedMusicType" placeholder="选择背景音乐" style="width: 200px;" class="musicSelect">
                           <el-option label="梦幻" value="bgm01_menghuan"></el-option>
                           <el-option label="忧伤" value="bgm02_youshang"></el-option>
                           <el-option label="舒缓" value="bgm03_shuhuan"></el-option>
                           <el-option label="欢快" value="bgm04_huankuai"></el-option>
                           <el-option label="治愈" value="bgm05_zhiyu"></el-option>
                           <el-option label="星河" value="bgm06_xinghe"></el-option>
                        </el-select>
                     </p>
                     </div>

                     <!-- 上传视频按钮 -->
                     <p><el-button type="primary" @click="uploadVideos" :disabled="files.length === 0" round size="large" color="#2b2278" style="width: 300px;">
                        上传视频
                     </el-button></p>

                     <!-- 加载状态 -->
                     <p></p>
                     <div v-if="loading">加载中，请耐心等待...</div>
                     <p></p>

                     </p>

                 </div>
              </div>

               
              <div class="col-md-6 padding_right_0">
                  <div>
                     <!--图片-->
                     <div v-if="!isRecording&&!videoUrl">
                        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                        <img src="../images/about-img.png" class="about_img">
                     </div>
                     
                     <!-- 录制预览 -->
                     <div v-if="isRecording" class="preview-container">
                        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                        <p class="about_text" style="font-size: 20px; font-weight: bold;">录制中......</p>
                        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>
                        <video ref="recordingPreview" width="600" autoplay playsinline></video>
                     </div>

                     <!-- 合成视频预览 -->
                     <div v-if="videoUrl" class="about_taital_main">
                        <h1 class="about_taital">旅行VLOG预览</h1>
                        <p class="about_text">上传或录制人像视频，您将得到专属于您的景区旅行视频。</p>

                        <!--字体-->
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
import { Upload } from '@element-plus/icons-vue'
 export default {
    data() {
      return {
        files: [], // 存储用户选择的文件以及录制的视频
        videoUrl: '', // 最终合成的视频URL
        loading: false, // 加载状态
        isRecording: false, // 是否正在录制
        mediaRecorder: null, // MediaRecorder对象
        recordedChunks: [], // 存储录制视频的片段
        videoCount: 0, // 录制视频的计数，用于给录制视频起名字
        stream: null ,// 摄像头的媒体流
        selectedMusicType: 'bgm01_menghuan' // 默认选择类型1
      };
    },
    methods: {
      // 处理用户选择的视频文件
      handleFileUpload(event) {
        const input_files = Array.from(event.target.files);
        this.files = [...this.files, ...input_files];
        this.$refs.fileInput.value = null; // 清空 input 以便后续多次选择
      },
      // 开始录制视频
      async startRecording() {
        if (this.isRecording) return; // 如果正在录制，退出方法
        this.isRecording = true;
        try {
          // 获取摄像头和麦克风的流
          this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
          this.$refs.recordingPreview.srcObject = this.stream;
  
          // 初始化MediaRecorder
          this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: 'video/webm' });
          this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
              this.recordedChunks.push(event.data);
            }
          };
  
          // 录制结束后处理视频
          this.mediaRecorder.onstop = async () => {
            const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
            const mp4File = await this.convertWebmToMp4(blob, `video${++this.videoCount}.mp4`);
            this.files.push(mp4File);
            this.recordedChunks = [];
            this.isRecording=false;
  
            // 停止摄像头
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
          };
  
          // 开始录制
          this.mediaRecorder.start();
          //this.isRecording = true;
        } catch (error) {
          console.error('Error accessing media devices.', error);
        }
      },
      // 停止录制视频
      stopRecording() {
        if (this.isRecording) {
          this.mediaRecorder.stop();
          this.isRecording = false;
        }
      },
      // 上传视频到后端
      async uploadVideos() {
        const formData = new FormData();
        this.files.forEach(file => {
          formData.append('files', file);
        });

        formData.append('musicType', this.selectedMusicType); // 将选择的音乐类型添加到 FormData 中
  
        this.loading = true; // 开始加载
  
        try {
          const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData
          });
          const result = await response.json();
          if (result.output) {
            console.log('Video URL:', result.output); // 添加日志，方便调试
            this.videoUrl = 'http://localhost:5000' + result.output; // 设定视频的 URL
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
        window.open('http://localhost:5000/download'); // 下载合成视频
      },
      // 将webm格式转换为mp4格式
      convertWebmToMp4(blob, fileName) {
        return new Promise((resolve) => {
          const file = new File([blob], fileName, { type: 'video/mp4' });
          resolve(file);
        });
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