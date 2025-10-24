<template>
  <div>
    <!-- 人像图片上传部分 -->
    <div>
      <input type="file" @change="selectFacePic" accept="image/jpeg, image/png" />
      <button @click="uploadFacePic">上传</button>
      <div v-if="imageUrl">
          <p>预览人脸图像：</p>
          <img :src="imageUrl" alt="预览" width="300" />
      </div>
      <div v-if="facePicInfo">
          <p>上传状态：{{ facePicInfo }}</p>
      </div>
    </div>
    <!-- 摄像头录制部分 -->
    <div>
      <el-button type="primary" @click="startCamera" :disabled="isCamera">开启摄像头</el-button>
      <el-button type="primary" @click="stopCamera" :disabled="!isCamera">关闭摄像头</el-button>
    </div>
    <!-- 录制预览 -->
    <div v-if="isCamera">
      <video ref="CameraVideo" width="640" height="480" autoplay></video>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      view:'view1',//摄像机序列或经典代号

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

      //user:'',//录像人姓名
      //position:'',//录像景点
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
        //清楚拍摄人以及拍摄位置
        //this.user=[];console.log('user:',this.user);
        //this.position=[];console.log('position:',this.position);
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
        await fetch('http://localhost:5000/uploadFrames', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: frameData }),
        });
      } catch (error) {
        console.error("Error sending frame to backend: ", error);
      }
  },


    //检查录制状态
    async checkRecordingStatus() {
      const response = await fetch('http://localhost:5000/isRecording');
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

    //摄像机序列或景点代号
    formData.append('view',this.view);

    await fetch('http://localhost:5000/uploadVideo', {
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
      const response = await fetch('http://localhost:5000/uploadFacePic', {
        method: 'POST',
        body: formData,
      });
      this.facePicInfo = await response.json();
      console.log(this.facePicInfo);
    } catch (error) {
      console.error('上传失败:', error);
    }
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

<style>
/* 你的样式 */
</style>