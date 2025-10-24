<template>
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
            <div class="container-fluid">
               <div class="logo"><router-link to="/"><img src="../images/logo.png"></router-link></div>
               <div class="menu_main">
                  <ul>
                     <li class="active"><router-link to="/" class="nav-link">首页</router-link></li>
                     <li><router-link to="/about" class="nav-link">关于景区</router-link></li>
                     <li> <router-link to="/services" class="nav-link">相关服务</router-link></li>
                     <li><router-link to="/blog" class="nav-link">旅行vlog</router-link></li>
                     <li><router-link to="/contact" class="nav-link">联系我们</router-link></li>
                  </ul>
               </div>
            </div>
         </div>
      </div>
      <!-- header section end -->
      <!-- blog section start -->
      <div class="blog_section layout_padding margin_top_90">
         <div class="container">
            <h1 class="blog_taital">旅行vlog</h1>
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