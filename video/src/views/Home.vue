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
                        <p class="banner_text">This is a travel website.</p>
                        <div class="read_bt"><router-link to="/about">Start</router-link></div>
                     </div>
                  </div>
                  <div class="carousel-item">
                     <div class="container">
                        <h1 class="banner_taital">沿途邮你</h1>
                        <p class="banner_text">这是一个旅游网站。</p>
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
                     <h1 class="about_taital">关于景区</h1>
                     <p class="about_text">景区有这些美丽的风景 </p>
                     <div class="readmore_bt"><router-link to="/about">景区风景</router-link></div>
                  </div>
               </div>
               <div class="col-md-6 padding_right_0">
                  <div><img src="../images/about-img.png" class="about_img"></div>
               </div>
            </div>
         </div>
      </div>
      <!-- header section end -->
      <!-- services section start -->
      <div class="services_section layout_padding">
         <div class="container">
            <h1 class="services_taital">相关服务</h1>
            <p class="services_text">不仅可以看到美丽风景，还可以获取旅行vlog。</p>
            <div class="services_section_2">
               <div class="row">
                  <div class="col-md-4">
                     <div><img src="../images/img-1.png" class="services_img"></div>
                     <div class="btn_main"><router-link to="/services">可获取服务1</router-link></div>
                  </div>
                  <div class="col-md-4">
                     <div><img src="../images/img-2.png" class="services_img"></div>
                     <div class="btn_main active"><router-link to="/services">可获取服务2</router-link></div>
                  </div>
                  <div class="col-md-4">
                     <div><img src="../images/img-3.png" class="services_img"></div>
                     <div class="btn_main"><router-link to="/services">可获取服务3</router-link></div>
                  </div>
               </div>
            </div>
         </div>
      </div>
      <!-- about section end -->
      <!-- blog section start -->
      <div class="blog_section layout_padding">
         <div class="container">
            <h1 class="blog_taital">旅行vlog</h1>
            <p class="blog_text">您可以获取为您拍摄的专属于您的旅行vlog</p>
            <div class="play_icon_main">
               <div class="play_icon"><router-link to="/blog"><img src="../images/play-icon.png"></router-link></div>
            </div>
         </div>
      </div>
      <!-- blog section end -->
      <!-- client section start -->
      <div class="client_section layout_padding">
         <div class="container">
            <h1 class="client_taital">景区评价</h1>
            <div class="client_section_2">
               <div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">
                  <ol class="carousel-indicators">
                     <li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>
                     <li data-target="#carouselExampleIndicators" data-slide-to="1"></li>
                     <li data-target="#carouselExampleIndicators" data-slide-to="2"></li>
                  </ol>
                  <div class="carousel-inner">
                     <div class="carousel-item active">
                        <div class="client_main">
                           <div class="box_left">
                              <p class="lorem_text">遇见圆明园，开启一场跨越时空的美丽邂逅。
                                 “圆明园从前有多美”展览以四季观赏线路为线索，将圆明园四十景、西洋楼一一串联。
                                 观展读者以九州清晏为地点，纵览圆明园以“家国天下”为设计理念的恢弘布局；赏春夏秋冬美景，寻觅皇家生活起居的点滴；探西洋建筑全貌，感受中西融合之美。
                                 展览以生动的图片和丰富的细节，通过园林构造、建筑特色、起居文化、四时之景等方面展现万园之园昔日的风采。
                                 观展者仿佛置身于圆明园的宏伟景象中，细品这座伟大园林蕴含的中国历史人文、思想情感与审美意趣，获得感官与心灵的双重触动。</p>
                           </div>
                           <div class="box_right">
                              <div class="client_taital_left">
                                 <div class="client_img"><img src="../images/client-img.png"></div>
                                 <div class="quick_icon"><img src="../images/quick-icon.png"></div>
                              </div>
                              <div class="client_taital_right">
                                 <h4 class="client_name">小红</h4>
                                 <p class="customer_text">旅客</p>
                              </div>
                           </div>
                        </div>
                     </div>
                     <div class="carousel-item">
                        <div class="client_main">
                           <div class="box_left">
                              <p class="lorem_text">“故宫”有两层含义，一是指故宫建筑群，又名紫禁城，是500多年间国家运行的中心和权力象征；
                                 二是指故宫博物院，它是我国最著名的博物院和世界顶级的博物馆，是五千年中华优秀传统文化的守护者和传承者。
                                 在我国的文化体系中，作为凝聚中华传统文化精髓、文化成果精品的殿堂，故宫有着独一无二的地标意义和文化象征。
                                 历经时代变迁，故宫由“宫”变“院”，其价值早已不限于朱红高墙内的宫阙亭台，有形的故宫成为中华优秀传统文化传承、中外文化交流的场所和见证者；
                                 无形的故宫延续着中华历史文脉，承载着国家记忆，彰显着中华文化的软实力。</p>
                           </div>
                           <div class="box_right">
                              <div class="client_taital_left">
                                 <div class="client_img"><img src="../images/client-img.png"></div>
                                 <div class="quick_icon"><img src="../images/quick-icon.png"></div>
                              </div>
                              <div class="client_taital_right">
                                 <h4 class="client_name">小明</h4>
                                 <p class="customer_text">旅客</p>
                              </div>
                           </div>
                        </div>
                     </div>
                     <div class="carousel-item">
                        <div class="client_main">
                           <div class="box_left">
                              <p class="lorem_text">从总体上来看，这一整座宏大的坛建筑群有内外两重宫墙。
                                 经过刻意设计的天坛宫墙，其平面的格局为南方北圆，即将内外宫墙的南墙与东西墙均按正角直交的矩形平面布置，
                                 而将内外宫墙的北墙都按照圆润的弧线形平面布置，从而突显了整座天坛建筑群所表征的古代中国人理念中的“天圆地方”的宇宙象征性内涵。
                                 北京天坛在初建时，被布置在北京内城城门外的东南，这与中国古代礼制建筑中所遵循的南郊祀典之礼相吻合。
                                 到了明代中叶，随着城南的人口积聚，在内城之外又砌筑了外城，形成了我们所熟知的“凸”字形的明清北京城城市平面。
                                 而天坛，以及与之相邻的同样用于南郊祀典的先农坛，都被纳入了北京外城之内。今日的天坛，位于北京外城中轴线的东侧。</p>
                           </div>
                           <div class="box_right">
                              <div class="client_taital_left">
                                 <div class="client_img"><img src="../images/client-img.png"></div>
                                 <div class="quick_icon"><img src="../images/quick-icon.png"></div>
                              </div>
                              <div class="client_taital_right">
                                 <h4 class="client_name">小黎</h4>
                                 <p class="customer_text">旅客</p>
                              </div>
                           </div>
                        </div>
                     </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
      <!-- client section start -->
      <!-- choose section start -->
      <div class="choose_section layout_padding">
         <div class="container">
            <h1 class="choose_taital">选择我们的原因</h1>
            <p class="choose_text">我们的景区有这些优点 </p>
            <div class="read_bt_1"><router-link to="/a">阅读更多</router-link></div>
            <div class="newsletter_box">
               <h1 class="let_text">更多详情</h1>
               <div class="getquote_bt"><router-link to="/contact">联系我们</router-link></div>
            </div>
         </div>
      </div>
      <!-- choose section end -->
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