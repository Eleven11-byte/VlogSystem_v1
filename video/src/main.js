import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'


//先拿到当前所创建的vue实例
//后面的方法都是在调用这个实例的方法
const app=createApp(App)
//路由挂载
app.use(router)
app.use(ElementPlus)
//挂载到id为app节点上。而id为app节点在index.html中体现：
app.mount('#app')


