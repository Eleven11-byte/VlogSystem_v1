import { defineConfig } from 'vite' // 导入 Vite 的配置函数
import vue from '@vitejs/plugin-vue'// 导入 Vue 插件（让 Vite 支持 Vue 项目）

// https://vitejs.dev/config/
// 导出 Vite 的配置对象——
// 各种配置项...
export default defineConfig({
// 注册 Vite 插件，没有这个插件，Vite 无法识别 Vue 语法，项目启动会报错。
  plugins: [
    vue()
  ],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: [
      'localhost',
      '127.0.0.1'
    ]
  }
})
