# MyFirstCG Program大作业

跟随开源OpenGL教程，实现了创建窗口、坐标变换、物体建模、自定义着色器、手动控制摄像机、Phong式光源、物体材质和光照纹理贴图。运行效果如下：

```c
$ make run
```

[![image](.\effect(final).png)](https://github.com/YifengChen2023/Phong-Lighting-Model/blob/main/effect(final).png)

可以移动鼠标控制摄像机的视角，按动`WSAD`键控制摄像机的运动方向。按`ESC`退出。

## 依赖

- glad库，用来管理OpenGL的函数指针
- glfw库，用来创建上下文处理输入
- glm库，用来定义几何、坐标等数学信息
- stb_image.h，用来导入纹理图片
- camera.h，定义摄像机
- shader.h，定义片元着色器和顶点着色器的构造
- resources，纹理资源
- `.fs`片元着色器，`.vs`顶点着色器

 都在压缩包中给出。

