# ZJU-D2NN

基于双层空间光调制器（SLM1/SLM2）的光学神经网络（D2NN）项目，包含：
- 真实硬件的数据采集与系统标定（SLM + 相机）
- 离线系统辨识训练（由采集到的相位-强度数据拟合传播模型）
- 纯仿真训练与推理（MNIST）

> 注：部分脚本依赖海康 MV 相机与 MVS SDK（`MvImport/`），以及双 HDMI 输出用于 SLM 显示。

## 依赖环境

- Python 3.8+
- 主要依赖：`numpy` `torch` `opencv-python` `tensorboardX`
- 安装：`pip install -r requirements.txt`

## 目录与文件说明（含使用方法）

### 顶层文件

- `LICENSE`：开源许可证。
- `requirements.txt`：Python 依赖列表，使用 `pip install -r requirements.txt` 安装。
- `Windows SDK Python.chm`：Windows 下相机 SDK/接口文档（离线帮助）。

- `capture_dataset.py`：采集离线数据集（SLM1 相位 ↔ 相机强度）。
  - 用途：随机生成 SLM1 相位并显示到 HDMI，同时采集相机灰度图，保存为 `.npz`。
  - 用法示例：
    ```bash
    python capture_dataset.py --num 200 --slm1_channel 1 --slm2_channel 2 --output dataset.npz
    ```

- `hdmi_example.py`：测试 HDMI 输出，循环显示图形。
  - 用途：确认 SLM/显示器通道可用。
  - 用法示例：
    ```bash
    python hdmi_example.py
    ```

- `read_from_camera.py`：单次抓取相机图像并返回 BGR 数组（含测试入口）。
  - 用途：快速验证相机连接与图像采集。
  - 用法示例：
    ```bash
    python read_from_camera.py
    ```

- `train_system.py`：系统传播模型（ASM + 薄透镜 + 位移/倾斜），提供可组合模块。
  - 用途：被 `train_system_offline.py` 调用，构建可学习的传播链路。
  - 用法：作为模块导入，例如 `from train_system import build_first_half_model`。

- `train_system_offline.py`：离线系统辨识训练（用采集数据拟合传播模型）。
  - 用途：输入 `.npz/.pt` 数据集，优化传播参数（距离/偏移/倾角等）。
  - 用法示例：
    ```bash
    python train_system_offline.py \
      --data dataset.npz \
      --wavelength 5.32e-7 --dx 1.25e-5 --dy 1.25e-5 \
      --f_4f 0.25 --f_camera 0.05 \
      --l2_to_cam_lens 0.25 --cam_lens_to_sensor 0.05
    ```

- `train_onn.py`：在真实/离线传播模型上训练 SLM2 相位（MNIST 分类）。
  - 用途：加载 TorchScript 传播模块，优化 SLM2 相位作为“权重”。
  - 用法示例：
    ```bash
    python train_onn.py --prop1_ts ./prop1.ts --prop2_ts ./prop2.ts --H 128 --W 128
    ```

- `logger.py`：TensorBoardX 日志封装。
  - 用途：训练过程中记录标量、图像、视频。
  - 用法：`from logger import Logger`。

- `pytorch_utils.py`：PyTorch 设备/张量小工具。
  - 用途：初始化 GPU、numpy↔torch 互转。
  - 用法：`import pytorch_utils as ptu; ptu.init_gpu()`。

- `utils.py`：HDMI 输出与相机采集辅助函数。
  - 用途：`write_to_hdmi()` 显示图像；`list_hdmi_outputs()` 列出输出；`read_from_camera()` 相机抓帧。
  - 用法：`import utils` 后调用对应函数。

### `MvImport/`（海康相机 SDK Python 封装）

- `MvImport/__init__.py`：包初始化。
- `MvImport/MvCameraControl_class.py`：相机控制主类（SDK 入口）。
- `MvImport/CameraParams_const.py`、`CameraParams_header.py`：相机参数常量/结构体。
- `MvImport/MvErrorDefine_const.py`、`MvISPErrorDefine_const.py`：错误码定义。
- `MvImport/PixelType_header.py`：像素格式定义。

> 以上文件由相机 SDK 提供或改造，用于 `read_from_camera.py` 和 `utils.py`。

### `simulation/`（纯仿真训练/推理）

- `simulation/system.py`：光学系统仿真（ASM、薄透镜、重采样），提供 `OpticalSystem`。
  - 用途：构建仿真传播链路，供训练/推理调用。
  - 用法：`import system; sys_model = system.OpticalSystem()`。

- `simulation/ONN_train.py`：仿真训练 SLM2 相位（MNIST）。
  - 用途：使用 `system.py` 的物理模型训练 SLM2。
  - 用法示例：
    ```bash
    python simulation/ONN_train.py --max_steps 500 --roi_box 32
    ```

- `simulation/main.py`：仿真推理单张图片（MNIST 或自定义图像）。
  - 用途：加载训练好的 SLM2 相位，生成相机强度图并预测类别。
  - 用法示例：
    ```bash
    python simulation/main.py --ckpt ./simulation/checkpoints/onn_slm2.pt --mnist_index 0
    ```

- `simulation/data/`：MNIST 数据默认下载目录（运行训练/推理后生成）。
- `simulation/checkpoints/`：仿真训练保存的权重目录。

### 其它目录

- `__pycache__/`、`simulation/__pycache__/`：Python 字节码缓存，可忽略。

## 常见流程参考

1) HDMI/相机连通性测试：
```bash
python hdmi_example.py
python read_from_camera.py
```

2) 采集离线数据（SLM1 相位 ↔ 相机强度）：
```bash
python capture_dataset.py --num 200 --output dataset.npz
```

3) 离线系统辨识（拟合传播参数）：
```bash
python train_system_offline.py --data dataset.npz \
  --wavelength 5.32e-7 --dx 1.25e-5 --dy 1.25e-5 \
  --f_4f 0.25 --f_camera 0.05 --l2_to_cam_lens 0.25 --cam_lens_to_sensor 0.05
```

4) 纯仿真训练与推理：
```bash
python simulation/ONN_train.py --max_steps 500
python simulation/main.py --ckpt ./simulation/checkpoints/onn_slm2.pt --mnist_index 0
```
