# 🤖 LMUCS: A Lightweight LLM-Driven UAV Control System with Multimodal Perception for Autonomous Material Search and Localization

本项目实现了一个通过自然语言指令控制大疆Tello无人机的智能系统。它集成了大语言模型（LLM）进行意图理解，YOLO模型进行实时目标检测，以及MiDaS模型进行单目深度估计，最终实现无人机的自主飞行和目标搜索。

## ✨ 核心功能

  * **自然语言控制**: 用户可以通过自然语言（中文或英文）下达复杂的飞行指令（如 "向前飞1米，然后左转90度"）。
  * **智能意图分类**: LLM自动将用户输入分类为 **飞行控制**、**目标搜索** 或 **程序控制**（如暂停、清空任务）。
  * **实时目标检测**: 使用YOLO模型实时检测无人机视野中的物体。
  * **深度感知**: 采用MiDaS模型估算物体距离，辅助PID控制器进行精准逼近。
  * **自主目标搜索**: 在"搜索任务"中，无人机能自主旋转、调整位置（使用PID控制器），直到找到并居中目标物体。
  * **可视化UI**: 基于PyQt5的图形界面，实时显示无人机画面、识别与深度图、日志信息以及三个核心任务队列（LLM指令、飞行控制、目标搜索）。
  * **模拟模式**: 内置`MockTello`模拟器，可读取本地视频文件进行**功能调试**，无需实体无人机。

## 🏛️ 系统架构

本系统采用前后端分离和多线程架构：

1.  **后端AI服务器 (`export_fastapi.py`)**:
      * 一个基于FastAPI的独立服务器。
      * 提供两个核心API端点：
          * `/ai_chat_ollama/`: 接收文本，调用**Ollama**（运行`qwen2.5_0.5b_drone_q4`模型）进行LLM推理，返回结构化指令。
          * `/yolo_predict/`: 接收图像，运行**YOLO**（`yolo11n_best_9_label_new.pt`）和**MiDaS**（`midas_model_torchscript.pt`）模型，返回检测框和深度信息。

2.  **前端控制应用 (`app.py` & `fly.py`)**:
      * 一个基于PyQt5的主应用程序 (`app.py`)。
      * 核心逻辑 (`fly.py`) 运行在多个并发线程中：
          * **LLM查询线程**: 从GUI获取用户输入，请求FastAPI服务器，并将LLM的解析结果放入对应的任务队列。
          * **YOLO预测线程**: 持续获取无人机图像，请求FastAPI服务器，并更新全局YOLO和深度结果。
          * **飞行控制线程**: 消费"飞行命令队列"，执行具体的无人机动作（如`move_up`）。
          * **目标搜索线程**: 消费"搜索队列"，激活PID控制器，根据YOLO和深度的反馈自动生成飞行指令以逼近目标。
          * **无人机状态线程**: 持续更新电池、高度等信息。

## 🚀 快速开始

### 1\. 依赖环境

请确保您已安装以下环境和库：

  * Python 3.10+
  * PyQt5
  * OpenCV
  * PyTorch & TorchVision
  * Ultralytics (YOLO)
  * FastAPI & Uvicorn
  * Requests
  * DJI-Tello-Py
  * Numpy
  * Pillow

### 2\. (必选) 配置Ollama

- 在之前介绍过

### 3\. (必选) 下载AI模型 (YOLO & MiDaS)

1.  从本项目的 **Releases** 页面下载模型文件。
2.  在项目根目录创建一个 `model` 文件夹。
3.  将下载的两个模型文件放入 `model/` 文件夹中：
      * `yolo11n_best_9_label_new.pt` (YOLOv11n模型)
      * `midas_model_torchscript.pt` (MiDaS深度估计模型)

### 4\. (必选) 修改硬编码路径

> ⚠️ **重要提示**: AI服务器 (`export_fastapi.py`) 中包含了模型的绝对路径，您必须将其修改为您本地的路径。

打开 `export_fastapi.py` 文件:

**修改为** 您本地的相对或绝对路径：

```python
# 修改后的路径 (示例)
val_model = YOLO(r"./model/yolo11n_best_9_label_new.pt")
midas_model = torch.jit.load(r"./model/midas_model_torchscript.pt")
```

## 💻 运行项目

您需要**启动两个进程**：1. 后端AI服务器；2. 前端GUI应用。

### 步骤 1: 启动后端AI服务器

在终端中，运行 `export_fastapi.py`：

```bash
python export_fastapi.py
```

当您看到成功运行在 `http://0.0.0.0:4000` 上的提示时，表示后端已准备就绪。

### 步骤 2: 启动前端GUI应用

打开**新的**终端窗口，运行 `app.py`：

```bash
python app.py
```

GUI界面将会启动，并开始连接（模拟的或真实的）无人机。

## 🕹️ 使用模式

### 1\. 模拟模式 (默认)

默认配置下，系统处于模拟状态。

  * **配置文件**: `fly.py` 中的 `is_mock = True`。
  * **工作方式**: `MockTello.py` 会启动并读取一个本地视频文件。
  * **注意**: 模拟器依赖 `shiyan1210.mp4` (在 `MockTello.py` 中硬编码)。请确保您有此文件，或将其路径更改为您自己的测试视频。


```python
# MockTello.py (约 16 行)
self.cap = cv2.VideoCapture(r"./shiyan1210.mp4")
```

### 2\. 真实无人机模式

要连接真实的Tello无人机，请按以下步骤操作：

1.  **连接无人机WiFi**:

      * 启动您的Tello无人机。
      * 在您的电脑上，连接到Tello的Wi-Fi网络（例如 `TELLO-XXXXX`）。

2.  **修改 `is_mock`**:

      * 打开 `fly.py` 文件。
      * 将 `is_mock` 变量改为 `False`。

    ```python
    # fly.py (约 21 行)
    is_mock = False
    ```

3.  **(可选) 检查无人机IP地址**:

      * 在 `fly.py` (约 40 行) 中，Tello的IP地址被硬编码。

    ```
    'tello': {'host': '192.168.179.164', ...}
    ```

      * Tello的默认IP通常是 `192.168.10.1`。如果您的IP不同（如代码所示），请保持不变；如果是默认IP，请修改 `host` 值为 `192.168.10.1`。

4.  **(可选) 连接到家庭WiFi**:

      * `change_tello_connect_wifi.py` 是一个工具脚本。如果您希望将Tello无人机和您的电脑同时连接到您的家庭WiFi（而不是直连），您可以使用此脚本为无人机配置WiFi的SSID和密码。

5.  **重新运行**: 按照“运行项目”部分的步骤，启动后端服务器和前端应用。现在，应用将连接到您的真实无人机。

## 📁 文件结构说明

```
.
├── app.py                  # PyQt5 GUI 主程序入口
├── fly.py                  # 核心业务逻辑 (多线程, 队列, 全局状态)
├── export_fastapi.py       # 后端AI服务器 (FastAPI, YOLO, MiDaS, Ollama)
├── fun_tools.py            # 辅助函数 (LLM提示工程, API客户端, PID控制器)
├── MockTello.py            # Tello无人机模拟器 (读取本地视频)
├── change_tello_connect_wifi.py # Tello连接到WiFi的脚本
├── ui/
│   ├── main_window.py          # PyQt5 UI 界面定义 (由 .ui 文件生成)
│   └── main_window.ui          # PyQt5 UI 界面定义
├── model/                      # 存放 AI 模型
│   ├── yolo11n_best_9_label_new.pt
│   └── midas_model_torchscript.pt
├── dataset/
│   └── train_prompt.json   # 存放用于 LLM 提示工程的 JSON
├── v/                      # 无人机获取的图片
└── shiyan1210.mp4          # 模拟器使用的视频文件
```