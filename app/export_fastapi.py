import base64
import os
import json
import asyncio
import datetime

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F


from ultralytics import YOLO

import requests
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse


def get_depth_img(midas_model, image):
    """
    获取深度图和距离图。

    参数:
        midas_model: MiDaS 模型实例
        image: 输入图片，可以是 NumPy 数组或 PIL.Image 对象

    返回:
        colored_depth: 可视化的彩色深度图
        distance_map: 距离图，值越大表示越远
    """
    # 确保输入为 PIL.Image 格式
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image = image.convert("RGB")
    image = image.resize((384, 384))  # 替换为模型要求的输入大小
    input_tensor = F.to_tensor(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    # 预测深度
    with torch.no_grad():
        depth_map = midas_model(input_tensor)

    # 转换为 NumPy 数组
    depth_map_np = depth_map.squeeze().cpu().numpy()

    # 保存或可视化深度图
    depth_min = depth_map_np.min()
    depth_max = depth_map_np.max()

    # 归一化到0~1之间
    normalized_depth_map = (depth_map_np - depth_min) / (depth_max - depth_min)

    # 对于 MiDaS 的逆深度：normalized_depth_map 中数值越大意味着越近，
    # 如果需要 0 表示近，1 表示远，则翻转一下
    distance_map = 1 - normalized_depth_map

    # 转为可视化的彩色深度图（仅用于查看）
    normalized_depth_255 = (normalized_depth_map * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(normalized_depth_255, cv2.COLORMAP_INFERNO)

    return colored_depth, distance_map


app = FastAPI()

val_model = YOLO(r"./model/yolo11n_best_9_label_new.pt")


# 加载 TorchScript 模型
midas_model = torch.jit.load(r"./model/midas_model_torchscript.pt")
midas_model.eval()

# Dictionary to store conversation history by chatid
start_time = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
conversations = {}


@app.post("/yolo_predict/")
async def yolo_predict(file: UploadFile = File(...)):
    # 读取文件内容为 NumPy 数组
    file_content = await file.read()

    # 加载文件到内存用于处理
    file_bytes = np.frombuffer(file_content, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 进行 YOLO 预测
    results = val_model.predict(source=img, save=False, conf=0.5)

    # 获取深度图和距离图
    colored_depth, distance_map = get_depth_img(midas_model=midas_model, image=img)

    img_height, img_width = img.shape[:2]
    img_area = img_width * img_height  # 整个图像的面积

    result = [{
        'name': 'null',
        'confidence': 0,
        'center_x_ratio': 0,
        'center_y_ratio': 0,
        'box_area_ratio': 0,
        'avg_distance_value': 0,
    }]

    if len(results[0].boxes.data) > 0:
        result.clear()
    yolo_results = []
    for box in results[0].boxes.data:
        x1, y1, x2, y2 = box[:4]
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height  # 物体框的面积
        box_area_ratio = box_area / img_area  # 物体框面积相对于图像面积的比例

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x_ratio = center_x / img_width
        center_y_ratio = center_y / img_height
        object_name = results[0].names[int(box[5])]
        confidence = box[4]

        # 计算在 depth_map 上的对应坐标(整数索引)
        scale_x = 384.0 / img_width
        scale_y = 384.0 / img_height

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        x1_scaled = max(0, min(x1_scaled, 383))
        x2_scaled = max(0, min(x2_scaled, 383))
        y1_scaled = max(0, min(y1_scaled, 383))
        y2_scaled = max(0, min(y2_scaled, 383))

        obj_depth_region = distance_map[y1_scaled:y2_scaled, x1_scaled:x2_scaled]

        if obj_depth_region.size > 0:
            avg_distance_value = obj_depth_region.mean()
        else:
            avg_distance_value = 0.5
        yolo_results.append({'x1': box[0].item().__round__(3),
                             'y1': box[1].item().__round__(3),
                             'x2': box[2].item().__round__(3),
                             'y2': box[3].item().__round__(3),
                             'confidence': confidence.item().__round__(3),
                             'object_name': object_name})
        result.append({
            'name': object_name,
            'confidence': confidence.item().__round__(3),
            'center_x_ratio': center_x_ratio.item().__round__(2),
            'center_y_ratio': center_y_ratio.item().__round__(2),
            'box_area_ratio': box_area_ratio.item().__round__(3),
            'avg_distance_value': float(avg_distance_value).__round__(3)
        })

    _, depth_img_encoded = cv2.imencode(".jpg", colored_depth)
    depth_img_encoded = depth_img_encoded.tobytes()  # 提取字节数据
    res = {
        'result': result,
        'yolo_result': yolo_results,
        'depth_img_encoded': base64.b64encode(depth_img_encoded).decode("utf-8")

    }
    return JSONResponse(content=jsonable_encoder(res))


@app.get("/")
async def root():
    return JSONResponse(content={"datetime": str(datetime.datetime.now()), 'msg': 'hello world!'})


# 新增的下载接口
@app.get("/download_image/")
async def download_image(image_path: str):
    """
    下载图像文件的接口。
    :param image_path: 客户端传递的图像的绝对路径
    :return: 返回图像文件作为响应
    """
    if os.path.exists(image_path):
        # 返回图像文件
        return FileResponse(path=image_path, media_type='image/jpeg', filename=os.path.basename(image_path))
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)


def ollama(prompt="1+1=?", model="qwen2.5_0.5b_drone_q4", system=''):
    t_json = {"model": model, "prompt": prompt,
              'stream': False, "keep_alive": 3600 * 24 * 7}
    if system != '':
        t_json['system'] = system
    r = requests.post("http://127.0.0.1:11434/api/generate",
                      timeout=600,
                      json=t_json,
                      stream=False)
    r.raise_for_status()
    return r.json()['response']


@app.post("/ai_chat_ollama/")
async def ai_chat_ollama(request: Request):
    json_data = await request.json()  # 从请求体中获取JSON数据

    chatid = json_data.get('chatid', f'unknown_{str(datetime.datetime.now())}')
    message = json_data.get('message')

    # Initialize conversation history if chatid is new
    if chatid not in conversations:
        conversations[chatid] = []

    # Skip if the message is null or empty
    if message:
        timestamp = datetime.datetime.now()

        # Add user message to conversation history
        conversations[chatid].append({
            'role': 'user',
            'message': message,
            'timestamp': str(timestamp)
        })

        # 调用模型进行对话生成
        response = ollama(prompt=message)

        # Add server reply to conversation history
        conversations[chatid].append({
            'role': 'server',
            'message': response,
            'timestamp': str(datetime.datetime.now())
        })
        # 记录对话日志
        with open(f'./log/conversations_log_{start_time}.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(conversations, ensure_ascii=False, indent=4))
        # 构建返回响应
        response_data = {'chatid': chatid,
                         'reply': response,
                         'history': conversations[chatid]
                         }
    else:
        response_data = {'chatid': chatid,
                         'history': conversations[chatid]
                         }
    return response_data


# 如果不借助ollama，请使用下面的，也需要修改fun_tools里面的请求路径
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_path = r'C:\Users\wp\Desktop\project\fly_new\model\qwen2.5_0.5b_yolo_new_9-merged'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
#                                              trust_remote_code=True).eval()
# 定义API的路由，处理POST请求
# @app.post("/ai_chat/")
# async def ai_chat(request: Request):
#     json_data = await request.json()  # 从请求体中获取JSON数据
#
#     chatid = json_data.get('chatid', f'unknown_{str(datetime.datetime.now())}')
#     message = json_data.get('message')
#
#     # Initialize conversation history if chatid is new
#     if chatid not in conversations:
#         conversations[chatid] = []
#
#     # Skip if the message is null or empty
#     if message:
#         timestamp = datetime.datetime.now()
#
#         # Add user message to conversation history
#         conversations[chatid].append({
#             'role': 'user',
#             'message': message,
#             'timestamp': str(timestamp)
#         })
#
#         # 构建对话输入
#         # t_i = t_input.replace('{}', message)
#         # chat = [{"role": "system", "content": f"{t_system}"},
#         #         {"role": "user", "content": f"{t_i}"}]
#         chat = [{"role": "user", "content": f"{message}"}]
#
#         # 调用模型进行对话生成
#         prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
#
#         # 异步执行生成，避免阻塞
#         loop = asyncio.get_event_loop()
#         outputs = await loop.run_in_executor(None, lambda: model.generate(input_ids=inputs.to(model.device),
#                                                                           max_new_tokens=150))
#         outputs = tokenizer.decode(outputs[0])
#
#         # 处理模型输出
#         response = outputs.split('<|im_start|>assistant\n')[-1]
#         response = response.replace('<|im_end|>', '')
#
#         # 如果指令超过10个，提示指令太多，请重新输入
#         if response.count(';') > 10:
#             response = '您输入的指令太多，请重新输入。'
#
#         # Add server reply to conversation history
#         conversations[chatid].append({
#             'role': 'server',
#             'message': response,
#             'timestamp': str(datetime.datetime.now())
#         })
#         # 记录对话日志
#         if not os.path.exists('./log'):
#             os.makedirs('./log')
#         with open(f'./log/conversations_log_{start_time}.json', 'w', encoding='utf-8') as f:
#             f.write(json.dumps(conversations, ensure_ascii=False, indent=4))
#         # 构建返回响应
#         response_data = {'chatid': chatid,
#                          'reply': response,
#                          'history': conversations[chatid]
#                          }
#     else:
#         response_data = {'chatid': chatid,
#                          'history': conversations[chatid]
#                          }
#     return response_data
#

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=4000)
