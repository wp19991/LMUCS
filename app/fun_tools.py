import base64
import json
import os
import random
import time

import cv2
import numpy as np
import requests
from djitellopy import Tello, BackgroundFrameRead

from MockTello import MockTello, Frame


def new_tello(tello_connect: dict, is_mock=True) -> (Tello | MockTello, BackgroundFrameRead):
    # 初始化 Tello 对象
    if is_mock:
        tello = MockTello(host=tello_connect['host'])
    else:
        tello = Tello(host=tello_connect['host'])
    tello.connect()
    tello.query_sdk_version()

    # tello.set_video_bitrate(Tello.BITRATE_5MBPS)
    # tello.set_video_resolution(Tello.RESOLUTION_720P)
    # tello.set_video_fps(Tello.FPS_30)
    # tello.set_video_direction(Tello.CAMERA_FORWARD)

    tello.get_battery()
    tello.get_udp_video_address()
    tello.streamon()
    frame_read = tello.get_frame_read()
    img = frame_read.frame
    while img is None:
        img = frame_read.frame
    tello.set_video_direction(0)
    return tello, frame_read


def __save_img_into_disk(img_bgr: cv2.typing.MatLike, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(os.path.abspath(save_path), f"img_forward_{time.time()}.jpg")

    cv2.imwrite(file_save_path, img_bgr)
    # print('保存图片',file_save_path)
    return file_save_path


# 获取无人机图片文件
def tello_get_img(tello: Tello | MockTello, frame_read: BackgroundFrameRead | Frame, save_path='') -> str:
    """
    tello获取前方图像
    :param tello: tello对象
    :param frame_read: 读取画面的对象，使用tello.get_frame_read()获取
    :return: 获取的图片的路径
    """
    # tello.set_video_direction(Tello.CAMERA_FORWARD) #频繁切换0/1会导致卡顿
    if type(tello) == MockTello:
        frame_read = tello.get_frame_read()
    t_img = frame_read.frame

    while t_img is None or t_img.shape[0] != 720:
        t_img = frame_read.frame

    if type(tello) is MockTello:
        t_img_bgr = t_img
    else:
        t_img_bgr = cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR)
    if save_path == '':
        save_path = os.path.join(os.path.expanduser("~"), "llm_drone_data")
    file_save_path = __save_img_into_disk(t_img_bgr, save_path)
    return file_save_path


# 获取无人机信息
def get_tello_info(tello: Tello | MockTello) -> dict:
    """
    获取无人机高度和电量信息
    :param tello: tello对象
    :return: 无人机高度和电量信息的字典
    """
    try:
        current_height = float(tello.get_distance_tof()).__round__(2)
    except:
        current_height = 0
    battery = float(tello.get_battery()).__round__(2)
    return {'current_height': current_height, 'battery': battery}


# 执行无人机飞行指令
def tello_run_skill(tello: Tello | MockTello, task_str: str = 'move_up 90') -> bool:
    """
    直接运行无人机指令
    :param tello: tello对象
    :param task_str: 只能接受 move_up 90 或者 move_up 90 cm. 或者 move_up 90 cm
    :return: 是否执行成功
    """
    task_str = task_str.replace('.', '').replace(';', '').replace(' degrees', '')

    t_type = 'cm'
    if ' m' == task_str[-2:]:
        t_type = 'm'
    elif ' ft' == task_str[-3:]:
        t_type = 'ft'
    elif ' in' == task_str[-3:]:
        t_type = 'in'

    task_str = task_str.replace(' cm', '').replace(' m', '')
    task_str = task_str.replace(' ft', '').replace(' in', '')
    task_str = task_str.strip()
    t_skill = task_str.split(' ')[0]
    t_data = int(task_str.split(' ')[-1]) if task_str.split(' ')[-1].isdigit() else 0

    if t_type == 'm':
        t_data = t_data * 100
    elif t_type == 'ft':
        t_data = int(t_data * 30.48)
    elif t_type == 'in':
        t_data = int(t_data * 2.54)

    t_turn_data = t_data  # 转向的时候不需要控制在0-100
    t_run_data = t_data
    if t_run_data < 0:
        t_run_data = 0
    if t_run_data > 100:
        t_run_data = 100
    try:
        if t_skill == 'take_off':
            tello.takeoff()
            # tello.move_up(60)
        elif t_skill == 'land':
            tello.land()
        elif t_skill == 'move_up':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_up(20 + t_run_data)
                tello.move_down(20)
            else:
                tello.move_up(t_run_data)
        elif t_skill == 'move_down':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_down(20 + t_run_data)
                tello.move_up(20)
            else:
                tello.move_down(t_run_data)
        elif t_skill == 'move_right':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_right(20 + t_run_data)
                tello.move_left(20)
            else:
                tello.move_right(t_run_data)
        elif t_skill == 'move_left':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_left(20 + t_run_data)
                tello.move_right(20)
            else:
                tello.move_left(t_run_data)
        elif t_skill == 'move_forward':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_forward(20 + t_run_data)
                tello.move_back(20)
            else:
                tello.move_forward(t_run_data)
        elif t_skill == 'move_back':
            if t_run_data < 20:
                # 用来回震荡来实现这个距离
                tello.move_back(20 + t_run_data)
                tello.move_forward(20)
            else:
                tello.move_back(t_run_data)
        elif t_skill == 'turn_left':
            tello.rotate_counter_clockwise(t_turn_data)
        elif t_skill == 'turn_right':
            tello.rotate_clockwise(t_turn_data)
        else:
            return False
    except:
        return False
    return True


def virtual_micro_adjust(tello, direction, step=20, offset=1, loop_times=3):
    """
    虚拟微调，通过前进+后退的震荡方式，模拟1cm级别的微调
    :param direction: 'forward', 'back', 'up', 'down'
    :param step: 每次移动距离（最小20cm）
    :param offset: 每次回退的偏移量，默认1cm
    :param loop_times: 震荡次数
    """
    for _ in range(loop_times):
        tello_run_skill(tello=tello, task_str=f'move_{direction} {step}')
        opposite_direction = 'back' if direction == 'forward' else 'forward'
        if direction == 'up':
            opposite_direction = 'down'
        elif direction == 'down':
            opposite_direction = 'up'
        tello_run_skill(tello=tello, task_str=f'move_{opposite_direction} {step - offset}')


def __get_llm_response(message_input: str, yolo_and_llm_base_url: str) -> str:
    """
    调用大语言模型获得结果
    :param message_input: 用户输入
    :param yolo_and_llm_base_url: 请求基础url地址
    :return: 大语言模型的结果
    """
    data = {"chatid": f"chatid_{random.randint(0, 65534):0>5}", "message": message_input}
    response = requests.post(yolo_and_llm_base_url + "/ai_chat_ollama/", json=data)
    # 打印返回的响应
    if response.status_code == 200 and response.json()['reply'][-1] == '.':
        return response.json()['reply']
    else:
        return "None"


with open('./dataset/train_prompt.json', 'r', encoding='utf-8') as file:
    train_prompt = json.loads(''.join(file.readlines()))


# 通过大语言模型解析
def use_llm_get_type_and_commend(user_input: str, yolo_and_llm_base_url: str) -> dict:
    """
    调用大语言模型获得关于用户指令的最终结果，可能有4中情况
    :param yolo_and_llm_base_url: 请求基础url地址
    :param user_input: 用户输入
    :return: 指令类型以及解析的指令 {"type": '寻找任务', "commend": resp_2}
    """
    problem_1_user_input = train_prompt['problem_1']['prompt'][0]
    problem_2_user_input = train_prompt['problem_2']['prompt'][0]
    problem_3_user_input = train_prompt['problem_3']['prompt'][0]
    problem_4_user_input = train_prompt['problem_4']['prompt'][0]

    t_1 = problem_1_user_input.replace('{}', user_input)
    resp_1 = __get_llm_response(message_input=t_1, yolo_and_llm_base_url=yolo_and_llm_base_url)
    if 'A.' in resp_1:
        # print('系统：', '当前指令为寻找任务')
        t_2 = problem_2_user_input.replace('{}', user_input)
        resp_2 = __get_llm_response(message_input=t_2, yolo_and_llm_base_url=yolo_and_llm_base_url)
        # print('系统：', f'需要寻找的物体为-> "{resp_2}"')
        return {"type": '寻找任务', "commend": resp_2}
    elif 'B.' in resp_1:
        # print('系统：', '当前指令为飞行控制指令任务')
        t_3 = problem_3_user_input.replace('{}', user_input)
        resp_3 = __get_llm_response(message_input=t_3, yolo_and_llm_base_url=yolo_and_llm_base_url)
        # print('系统：', f'飞行控制指令为-> "{resp_3}"')
        return {"type": '飞行控制指令任务', "commend": resp_3}
    elif 'C.' in resp_1:
        # print('系统：', '当前指令为程序控制任务')
        t_4 = problem_4_user_input.replace('{}', user_input)
        resp_4 = __get_llm_response(message_input=t_4, yolo_and_llm_base_url=yolo_and_llm_base_url)
        # print('系统：', f'程序控制任务为-> "{resp_4}"')
        return {"type": '程序控制任务', "commend": resp_4}

    # print('系统：', f'当前指令未能识别：{resp_1}')
    return {"type": '其他任务', "commend": "None."}


def __draw_boxes_with_labels(colored_depth, boxes, img_width, img_height):
    """
    在深度图上绘制带标签的检测框，并确保标签显示在框内或附近。

    :param colored_depth: 深度图（已经归一化和彩色化）
    :param boxes: 检测框列表，格式为 [(x1, y1, x2, y2, confidence, object_name), ...]
    :param img_width: 原始图像宽度
    :param img_height: 原始图像高度
    :return: 带检测框和标签的深度图
    """
    for box in boxes:
        x1, y1, x2, y2, confidence, object_name = box

        # 将框的坐标从原始图像尺寸映射到深度图尺寸
        x1 = int(x1 * (384 / img_width))
        y1 = int(y1 * (384 / img_height))
        x2 = int(x2 * (384 / img_width))
        y2 = int(y2 * (384 / img_height))

        # 绘制矩形框和标签
        cv2.rectangle(colored_depth, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
        label = f"{object_name} {confidence:.2f}"
        cv2.putText(colored_depth, label, (x1, int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return colored_depth


# YOLO 预测
def use_yolo_predict(img_path: str, yolo_and_llm_base_url: str) -> dict:
    with open(img_path, "rb") as img_file:
        files = {'file': img_file}  # 上传的文件
        # 发送 POST 请求
        yolo_predict_response = requests.post(yolo_and_llm_base_url + '/yolo_predict/', files=files)
    res = yolo_predict_response.json()

    # 保存并显示深度图
    depth_img_data = base64.b64decode(res["depth_img_encoded"])
    depth_img = cv2.imdecode(np.frombuffer(depth_img_data, np.uint8), cv2.IMREAD_COLOR)

    # 提取结果并转换为绘制格式
    img_width, img_height = 960, 720  # 假定深度图为 384x384
    box_data = []
    for r in res["yolo_result"]:
        box_data.append((r['x1'], r['y1'], r['x2'], r['y2'], r["confidence"], r["object_name"]))

    # 绘制检测框和标签到深度图上
    colored_depth_with_boxes = __draw_boxes_with_labels(
        colored_depth=depth_img,
        boxes=box_data,
        img_width=img_width,
        img_height=img_height
    )
    if not os.path.exists('./data'):
        os.mkdir('./data')
    save_predicted_img_path = os.path.join('./data', 'predicted_' + os.path.basename(img_path))
    cv2.imwrite(save_predicted_img_path, colored_depth_with_boxes)

    res = {'result': res['result'],
           'yolo_predicted_img_file_path': save_predicted_img_path,
           'save_predicted_depth_img_path': save_predicted_img_path,
           }
    return res


class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point=0.):
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.set_point = set_point  # 目标值
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        error = self.set_point - current_value
        current_time = time.time()
        dt = current_time - self.last_time

        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.last_error = error
        self.last_time = current_time

        return output


# 假设 depth_value 范围为 [0,1]，1表示远，0表示近
# 我们设定一个映射：当 depth=1 时，我们期望占比为 0.2（更大目标占比）
# 当 depth=0 时，我们期望占比为 0.1（原始值）
# 可根据实际需求调整这两个值
def depth_to_setpoint(depth_value, min_set_point=0.1, max_set_point=0.2):
    # 线性插值，如远处时加大set_point，近处时保持初始set_point
    # 可以根据自己的控制策略进行更复杂的映射
    # 当depth_value越接近1，set_point越接近 max_set_point
    # 当depth_value越接近0，set_point越接近 min_set_point
    return min_set_point + (max_set_point - min_set_point) * depth_value


def test():
    a = use_yolo_predict(img_path=r"./v/a.jpg", yolo_and_llm_base_url='http://127.0.0.1:4000')
    print(a)

    test_dataset = [
        "Okay, let's fly 50 centimeters forward first, then 50 centimeters to the right, and then turn left 45 degrees. That's it.",
        "Bring the drne to the riht by 200 centieters, then move foward a bit.",
        "起飞无人机，然后向前飞行1米，再向左转动90度，之后后退1米，然后再向右飞行50厘米，最后降落无人机",
        "无人机请飞到森林北边，看看是否有高温导致的潜在火情。",
        '先帮我找雪碧之后帮我找可乐',
        '清掉现在的控制和搜索任务',
        "请暂停当前的控制和搜索任务",
        "起飞无人机，然后向前飞行1米，之后向左飞行50厘米，再向左转动90度，之后后退1米，然后再向右飞行50厘米，最后向下飞行20厘米",
        "定位最近的伤员，投放急救包。",
        "Take off the drone, then slide right by 1 foot, after that rise up 1.5 meters, next reverse 120 centimeters, subsequently sway to the left by 2 meters, afterwards make it go down, following that shift right by 100 centimeters, and in the end land the aircraft."
    ]

    for i in test_dataset:
        t_res = use_llm_get_type_and_commend(user_input=i, yolo_and_llm_base_url='http://127.0.0.1:4000')
        print(i)
        print(t_res)


def test_2():
    tello_info = {'host': '192.168.137.136'}
    tello, frame_read = new_tello(tello_connect=tello_info, is_mock=False)

    save_num = 0
    while True:
        save_num += 1
        now_info = get_tello_info(tello=tello)
        current_height = now_info['current_height']
        battery = now_info['battery']
        save_path = tello_get_img(tello=tello, frame_read=frame_read, save_path='test_get_img')
        temp_str = (f"[battery:{battery}]-[tello_get_img:{save_num}]-"
                    f"[current_height:{current_height}]-[save_path:{save_path}]")
        print(f"\r{temp_str}", end="", flush=True)  # \r 回到行首，end="" 避免换行
        time.sleep(0.5)
    pass


if __name__ == "__main__":
    test()
