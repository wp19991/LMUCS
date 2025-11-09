import datetime
import queue
from queue import PriorityQueue
import threading
import time

from fun_tools import tello_run_skill, use_llm_get_type_and_commend, new_tello, PIDController, use_yolo_predict, \
    tello_get_img, get_tello_info, depth_to_setpoint

# 当前是否为模拟状态
is_mock = True
# 修改状态前用 with state_lock:
state_lock = threading.Lock()
# 是否退出所有线程
is_stop_thread = False

# 共享数据的线程锁
global_data_lock = threading.Lock()

# tello lock
tello_lock = threading.Lock()

# 全局变量
global_data = {
    # 'yolo_and_llm_base_url': 'http://172.16.34.117:4000',
    # 'yolo_and_llm_base_url': 'http://10.12.0.100:51041',
    # 'yolo_and_llm_base_url': 'http://172.16.34.133:4000',
    'yolo_and_llm_base_url': 'http://127.0.0.1:4000',
    'tello': {'host': '192.168.179.164',
              'udp_video_address': '',
              'battery': 70,
              'current_height': 99.7},
    'img': {'forward': './v/a.jpg',
            'yolo_predicted_save_path': './v/a.png',  # yolo识别后保存的文件路径
            'yolo_predicted_result': [],  # yolo识别的结果
            'save_predicted_depth_img_path': './v/a.png'
            },
    'log_str': [],
    'log_queue': queue.Queue(),  # 创建队列，用于存储日志消息
    'tello_searching_run_skill_history': [],  # 搜寻任务执行的命令
    # 优先级队列需要插入元祖 ： (1,str)  (2,str)
    # 数字越小优先级越高，可以是小数，负数
    # 跟随则添加在这个值+1
    'priority_value_normal': 100,
    # 紧急添加在这个值-1，这样就可以解决需要在搜寻任务的过程中提前运行某个指令
    'priority_value_emergency': 100,
    # 用于存储用户输入的自然语言指令
    'llm_query_queue': PriorityQueue(),
    # 执行状态 running pause
    'tello_command_status': 'running',
    # 用于传递无人机控制指令
    'tello_command_queue': PriorityQueue(),
    # 执行状态 running pause
    'yolo_search_status': 'running',
    # 当前搜索的物品名称
    'now_yolo_search_name': '',
    # yolo搜寻目标的队列，无人机会根据需要搜寻目标按顺序的搜寻
    'yolo_search_queue': PriorityQueue(),
    # 搜寻找不到物体需要用户输入
    'search_need_user_input': False,
    # 是否停止物品搜索
    # 当外界先直接停止所有任务和清空正在运行的任务的时候，可以通过这个结束正在进行搜索的物品
    'is_stop_search_object': False,
}


def add_log(log_str):
    global global_data
    with global_data_lock:
        global_data['log_str'].append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_str}")
        global_data['log_queue'].put(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_str}")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_str}")


# 更新无人机信息的线程
def update_drone_info_for_thread():
    global global_data
    while not is_stop_thread:
        tello_info = get_tello_info(tello=tello)
        with global_data_lock:
            global_data['tello']['current_height'] = tello_info['current_height']
            global_data['tello']['battery'] = tello_info['battery']

        # if not is_mock:
        file_save_path = tello_get_img(tello=tello, frame_read=frame_read)
        with global_data_lock:
            global_data['img']['forward'] = file_save_path

        # add_log(f'当前无人机状态: '
        #         f'高度:{global_data["tello"]["current_height"]}, 电池:{global_data["tello"]["battery"]}')
        time.sleep(0.2)


def get_img_and_predicted_immediately():
    global global_data

    # if not is_mock:
    file_save_path = tello_get_img(tello=tello, frame_read=frame_read)
    with global_data_lock:
        global_data['img']['forward'] = file_save_path

    res = use_yolo_predict(img_path=global_data['img']['forward'],
                           yolo_and_llm_base_url=global_data['yolo_and_llm_base_url'])
    with global_data_lock:
        global_data['img']['yolo_predicted_save_path'] = res['yolo_predicted_img_file_path']
        global_data['img']['yolo_predicted_result'] = res['result']
        global_data['img']['save_predicted_depth_img_path'] = res['save_predicted_depth_img_path']


def tello_control_thread():
    global global_data
    while True:
        time.sleep(0.1)
        if global_data['tello_command_status'] == 'running':
            if not global_data['tello_command_queue'].empty():
                with global_data_lock:
                    tello_command = global_data['tello_command_queue'].get()[1]
                with tello_lock:
                    t_res_run = tello_run_skill(tello=tello, task_str=tello_command)
                get_img_and_predicted_immediately()
                if not t_res_run:
                    add_log(f'[tello_control_thread] Flight control command execution failed: "{tello_command}"')
                else:
                    add_log(f'[tello_control_thread] Flight control command executed successfully: "{tello_command}"')


def adjust_drone_position_based_on_yolo(need_search_object_target):
    global global_data
    # 进行物体的搜索
    # 内部进行循环来进行搜索，当搜 索到物品或者当前搜索任务为空的时候返回，跳出循环
    # 当暂停的时候继续进行内部的循环

    # 当搜索不到的时候，此时队列为空
    # 要求用户输入
    # 当在视野中发现物体的时候自动接管
    is_search_ok = False
    need_box_area_ratio = 0.04

    # PID 控制器，分别用于 X 轴（水平）、Y 轴（垂直）和 Z 轴（前进后退）
    pid_x = PIDController(Kp=0.6, Ki=0.00, Kd=0.1, set_point=0.5)  # 目标 x 比例中心
    pid_y = PIDController(Kp=0.4, Ki=0.00, Kd=0.1, set_point=0.5)  # 目标 y 比例中心
    pid_z = PIDController(Kp=0.6, Ki=0.00, Kd=0.1, set_point=need_box_area_ratio)  # 目标占画面5%

    # 没有找到的话先转一圈，然后在请求用户输入
    dont_see_object_turn_around_count = 6

    def flash_object_search():
        get_img_and_predicted_immediately()
        result_obj = [{'name': need_search_object_target, 'confidence': 0.01,
                       'center_x_ratio': 0.6, 'center_y_ratio': 0.5,
                       'box_area_ratio': 0.1, 'avg_distance_value': 0.5}]
        # 每次循环可以从global_data获得最新的yolo识别结果
        for p in global_data['img']['yolo_predicted_result']:
            if p['name'].lower() in need_search_object_target.lower():
                result_obj.append(p.copy())
        result_obj.sort(key=lambda x: x['confidence'], reverse=True)
        object_search = result_obj[0]
        if float(object_search['confidence']) >= 0.8:
            with global_data_lock:
                global_data['search_need_user_input'] = False

        center_x_ratio = object_search['center_x_ratio']
        center_y_ratio = object_search['center_y_ratio']
        box_area_ratio = object_search['box_area_ratio']
        depth_value = object_search['avg_distance_value']
        add_log(f"[adjust_drone] Object detected, "
                f"current object x:{center_x_ratio}%, y:{center_y_ratio}%, "
                f"occupies {box_area_ratio}% of the area.")
        return object_search

    while not is_search_ok:
        # 每次pid调整之前，保证当前的图是最新的
        object_search = flash_object_search()

        if global_data['is_stop_search_object']:
            add_log('[adjust_drone] Current search task stopped, task completed.')
            return
        if global_data['yolo_search_status'] != 'running':
            # 当暂停的时候继续进行内部的循环，进行下一轮的等候
            time.sleep(1)
            continue

        if global_data['search_need_user_input']:
            # 等待用户输入的时候进行下一轮的等候
            time.sleep(1)
            dont_see_object_turn_around_count = 5
            continue

        if float(object_search['confidence']) >= 0.8:  # 置信度超过0.8
            add_log(f'[adjust_drone] Object detected: {need_search_object_target}, '
                    f'PID adjustment in progress.')
            # 找到了，将看不到转一圈的次数重置，等待下次看不到物体再进行转动一圈
            dont_see_object_turn_around_count = 5

            center_x_ratio = object_search['center_x_ratio']
            center_y_ratio = object_search['center_y_ratio']
            box_area_ratio = object_search['box_area_ratio']
            depth_value = object_search['avg_distance_value']

            # 判断物体框的面积是否超过视图面积的10%
            if box_area_ratio >= need_box_area_ratio:
                add_log(f"[adjust_drone] Target is approaching, hovering...")
                time.sleep(5)
                return

            # 使用PID控制器调整X、Y、Z方向
            # 在更新控制时动态调整pid_z的set_point
            pid_z.set_point = depth_to_setpoint(depth_value,
                                                min_set_point=need_box_area_ratio * 0.9,
                                                max_set_point=need_box_area_ratio)

            pid_x_output = pid_x.update(center_x_ratio)
            pid_y_output = pid_y.update(center_y_ratio)
            pid_z_output = pid_z.update(box_area_ratio)
            # 根据pid的输出动态调整移动距离
            degree_x = int(min(max(abs(pid_x_output * 90), 1), 60))  # 确保转动角度在0到90度之间
            distance_y = int(min(max(abs(pid_y_output * 100), 5), 100))  # 确保移动距离在1到60cm之间
            distance_z = int(min(max(abs(pid_z_output * 2000), 5), 100))  # 确保移动距离在1到60cm之间

            # 根据 PID 控制器的输出调整无人机的位置
            if pid_x_output > 0:
                add_log(f"[adjust_drone] PID control: Turn left {degree_x} degrees.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'turn_left {degree_x}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'turn_left {degree_x}')
            elif pid_x_output < 0:
                add_log(f"[adjust_drone] PID control: Turn right {degree_x} degrees.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'turn_right {degree_x}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'turn_right {degree_x}')

            object_search = flash_object_search()
            box_area_ratio = object_search['box_area_ratio']
            # 判断物体框的面积是否超过视图面积的10%
            if box_area_ratio >= need_box_area_ratio:
                add_log(f"[adjust_drone] Target is approaching, hovering...")
                time.sleep(5)
                return

            if pid_y_output > 0:
                add_log(f"[adjust_drone] PID control: up {distance_y} cm.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'move_up {distance_y}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'move_up {distance_y}')
            elif pid_y_output < 0:
                add_log(f"[adjust_drone] PID control: down {distance_y} cm.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'move_down {distance_y}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'move_down {distance_y}')

            object_search = flash_object_search()
            box_area_ratio = object_search['box_area_ratio']
            # 判断物体框的面积是否超过视图面积的10%
            if box_area_ratio >= need_box_area_ratio:
                add_log(f"[adjust_drone] Target is approaching, hovering...")
                time.sleep(5)
                return

            if pid_z_output > 0:
                add_log(f"[adjust_drone] PID control: Move forward {distance_z} cm.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'move_forward {distance_z}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'move_forward {distance_z}')
            elif pid_z_output < 0:
                add_log(f"[adjust_drone] PID control: Move backward {distance_z} cm.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'move_back {distance_z}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'move_back {distance_z}')

            object_search = flash_object_search()
            box_area_ratio = object_search['box_area_ratio']
            # 判断物体框的面积是否超过视图面积的10%
            if box_area_ratio >= need_box_area_ratio:
                add_log(f"[adjust_drone] Target is approaching, hovering...")
                time.sleep(5)
                return
        else:
            if dont_see_object_turn_around_count < 0:
                # 搜索不到的时候要求用户输入
                with global_data_lock:
                    global_data['search_need_user_input'] = True
            else:
                dont_see_object_turn_around_count -= 1
                add_log(f"[adjust_drone] Not detected: Turn right {60} degrees.")
                with tello_lock:
                    tello_run_skill(tello=tello, task_str=f'turn_right {60}')
                with global_data_lock:
                    global_data['tello_searching_run_skill_history'].append(f'turn_right {60}')
                time.sleep(2)


def search_thread():
    global global_data
    while True:
        time.sleep(0.1)
        if global_data['yolo_search_status'] == 'running':
            if not global_data['yolo_search_queue'].empty():
                with global_data_lock:
                    # 如果进行搜索了，那么就打开搜索
                    global_data['is_stop_search_object'] = False
                    yolo_search_object = global_data['yolo_search_queue'].get()[1]

                    if '面包' in yolo_search_object:
                        yolo_search_object = 'bread'
                    elif '可乐' in yolo_search_object:
                        yolo_search_object = 'cola'
                    elif '芬达' in yolo_search_object:
                        yolo_search_object = 'fanta'
                    elif '雪碧' in yolo_search_object:
                        yolo_search_object = 'sprite'
                    elif '蛋糕' in yolo_search_object:
                        yolo_search_object = 'cake'
                    elif '饼干' in yolo_search_object:
                        yolo_search_object = 'biscuit'
                    elif '感冒药' in yolo_search_object:
                        yolo_search_object = 'coldrex'
                    elif '止疼药' in yolo_search_object:
                        yolo_search_object = 'painkillers'
                    elif '碘' in yolo_search_object:
                        yolo_search_object = 'iodophor'

                    global_data['now_yolo_search_name'] = yolo_search_object
                add_log(f'[search_thread] The current searched item is: {yolo_search_object}')
                adjust_drone_position_based_on_yolo(need_search_object_target=yolo_search_object)
                with global_data_lock:
                    global_data['now_yolo_search_name'] = ''


def yolo_predicted_thread():
    global global_data
    while True:
        time.sleep(0.3)
        res = use_yolo_predict(img_path=global_data['img']['forward'],
                               yolo_and_llm_base_url=global_data['yolo_and_llm_base_url'])
        with global_data_lock:
            global_data['img']['yolo_predicted_save_path'] = res['yolo_predicted_img_file_path']
            global_data['img']['yolo_predicted_result'] = res['result']
            global_data['img']['save_predicted_depth_img_path'] = res['save_predicted_depth_img_path']
        # add_log(str(res['result']))
        # add_log(f"yolo_predicted_thread 当前场景识别到物体: "
        #         f"{', '.join([str(i['name']) + ':' + str(i['confidence']) for i in res['result']])}")


def llm_query_thread():
    global global_data
    while True:
        time.sleep(0.2)
        if not global_data['llm_query_queue'].empty():
            with global_data_lock:
                llm_query = global_data['llm_query_queue'].get()[1]
            res = use_llm_get_type_and_commend(user_input=llm_query,
                                               yolo_and_llm_base_url=global_data['yolo_and_llm_base_url'])
            add_log(f"[llm_query_thread] res: {str(res)}")

            with global_data_lock:
                if res['type'] == '寻找任务':
                    global_data['priority_value_normal'] += 1
                    for i in res['commend'].split('; '):
                        global_data['yolo_search_queue'].put((global_data['priority_value_normal'],
                                                              i.replace('.', '')))
                elif res['type'] == '飞行控制指令任务':
                    # 当有寻找任务的时候，这时候的指令具有高优先级
                    if not global_data['yolo_search_queue'].empty():
                        # 这时候应该清空当前执行的搜寻指令中飞行控制的列表
                        while not global_data['tello_command_queue'].empty():
                            global_data['tello_command_queue'].get()
                        # 向飞行控制的列表中添加紧急需要执行的命令
                        global_data['priority_value_emergency'] -= 1
                        for i in res['commend'].split('; '):
                            global_data['tello_command_queue'].put((global_data['priority_value_emergency'],
                                                                    i.replace('.', '')))
                        # 用户已经输入过了，搜索继续进行
                        global_data['search_need_user_input'] = False
                    else:
                        # 当前没有搜寻的任务的时候，就进行普通的添加
                        global_data['priority_value_normal'] += 1
                        for i in res['commend'].split('; '):
                            global_data['tello_command_queue'].put((global_data['priority_value_normal'],
                                                                    i.replace('.', '')))
                elif res['type'] == '程序控制任务':
                    # 对队列进行修改和判断
                    # 当用户使用语言进行暂停的时候，只能由语言来恢复
                    if 'pause_task' in res['commend']:
                        global_data['tello_command_status'] = 'pause'
                        global_data['yolo_search_status'] = 'pause'
                    elif 'pause_fly_task' in res['commend']:
                        global_data['tello_command_status'] = 'pause'
                    elif res['commend'] == 'pause_search_task':
                        global_data['yolo_search_status'] = 'pause'
                    elif 'start_task' in res['commend'] or 'continue_task' in res['commend']:
                        global_data['tello_command_status'] = 'running'
                        global_data['yolo_search_status'] = 'running'
                    elif 'start_fly_task' in res['commend'] or 'continue_fly_task' in res['commend']:
                        global_data['tello_command_status'] = 'running'
                    elif 'start_search_task' in res['commend'] or 'continue_search_task' in res['commend']:
                        global_data['yolo_search_status'] = 'running'
                    elif 'clear_task' in res['commend']:
                        while not global_data['tello_command_queue'].empty():
                            global_data['tello_command_queue'].get()
                        while not global_data['yolo_search_queue'].empty():
                            global_data['yolo_search_queue'].get()
                        global_data['tello_command_status'] = 'running'
                        global_data['yolo_search_status'] = 'running'
                        global_data['is_stop_search_object'] = True
                    elif 'clear_fly_task' in res['commend']:
                        while not global_data['tello_command_queue'].empty():
                            global_data['tello_command_queue'].get()
                        global_data['tello_command_status'] = 'running'
                    elif 'clear_search_task' in res['commend']:
                        while not global_data['yolo_search_queue'].empty():
                            global_data['yolo_search_queue'].get()
                        global_data['yolo_search_status'] = 'running'
                        global_data['is_stop_search_object'] = True


tello, frame_read = new_tello(tello_connect=global_data['tello'], is_mock=is_mock)

if __name__ == '__main__':
    # tello = new_tello(tello_connect=global_data['tello'], is_mock=True)

    # 无人机数据更新线程
    thread_1 = threading.Thread(target=update_drone_info_for_thread, daemon=True)
    thread_1.start()
    # yolo模型对图片进行预测线程
    thread_2 = threading.Thread(target=yolo_predicted_thread, daemon=True)
    thread_2.start()
    # 大语言模型解析线程
    thread_3 = threading.Thread(target=llm_query_thread, daemon=True)
    thread_3.start()
    # 无人机执行指令线程线程
    thread_4 = threading.Thread(target=tello_control_thread, daemon=True)
    thread_4.start()
    # 搜索物品线程线程
    thread_5 = threading.Thread(target=search_thread, daemon=True)
    thread_5.start()

    t_llm_data = [
        "找到目标建筑后搜索其周围的路标",
        "Bring the drne to the riht by 200 centieters, then move foward a bit.",
        "Proceed forward 6 inches, then fly it upward, and subsequently land the drone gently.",
        "起飞无人机，然后向前飞行1米，再向左转动90度，之后后退1米，然后再向右飞行50厘米，最后降落无人机",
        "无人机请飞到森林北边，看看是否有高温导致的潜在火情。",
        '找到桥梁的位置，然后检查桥下是否有船只通过。',
        '无人机检查废墟下的空隙，并寻找生命迹象。',
        "Go left 2 feet, then move back by 15 centimeters, afterwards descend 10 feet, and finally land the drone gently.",
        "起飞无人机，然后向前飞行1米，之后向左飞行50厘米，再向左转动90度，之后后退1米，然后再向右飞行50厘米，最后向下飞行20厘米",
        "定位最近的伤员，投放急救包。",
        "请暂停当前的飞行任务。",
        "启动飞行任务控制程序。",
        "Take off the drone, then slide right by 1 foot, after that rise up 1.5 meters, next reverse 120 centimeters, subsequently sway to the left by 2 meters, afterwards make it go down, following that shift right by 100 centimeters, and in the end land the aircraft."
    ]

    # t_llm_data = ['Search for the Sprite and Coca.']

    for o in t_llm_data:
        global_data['priority_value_normal'] += 1
        global_data['llm_query_queue'].put((global_data['priority_value_normal'], o))

    while True:
        time.sleep(1)
        add_log('time.sleep(1).')
