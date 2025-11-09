import os
import sys
import queue
import datetime
import threading

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow

from ui.main_window import Ui_MainWindow as main_window
from fly import global_data, update_drone_info_for_thread, yolo_predicted_thread, llm_query_thread, \
    tello_control_thread, search_thread, global_data_lock, add_log, is_mock

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class main_win(QMainWindow, main_window):
    def __init__(self):
        super(main_win, self).__init__()
        self.setupUi(self)

        self.pushButton_send.clicked.connect(self.send_pushButton_event)
        self.pushButton_clear_llm_query_queue.clicked.connect(self.clear_llm_query_queue_pushButton_event)
        self.pushButton_clear_tello_command_queue.clicked.connect(self.clear_tello_command_queue_pushButton_event)
        self.pushButton_clear_yolo_search_queue.clicked.connect(self.clear_search_queue_pushButton_event)


        self.timer_for_update_listWidget_queue = QtCore.QTimer(self)
        self.timer_for_update_listWidget_queue.timeout.connect(self.update_list_widget_queue)  # 超时后调用的函数
        self.timer_for_update_listWidget_queue.start(100)  # 每 0.1 秒触发一次

        self.timer_for_update_img = QtCore.QTimer(self)
        self.timer_for_update_img.timeout.connect(self.update_img)  # 超时后调用的函数
        self.timer_for_update_img.start(100)  # 每 0.1 秒触发一次

        self.timer_for_update_time_and_log = QtCore.QTimer(self)
        self.timer_for_update_time_and_log.timeout.connect(self.update_time_and_log)  # 超时后调用的函数
        self.timer_for_update_time_and_log.start(100)  # 每 0.1 秒触发一次

    def clear_llm_query_queue_pushButton_event(self):
        with global_data_lock:
            while not global_data['llm_query_queue'].empty():
                global_data['llm_query_queue'].get()

    def clear_tello_command_queue_pushButton_event(self):
        with global_data_lock:
            while not global_data['tello_command_queue'].empty():
                global_data['tello_command_queue'].get()
            global_data['tello_command_status'] = 'running'

    def clear_search_queue_pushButton_event(self):
        with global_data_lock:
            while not global_data['yolo_search_queue'].empty():
                global_data['yolo_search_queue'].get()
            global_data['yolo_search_status'] = 'running'
            global_data['is_stop_search_object'] = True

    def send_pushButton_event(self):
        text = self.textEdit.toPlainText()
        add_log(f"[app] llm_query_queue.put({text})")
        with global_data_lock:
            global_data['priority_value_normal'] += 1
            global_data['llm_query_queue'].put((global_data['priority_value_normal'], text))

    def update_list_widget_queue(self):
        if not is_mock:
            items1 = sorted(list(global_data['llm_query_queue'].queue))
            self.listWidget_llm_query_queue.clear()
            for item in items1:
                self.listWidget_llm_query_queue.addItem(item[1])

            items2 = sorted(list(global_data['tello_command_queue'].queue))
            self.listWidget_tello_command_queue.clear()
            for item in items2:
                self.listWidget_tello_command_queue.addItem(item[1])

            items3 = sorted(list(global_data['yolo_search_queue'].queue))
            self.listWidget_yolo_search_queue.clear()
            for item in items3:
                self.listWidget_yolo_search_queue.addItem(item[1])

        items4 = sorted(global_data['img']['yolo_predicted_result'], key=lambda x: x['avg_distance_value'])
        self.listWidget_yolo_predcited_items.clear()
        self.listWidget_yolo_predcited_items.addItem("   name     conf  dis   box  ")
        for item in items4:
            self.listWidget_yolo_predcited_items.addItem(f"{item['name']: ^11} "
                                                         f"{item['confidence']: <5} "
                                                         f"{item['avg_distance_value']: <5} "
                                                         f"{item['box_area_ratio']: <5}")

    def update_img(self):
        pixmap = QtGui.QPixmap(global_data['img']['forward'])
        self.label_img.setPixmap(pixmap)
        pixmap_predicted = QtGui.QPixmap(global_data['img']['save_predicted_depth_img_path'])
        self.label_img_predicted.setPixmap(pixmap_predicted)

    def update_time_and_log(self):
        current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.label_nowtime.setText(current_time)
        tello_battery = global_data['tello']['battery']
        self.label_tello_battery.setText(f"{tello_battery}%")
        tello_current_height = global_data['tello']['current_height']
        self.label_tello_current_height.setText(f"{tello_current_height}cm")
        if not is_mock:
            now_yolo_search_name = global_data['now_yolo_search_name']
            self.label_now_search_name.setText(f"Searching for: {now_yolo_search_name}")
        else:
            self.label_now_search_name.setText(f"Searching for: bread")

        if global_data['search_need_user_input']:
            self.label.setStyleSheet("color: red;")
            self.pushButton_send.setStyleSheet("background-color: orange;")
        else:
            self.label.setStyleSheet("color: black;")
            self.pushButton_send.setStyleSheet('')

        # 更新日志信息
        if not global_data['log_queue'].empty() and not is_mock:
            try:
                while not global_data['log_queue'].empty():
                    log = global_data['log_queue'].get_nowait()  # 从队列获取日志
                    log = log.replace('寻找任务', 'Search Task')
                    log = log.replace('飞行控制指令任务', 'Flight Control Command Task')
                    log = log.replace('程序控制任务', 'Program Control Task')
                    self.textBrowser.append(log)
            except queue.Empty:
                pass


class App(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        self.main_windows = main_win()
        if not is_mock:
            self.main_windows.textBrowser.clear()

    def run(self):
        self.main_windows.show()
        sys.exit(self.exec_())


if __name__ == "__main__":
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

    # t_llm_data = ['Search for the Sprite and Coca.']
    '''
    t_llm_data = [
        "向前飞行50厘米，之后向右飞行50厘米，最后右转50度",
        "向前飞行80厘米，之后向左飞行80厘米，最后左转80度",
        "请暂停当前的飞行任务。",
        # "启动飞行任务控制程序",
        # "搜索急救包送到伤员的地方",
    ]

    for o_t_llm_data in t_llm_data:
        global_data['priority_value_normal'] += 1
        global_data['llm_query_queue'].put((global_data['priority_value_normal'], o_t_llm_data))
    '''

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    App().run()
