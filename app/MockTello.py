import time
import random

import cv2
import numpy as np


# 模拟返回一个含有视频帧的frame对象
class Frame:
    def __init__(self, frame):
        # 返回视频中的当前帧
        self.frame = frame

class MockTello:
    def __init__(self, host):
        self.host = host
        self.cap = cv2.VideoCapture(r"./shiyan1210.mp4")

        if not self.cap.isOpened():
            print("Error opening video file")
            return

        # 获取视频帧率和总帧数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"Video FPS: {self.fps}")
        print(f"Total Frames: {self.total_frames}")
        print(f"Video Duration: {self.duration:.2f} seconds")

        # 如果获取的帧率不合理，默认设为30
        if self.fps <= 0:
            self.fps = 30

        # 当前秒的位置
        self.current_time_sec = 0

    def connect(self):
        time.sleep(1)
        print("MockTello connected.")

    def query_sdk_version(self):
        time.sleep(0.5)
        return "v1.0"

    def get_battery(self):
        time.sleep(0.3)
        return random.uniform(80, 90)

    def get_udp_video_address(self):
        time.sleep(0.2)
        return "udp://0.0.0.0:11111"

    def streamon(self):
        time.sleep(0.2)
        print("MockTello video streaming on.")

    def get_frame_read(self):
        # 根据时间戳跳转到下一秒的帧
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time_sec * 440)
        # 读取帧
        ret, frame = self.cap.read()
        # print(frame.shape)
        if not ret:
            print(f"Cannot read frame at {self.current_time_sec:.2f} seconds")
            return None
        # 更新当前时间戳为下一秒
        self.current_time_sec += 1
        # 检查是否超出视频时长
        if self.current_time_sec >= self.duration:
            print("Reached end of video")
            return None
        return Frame(frame)

    def set_video_direction(self, direction):
        time.sleep(0.1)
        print(f"MockTello video direction set to {direction}.")

    def query_distance_tof(self):
        time.sleep(0.1)
        return random.uniform(90, 100) # 模拟当前高度

    def move_right(self, distance):
        time.sleep(5)
        print(f"MockTello move right by {distance} cm.")

    def move_left(self, distance):
        time.sleep(5)
        print(f"MockTello move left by {distance} cm.")

    def move_up(self, distance):
        time.sleep(5)
        print(f"MockTello move up by {distance} cm.")

    def move_down(self, distance):
        time.sleep(5)
        print(f"MockTello move down by {distance} cm.")

    def move_forward(self, distance):
        time.sleep(5)
        print(f"MockTello move forward by {distance} cm.")

    def move_back(self, distance):
        time.sleep(5)
        print(f"MockTello move back by {distance} cm.")

    def rotate_clockwise(self, angle):
        time.sleep(5)
        print(f"MockTello rotate clockwise by {angle} degrees.")

    def rotate_counter_clockwise(self, angle):
        time.sleep(5)
        print(f"MockTello rotate counterclockwise by {angle} degrees.")

    def takeoff(self):
        time.sleep(5)
        print("MockTello takeoff.")

    def land(self):
        time.sleep(5)
        print("MockTello land.")


def test_video_extraction():
    mock_tello = MockTello("192.168.10.1")

    if not mock_tello.cap.isOpened():
        print("Failed to open video file. Exiting.")
        return

    frame_count = 0

    while True:
        # 1. 获取 Frame 对象，我们改名叫 'frame_obj' 以免混淆
        frame_obj = mock_tello.get_frame_read()

        # 2. 检查对象本身是否为 None (视频结束或读取失败)
        if frame_obj is None:
            print("Video reading complete or error.")
            break

        # 3. 从 Frame 对象中提取真正的帧数据 (np.ndarray)
        #    这就是你需要的关键一步！
        actual_frame_data = frame_obj.frame
        print(actual_frame_data.shape)

        # 4. 现在，检查这个提取出的数据 (actual_frame_data)
        if isinstance(actual_frame_data, np.ndarray):
            # 5. 使用提取出的数据进行显示
            cv2.imshow('Test Frame', actual_frame_data)

            # 打印我们模拟的当前时间
            print(
                f"Frame {frame_count}: Displaying frame from video time {mock_tello.current_time_sec - 1:.2f} seconds")
            frame_count += 1

            # 每帧等待1000毫秒（1秒），方便查看
            if cv2.waitKey(440) & 0xFF == ord('q'):
                break
        else:
            # 如果 frame_obj.frame 不是 np.ndarray (例如 None 或其他)
            print("Invalid frame data encountered, skipping.")

    cv2.destroyAllWindows()
    if mock_tello.cap.isOpened():
        mock_tello.cap.release()



if __name__ == "__main__":
    # 执行测试
    test_video_extraction()
