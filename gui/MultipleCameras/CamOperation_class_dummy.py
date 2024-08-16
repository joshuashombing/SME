# -- coding: utf-8 --
import threading
import time
from tkinter import *

import cv2
from PIL import Image, ImageTk

from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from config import DETECTOR_PARAMS_TOP_CAMERA, DETECTOR_PARAMS_SIDE_CAMERA
from engine import AIEngine


class CameraOperation:

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None,
                 b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False,
                 buf_save_image=None,
                 n_save_image_size=0, frame_rate=0, exposure_time=0, gain=0):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.h_thread_show = None
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

        self.ai_engine = AIEngine()
        self.latest_signal_change = time.time()

    # 转为16进制字符串
    def To_hex_str(self, num):
        return ""

    # 打开相机
    def Open_device(self):
        return 0

    # 开始取图
    def Start_grabbing(self, index, root, panel, lock, result_label):
        self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread,
                                                args=(self, index, root, panel, lock, result_label))
        self.h_thread_handle.start()

    # 停止取图
    def Stop_grabbing(self):
        self.ai_engine.stop()

    # 关闭相机
    def Close_device(self):
        return None

    # 设置触发模式
    def Set_trigger_mode(self, strMode):
        return None

    # 软触发一次
    def Trigger_once(self, nCommand):
        return None

    # 获取参数
    def Get_parameter(self):
        return None

    # 设置参数
    def Set_parameter(self, frameRate, exposureTime, gain):
        return None

    # 取图线程函数
    def Work_thread(self, index, root, panel, lock, result_label):
        params = DETECTOR_PARAMS_TOP_CAMERA if index == 1 else DETECTOR_PARAMS_SIDE_CAMERA
        detector = SpringMetalDetector(**params).build()

        video_paths = {
            0: "../../sample/berr/samping/Video_20240420182351175.avi",
            1: "../../sample/dented/atas/Video_20240420173419630.avi"
        }

        cap = cv2.VideoCapture(video_paths[index])

        print("Start grabbing frame...")
        while True:

            (grabbed, numArray) = cap.read()
            if not grabbed:
                break

            try:
                if numArray is not None:
                    result = detector.predict(numArray)
                    result = detector.track(result)
                    self.ai_engine.process(numArray, result)
                    numArray = self.ai_engine.draw_result(numArray, result)
                    frame = Image.fromarray(numArray)
                    lock.acquire()
                    imgtk = ImageTk.PhotoImage(image=frame, master=root)
                    panel.imgtk = imgtk
                    panel.config(image=imgtk)
                    root.obr = imgtk
                    lock.release()  # 释放锁
            except Exception:
                pass

    def show_inspection(self, result_label):
        if self.ai_engine.good_event.is_set():
            self.latest_signal_change = time.time()
            self.ai_engine.good_event.clear()
            # show good signal
        if self.ai_engine.defect_event.is_set():
            self.latest_signal_change = time.time()
            self.ai_engine.defect_event.clear()
            # show ng signal

        if (time.time() - self.latest_signal_change > 2
                and not self.ai_engine.good_event.is_set()
                and not self.ai_engine.defect_event.is_set()):
            # show waiting signal
            pass

    # 存jpg图像
    def Save_jpg(self, buf_cache):
        pass

    # 存BMP图像
    def Save_Bmp(self, buf_cache):
        pass

    # Mono图像转为python数组
    def Mono_numpy(self, data, nWidth, nHeight):
        return None

    # 彩色图像转为python数组
    def Color_numpy(self, data, nWidth, nHeight):
        return None
