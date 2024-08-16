# -- coding: utf-8 --
import ctypes
import inspect
import logging
import sys
import threading
import time
import tkinter.messagebox
from tkinter import *

import cv2
import numpy as np
from PIL import Image, ImageTk

from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.utils import get_now_str, resize_image
from config import DETECTOR_PARAMS_TOP_CAMERA, DETECTOR_PARAMS_SIDE_CAMERA, PREPROCESSOR_PARAMS_SIDE_CAMERA, \
    PREPROCESSOR_PARAMS_TOP_CAMERA, SAVE_VIDEO, RESULT_DIR
from engine_v2 import draw_result

logger = logging.getLogger("AutoInspection")

sys.path.append("../MvImport")
from MvCameraControl_class import *


# Forcefully close a thread
def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


# Stop thread
def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


class CameraOperation:
    _save_video_time = None

    def __init__(self, obj_cam, ai_engine, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None,
                 b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False,
                 buf_save_image=None,
                 n_save_image_size=0, frame_rate=0, exposure_time=0, gain=0, dummy=False):

        self.obj_cam = obj_cam
        self.ai_engine = ai_engine
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
        self.h_thread_show_handle = None
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

        self.video_writer = None

        self.dummy = dummy

        self._detector_params = {
            0: DETECTOR_PARAMS_TOP_CAMERA,
            1: DETECTOR_PARAMS_TOP_CAMERA
        }

        self._preprocessor_params = {
            0: PREPROCESSOR_PARAMS_TOP_CAMERA,
            1: PREPROCESSOR_PARAMS_TOP_CAMERA
        }

        self.latest_signal_change = time.perf_counter()
        self.prev_time_grabbing = time.perf_counter()

        self._frame_width = 482
        self._frame_height = None
        img = Image.fromarray(np.zeros((363, 482, 4), dtype=np.uint8), "RGBA")
        self.current_frame = ImageTk.PhotoImage(image=img)

        if CameraOperation._save_video_time is None:
            CameraOperation._save_video_time = get_now_str(microsecond=False)

        self.save_dir = RESULT_DIR / "videos" / CameraOperation._save_video_time

        if SAVE_VIDEO:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # @property
    # def frame_width(self):
    #     return self._frame_width
    #
    # @property
    # def frame_height(self):
    #     return self._frame_height

    def set_frame_size(self, w, h):
        self._frame_width = w
        self._frame_height = h

    # Convert to hexadecimal string
    def To_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""

        if num is None:
            return hexStr

        if num < 0:
            num = num + 2 ** 32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    # Turn on the camera
    def Open_device(self):
        if not self.b_open_device:
            # Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_closed = False
                return ret
            self.b_open_device = True
            self.b_thread_closed = False

            # Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        logger.info("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    logger.info("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                logger.info("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                logger.info("warning: set trigger mode off fail! ret[0x%x]" % ret)
            return 0

    # Start taking pictures
    def Start_grabbing(self, index, root, panel, lock, show_result_func):
        # self.show_result_func = show_result_func
        if self.dummy:
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread_video,
                                                        args=(self, index, root, panel, lock))
                self.h_thread_handle.start()

                if self.ai_engine is not None:
                    self.h_thread_show_handle = threading.Thread(
                        target=CameraOperation.Work_thread_show,
                        args=(self, index, show_result_func)
                    )
                    self.h_thread_show_handle.start()

                self.b_thread_closed = True

                # self.ai_engine.start_save_video(class_name="good")
            except:
                tkinter.messagebox.showerror('show error', 'error: unable to start thread')
                self.b_start_grabbing = False
            return 0

        if not self.b_start_grabbing and self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread,
                                                        args=(self, index, root, panel, lock))
                self.h_thread_handle.start()
                self.b_thread_closed = True

                if self.ai_engine is not None:
                    self.h_thread_show_handle = threading.Thread(
                        target=CameraOperation.Work_thread_show,
                        args=(self, index, show_result_func)
                    )
                    self.h_thread_show_handle.start()

            except:
                tkinter.messagebox.showerror('show error', 'error: unable to start thread')
                self.b_start_grabbing = False
            return ret

    # def stop_ai_engine(self):
    #     self.ai_engine.stop()
    #     for p in self.processes:
    #         p.join()

    # Stop taking pictures
    def Stop_grabbing(self):
        if self.b_start_grabbing and self.b_open_device:
            # Exit thread
            if self.b_thread_closed:
                self.b_exit = True
                # Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                self.b_start_grabbing = True
                return ret
            self.b_start_grabbing = False

            if self.video_writer is not None:
                self.video_writer.release()
                # self.video_writer = None

            return 0

    # Turn off camera
    def Close_device(self):
        if self.b_open_device:
            # Exit the thread
            if self.b_thread_closed:
                self.b_exit = True
                Stop_thread(self.h_thread_handle)
                if self.h_thread_show_handle is not None:
                    Stop_thread(self.h_thread_show_handle)
                self.b_thread_closed = False

            # self.stop_ai_engine()
            ret = self.obj_cam.MV_CC_StopGrabbing()
            ret = self.obj_cam.MV_CC_CloseDevice()
            return ret

        # Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False

    # Set trigger mode
    def Set_trigger_mode(self, strMode):
        if self.b_open_device:
            if "continuous" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
                if ret != 0:
                    return ret
                else:
                    return 0
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "On")
                if ret != 0:
                    return ret
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    return ret
                return ret

    # Soft trigger once
    def Trigger_once(self, nCommand):
        if self.b_open_device:
            if 1 == nCommand:
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                return ret

    # Get parameters
    def Get_parameter(self):
        if self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            self.gain = stFloatParam_gain.fCurValue
            return ret

    # Setting parameters
    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            return -1
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            return ret

    # Fetching thread function
    def Work_thread(self, index, root, panel, lock):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        img_buff = None

        buf_cache = None

        detector = SpringMetalDetector(**self._detector_params[index]).build()
        # pre_processor = DefectPreprocessor(**self._preprocessor_params[index])
        process_frame = self.ai_engine.process_frame if (self.ai_engine is not None) else (lambda t, f, r: None)
        video_path = str(self.save_dir / f"{self._detector_params[index]['camera_name']}.avi")

        logger.info(f"[Camera {index}] Start grabbing frame...")
        while True:
            if self.b_exit:
                if img_buff is not None:
                    del img_buff
                logger.info('now break')
                break

            timestamp = time.perf_counter()
            # self.ai_engine.push_object()
            # self.show_inspection_result(index, show_result_func)

            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 5000)
            if 0 == ret:
                if buf_cache is None:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                # logger.info("Camera[%d]:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                #     index, self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()
            else:
                logger.info("Camera[" + str(index) + "]:no data, ret = " + self.To_hex_str(ret))
                continue

            # 转换像素结构体赋值
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType

            # RGB direct display
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                # logger.info("PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType")
                numArray = CameraOperation.Color_numpy(self, buf_cache, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
            else:
                # logger.info("PixelType_Gvsp_RGB8_Packed !!!==== self.st_frame_info.enPixelType", PixelType_Gvsp_RGB8_Packed, self.st_frame_info.enPixelType)
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                ret = self.obj_cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
                if ret != 0:
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = CameraOperation.Color_numpy(self, img_buff, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)

            # Merge OpenCV into Tkinter interface
            if numArray is not None:
                # lock.acquire()
                if SAVE_VIDEO:
                    self.save_video(
                        cv2.cvtColor(numArray, cv2.COLOR_RGB2BGR),
                        (self.st_frame_info.nWidth, self.st_frame_info.nHeight),
                        video_path
                    )
                result = detector.predict(numArray)
                # if self.ai_engine is not None:
                #     self.ai_engine.process_frame(numArray, result, timestamp)
                process_frame(timestamp, numArray, result)
                frame = resize_image(numArray, width=self._frame_width, height=self._frame_height)
                frame = Image.fromarray(frame)
                try:
                    # lock.acquire()
                    self.current_frame = ImageTk.PhotoImage(image=frame)
                    panel.imgtk = self.current_frame
                    panel.config(image=self.current_frame)
                    root.obr = self.current_frame
                except Exception as e:
                    logger.error(f"Warning: {e}")
                # lock.release()  # 释放锁

            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)

        if self.ai_engine is not None:
            self.ai_engine.clear_result()
        logger.info(f"[Camera {index}] Stop grabbing frame...")

    def Work_thread_video(self, index, root, panel, lock):
        
        # video_paths = {
        #     0: r"C:\Users\joshua.christoper\Desktop\sme-automation-inspection-internal\sample\good\samping\Video_20240420190216570.avi",
        #     1: r"C:\Users\joshua.christoper\Desktop\sme-automation-inspection-internal\sample\good\atas\Video_20240420185538514.avi"
        # }
        video_paths = {
            0: r"C:\Users\maftuh.mashuri\Documents\DATA\datasets-130524\kamera-atas_berr_1715572446-06089.avi",
            1: r"C:\Users\maftuh.mashuri\Documents\DATA\datasets-130524\kamera-atas_good_1715572064-1064756.avi"
        }
        # video_paths = {
        #     0: "D:\\maftuh\\DATA\\datasets-130524\\kamera-samping_good_1715572063-9403038.avi",
        #     1: "D:\\maftuh\\DATA\\datasets-130524\\kamera-atas_good_1715572064-1064756.avi"
        # }
        # video_paths = {
        #     0: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_09-54-26_defect\kamera-samping.avi",
        #     1: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_09-54-26_defect\kamera-atas.avi"
        # }
        # video_paths = {
        #     0: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_10-03-00\kamera-samping.avi",
        #     1: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_10-03-00\kamera-atas.avi"
        # }
        # video_paths = {
        #     0: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_09-54-26_defect\kamera-samping.avi",
        #     1: r"C:\Users\maftuh.mashuri\Downloads\New folder\New folder\2024-07-09_09-54-26_defect\kamera-atas.avi"
        # }
        # video_paths = {
        #     0: "D:\\maftuh\\DATA\\datasets-130524\\kamera-samping_dented_1715571762-6297863.avi",
        #     1: "D:\\maftuh\\DATA\\datasets-130524\\kamera-atas_dented_1715571762-8547602.avi"
        # }
        #
        # video_paths = {
        #     0: "D:\\maftuh\\DATA\\2024-05-15_13-00\\kamera-samping_shift_1715755117-47417.avi",
        #     1: "D:\\maftuh\\DATA\\2024-05-15_13-00\\kamera-atas_shift_1715755117-5189724.avi"
        # }
        # video_paths = {
        #     0: "D:\\maftuh\\DATA\\2024-05-15_13-00\\kamera-samping_berr_1715754957-2755263.avi",
        #     1: "D:\\maftuh\\DATA\\2024-05-15_13-00\\kamera-atas_berr_1715754957-310977.avi"
        # }
        # video_paths = {
        #     0: "D:\\maftuh\\DATA\\datasets-130524\\kamera-samping_berr_1715572445-9136422.avi",
        #     1: "D:\\maftuh\\DATA\\datasets-130524\\kamera-atas_berr_1715572446-06089.avi"
        # }

        cap = cv2.VideoCapture(video_paths[index])

        detector = SpringMetalDetector(**self._detector_params[index]).build()
        # pre_processor = DefectPreprocessor(**self._preprocessor_params[index])
        process_frame = self.ai_engine.process_frame if (self.ai_engine is not None) else (lambda t, f, r: None)
        logger.info(f"[Camera {index}] Start grabbing frame...")
        while True:

            (grabbed, numArray) = cap.read()
            if not grabbed or self.b_exit:
                break

            # time.sleep(0.05)
            timestamp = time.perf_counter()

            # self.show_inspection_result(index, show_result_func)
            # try:
            if numArray is not None:
                # lock.acquire()
                numArray = cv2.cvtColor(numArray, cv2.COLOR_BGR2RGB)
                # print("camera id", index)
                # self.frame_queue.put((time.perf_counter(), numArray))
                result = detector.predict(numArray)
                process_frame(timestamp, numArray, result)

                frame = resize_image(numArray, width=self._frame_width, height=self._frame_height)
                frame = draw_result(frame, result["boxes_n"], result["track_ids"])
                frame = Image.fromarray(frame)
                try:
                    # lock.acquire()
                    # print(self.current_frame, panel, root)
                    self.current_frame = ImageTk.PhotoImage(image=frame)
                    panel.imgtk = self.current_frame
                    panel.config(image=self.current_frame)
                    root.obr = self.current_frame
                    # lock.release()  # 释放锁
                except Exception as e:
                    logger.error(f"Warning: {e}")
                # lock.release()
                # except Exception as e:
            #     logger.info("Error:", e)
        # self.stop_ai_engine()

        if self.ai_engine is not None:
            self.ai_engine.clear_result()

    def Work_thread_show(self, index, show_result_func):
        if self.ai_engine is None:
            return

        while True:
            if self.b_exit or self.ai_engine.is_stopped:
                break

            self.show_inspection_result(index, show_result_func)
            time.sleep(0.01)

            # try:
            #     frame = self.ai_engine.get_frame_show()
            #     if frame is None:
            #         continue
            #
            #     frame = Image.fromarray(frame)
            #     lock.acquire()
            #     self.current_frame = ImageTk.PhotoImage(image=frame)
            #     panel.imgtk = self.current_frame
            #     panel.config(image=self.current_frame)
            #     root.obr = self.current_frame
            #     lock.release()
            # except Exception as e:
            #     logger.error(f"Error at Work_thread_show: {e}")

    def show_inspection_result(self, camera_id, show_func):
        if self.ai_engine is None:
            return

        if self.ai_engine.good_event.is_set():
            self.latest_signal_change = time.perf_counter()
            self.ai_engine.good_event.clear()
            show_func(camera_id + 1, "empty")
            time.sleep(0.05)
            show_func(camera_id + 1, "good")

        if self.ai_engine.defect_event.is_set():
            self.latest_signal_change = time.perf_counter()
            self.ai_engine.defect_event.clear()
            show_func(camera_id + 1, "empty")
            time.sleep(0.05)
            show_func(camera_id + 1, "defect")

        if (time.perf_counter() - self.latest_signal_change > 2
                and not self.ai_engine.good_event.is_set()
                and not self.ai_engine.defect_event.is_set()):
            self.latest_signal_change = time.perf_counter()
            show_func(camera_id + 1, "empty")

    def save_video(self, frame, size, filename):
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

        self.video_writer.write(frame)

    # Mono image converted to python array
    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    # Convert color image to python array
    def Color_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray
