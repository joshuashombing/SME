# -- coding: utf-8 --
import ctypes
import inspect
import queue
import sys
import threading
import time
import tkinter.messagebox
from collections import OrderedDict
from ctypes import *
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from tkinter import *
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from anomalib.models.detection.defect_detection_v1 import DefectPredictor
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.prepare_defect import create_anomalib_data
from anomalib.pre_processing.utils import resize_image, draw_bbox
from relay import Relay

sys.path.append("../MvImport")
from MvCameraControl_class import *


# 强制关闭线程
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


# 停止线程
def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


@dataclass
class TimeBundle:
    obj: Any
    timestamp: float = field(init=False)

    def __post_init__(self):
        self.timestamp = time.time()


relay = Relay()
relay.open()


# CONFIG_DEFECT_MODEL = {
#     "kamera-atas": dict(
#         path=,
#         distance_thresholds = (0.35, 0.5),
#         conf_threshold = 0.5,
#         output_patch_shape = (680, 560),
#         expand_bbox_percentage = 0.2,
#         device = "auto",
#         pre_processor = lambda x: resize_image(x, width=640),
#     ),
#     "kamera-samping": dict(
#         distance_thresholds=(0.35, 0.5),
#         conf_threshold=0.5,
#         output_patch_shape=(680, 320),
#         expand_bbox_percentage=0.2,
#         device="auto",
#         pre_processor=lambda x: resize_image(x, width=640),
#     )
# }


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

        self.buf_lock = threading.Lock()  # 取图和存图的buffer锁
        # self.frame_lock = threading.Lock()  # 取图和存图的buffer锁
        self.frame_queue = Queue()
        self.frame = None
        self.model_index = {
            0: "kamera-samping",
            1: "kamera-atas"
        }

        self.video_writer = None
        self.output_dir = Path("D:\\maftuh\\Projects\\SME\\anomalib\\datasets\\2024-05-15_13-00")

        self.text_result = ""
        self.color = (0, 255, 0)

        self.frame_queue = Queue()
        self.detection_queue = Queue()
        self.result_queue = Queue()
        self.grab_thread = None
        self.detect_thread = None
        self.inspection_thread = None

        self.prediction = OrderedDict()

    # 转为16进制字符串
    def To_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2 ** 32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    # 打开相机
    def Open_device(self):
        if self.b_open_device is False:
            # ch:选择设备并创建句柄 | en:Select device and create handle
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

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                print("warning: set trigger mode off fail! ret[0x%x]" % ret)
            return 0

    # 开始取图
    def Start_grabbing(self, index, root, panel, lock, result_label):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread,
                                                        args=(self, index, root, panel, lock, result_label))
                self.h_thread_handle.start()
                self.b_thread_closed = True

                # self.h_thread_show = threading.Thread(target=CameraOperation.thread_show,
                #                                         args=(self, root, panel))
                # self.h_thread_show.start()
            except:
                tkinter.messagebox.showerror('show error', 'error: unable to start thread')
                self.b_start_grabbing = False
            return ret

    # 停止取图
    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            # 退出线程
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
                self.video_writer = None

            return 0

    # 关闭相机
    def Close_device(self):
        if self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                self.b_exit = True
                Stop_thread(self.h_thread_handle)
                # Stop_thread(self.h_thread_show)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            ret = self.obj_cam.MV_CC_CloseDevice()
            return ret

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False

    # 设置触发模式
    def Set_trigger_mode(self, strMode):
        if True == self.b_open_device:
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

    # 软触发一次
    def Trigger_once(self, nCommand):
        if True == self.b_open_device:
            if 1 == nCommand:
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                return ret

    # 获取参数
    def Get_parameter(self):
        if True == self.b_open_device:
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

    # 设置参数
    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            return -1
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            return ret

    # 取图线程函数
    def Work_thread(self, index, root, panel, lock, result_label):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        img_buff = None

        buf_cache = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        save_image = False
        detector = {
            "kamera-atas": SpringMetalDetector(
                path="D:\\repos\\sme-automation-inspection-internal\\runs\\segment\\train\\weights\\best.pt",
                pre_processor=lambda x: resize_image(x, width=640),
                camera_name="kamera-atas",
                output_patch_shape=(680, 560),
                distance_thresholds=(0.4, 0.5)
            ),
            "kamera-samping": SpringMetalDetector(
                path="D:\\repos\\sme-automation-inspection-internal\\tools\\yolov8.pt",
                pre_processor=lambda x: resize_image(x, width=640),
                camera_name="kamera-samping",
                output_patch_shape=(680, 320),
                distance_thresholds=(0.4, 0.5)
            ),
        }[self.model_index[index]]

        detector.build()

        if not save_image:
            inspector = {
                "kamera-atas": DefectPredictor(
                    config_path="D:\\repos\\sme-automation-inspection-internal\\results\\patchcore\\mvtec\\spring_sheet_metal\\run.2024-05-15_15-29-01\\config.yaml",
                    root="D:\\repos\\sme-automation-inspection-internal",
                ),
                "kamera-samping": DefectPredictor(
                    config_path="D:\\repos\\sme-automation-inspection-internal\\results\\patchcore\\mvtec\\spring_sheet_metal\\run.2024-05-15_15-33-18\\config.yaml",
                    root="D:\\repos\\sme-automation-inspection-internal"
                )
            }[self.model_index[index]]
            inspector.build()

        # output_dir = Path("D:\\repos\\sme-automation-inspection-internal\\datasets") / self.model_index[index] / "spring_sheet_metal"
        class_name = "good"

        class_names = {
            0: "good",
            1: "defect"
        }
        color_map = {
            0: (0, 255, 0),
            1: (0, 0, 255)
        }

        while True:
            if self.b_exit:
                if img_buff is not None:
                    del img_buff
                print('now break')
                break
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 5000)
            if 0 == ret:
                if buf_cache is None:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                # print("Camera[%d]:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                #     index, self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()
            else:
                print("Camera[" + str(index) + "]:no data, ret = " + self.To_hex_str(ret))
                continue

            # 转换像素结构体赋值
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType

            # RGB直接显示
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                # print("PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType")
                numArray = CameraOperation.Color_numpy(self, buf_cache, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
            else:
                # print("PixelType_Gvsp_RGB8_Packed !!!==== self.st_frame_info.enPixelType", PixelType_Gvsp_RGB8_Packed, self.st_frame_info.enPixelType)
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

            # 合并OpenCV到Tkinter界面中
            if numArray is not None:
                # self.buf_lock.acquire()  # 加锁
                # numArray = cv2.cvtColor(numArray, cv2.COLOR_RGB2BGR)
                # if model is not None:
                inputArray = cv2.cvtColor(numArray, cv2.COLOR_RGB2BGR)
                result = detector.predict(inputArray)

                # for object_id, pt in detector.tracker.objects.items():
                #     cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                #     cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)

                # result.masks = None
                # result.names = {}
                # numArray = result
                # print(self.model_index[index], len(patches))
                # frame_queue.put(cv2.cvtColor(numArray, cv2.COLOR_RGB2BGR))
                # class_name = "shift"
                # self.save_video(
                #     inputArray,
                #     (self.st_frame_info.nWidth, self.st_frame_info.nHeight),
                #     str(self.output_dir / f"{self.model_index[index]}_{class_name}_{str(time.time()).replace('.', '-')}.avi")
                # )
                # result = model.detector.predict(inputArray, device=model.device)
                # numArray = result[0].plot()
                # numArray = cv2.cvtColor(numArray, cv2.COLOR_BGR2RGB)

                # numArray = cv2.cvtColor(numArray, cv2.COLOR_BGR2RGB)
                # numArray = cv2.resize(numArray, (500, 00))
                # self.frame = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
                inputArrayBGR = cv2.cvtColor(numArray.copy(), cv2.COLOR_RGB2BGR)
                # result = detector.predict(inputArrayBGR)
                # patches = detector.post_process(inputArrayBGR, result)
                # if save_image:
                #     for patch in patches:
                #         patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                #         create_anomalib_data(patch, output_dir, class_name=class_name, create_mask=class_name != "good")
                #
                if len(result) > 0:
                    try:
                        result = detector.track(result)
                        boxes = result.boxes.xyxyn
                        scores = [None for _ in range(len(boxes))]
                        labels = [None for _ in range(len(boxes))]
                        patches_map = detector.object_post_process(inputArray, result)
                        patches = [patch for _, patch in patches_map.values()]
                        result_defect = inspector.predict(patches)
                        if result_defect is not None:
                            scores = result_defect["pred_scores"].cpu().numpy()
                            labels = result_defect["pred_labels"].cpu().long().numpy()
                            print(self.model_index[index], list(patches_map.keys()), labels)

                            for i, object_id in enumerate(patches_map.keys()):
                                if object_id not in self.prediction:
                                    self.prediction[object_id] = []
                                self.prediction[object_id].append(labels[i])

                        all_object_ids = list(self.prediction.keys())
                        for object_id in all_object_ids:
                            predictions = self.prediction[object_id]
                            if index == 1:
                                miliseconds = "1000"
                                if len(predictions) >= 3:
                                    if np.sum(predictions) >= 2:
                                        relay.write(f"{str(index + 1)}-{miliseconds}=")
                                        print("pushing the object id", object_id, f"{str(index + 1)}-{miliseconds}=", "as defect")
                                        result_label.config(background="#CD202E", text='Defect')
                                    else:
                                        print("passing the object id", object_id, f"{str(index + 1)}-{miliseconds}=", "as good")
                                        result_label.config(background="#1DB74E", text='Good')  # Left good
                                    self.prediction.pop(object_id)
                            else:
                                miliseconds = "2500"
                                if predictions[0] == 1:
                                    relay.write(f"{str(index + 1)}-{miliseconds}=")
                                    print("pushing the object id", object_id, f"{str(index + 1)}-{miliseconds}=", "as defect")
                                    result_label.config(background="#CD202E", text='Defect')
                                else:
                                    print("passing the object id", object_id, f"{str(index + 1)}-{miliseconds}=", "as good")
                                    result_label.config(background="#1DB74E", text='Good')  # Left good
                                self.prediction.pop(object_id)

                        for box, object_id in zip(result.boxes.xyxyn, result.boxes.track_id):
                            draw_bbox(
                                inputArrayBGR,
                                box=box,
                                label=str(object_id),
                                score=None,
                                color=(255, 255, 255)
                            )
                        numArray = cv2.cvtColor(inputArrayBGR, cv2.COLOR_BGR2RGB)

                        # object_ids = set(detector.patches.keys()).copy()
                        # for object_id in object_ids:
                        #     all_exists = [patch is not None for patch in detector.patches[object_id]]
                        #     if all(all_exists):
                        #         patches = detector.patches.pop(object_id)
                        #         if len(patches) > 0 and not save_image:
                        #             result_defect = inspector.predict(patches)
                        #             if result_defect is not None:
                        #                 scores = result_defect["pred_scores"].cpu().numpy()
                        #                 labels = result_defect["pred_labels"].cpu().long().numpy()
                        #                 print("".center(50, "="))
                        #                 print("Camera", self.model_index[index])
                        #                 print("Object Id", object_id)
                        #                 print("Score", scores)
                        #                 print("Labels", labels)
                        #                 print("".center(50, "="))
                        #
                        #                 if len(labels) == 3:
                        #                     if np.sum(labels) >= 2:
                        #                         relay.write(str(index + 1))
                        #                         print("pushing the object", str(index + 1))
                        #                         self.text_result = "Defect"
                        #                         self.color = (0, 0, 255)
                        #                         # inputArrayBGR = cv2.putText(inputArrayBGR, 'Defect', (20, 20), cv2.FONT_HERSHEY_SIMPLEX ,
                        #                         #                     2, (0, 0, 255), 1, cv2.LINE_AA)
                        #                         # text_result.config(text="Updated Text")
                        #
                        #                 else:
                        #                     if labels[0] == 1:
                        #                         relay.write(str(index + 1))
                        #                         print("pushing the object", str(index + 1))
                        #                         self.text_result = "Defect"
                        #                         self.color = (0, 0, 255)
                        # inputArrayBGR = cv2.putText(inputArrayBGR, 'Defect', (20, 20), cv2.FONT_HERSHEY_SIMPLEX ,
                        #                     2, (0, 0, 255), 1, cv2.LINE_AA)

                        # inputArrayBGR = cv2.putText(inputArrayBGR, self.text_result, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        #                             2, self.color, 1, cv2.LINE_AA)

                    except Exception as e:
                        # result_label.config(background="#1DB74E", text='Good')  # Left good
                        print(e)

                    # result_label.config(background="#1DB74E", text='Good')  # Left good


                frame = Image.fromarray(numArray).resize((500, 500), Image.Resampling.LANCZOS)
                lock.acquire()
                # print(frame.size)
                # self.frame_queue.put(frame)
                imgtk = ImageTk.PhotoImage(image=frame, master=root)
                panel.imgtk = imgtk
                panel.config(image=imgtk)
                root.obr = imgtk
                lock.release()  # 释放锁

            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)

    def inspect_object(self, index):
        model = {
            "kamera-atas": DefectPredictor(
                config_path="D:\\repos\\sme-automation-inspection-internal\\results\\patchcore\\mvtec\\spring_sheet_metal\\run.2024-05-15_15-29-01\\config.yaml",
                root="D:\\repos\\sme-automation-inspection-internal",
            ),
            "kamera-samping": DefectPredictor(
                config_path="D:\\repos\\sme-automation-inspection-internal\\results\\patchcore\\mvtec\\spring_sheet_metal\\run.2024-05-15_15-33-18\\config.yaml",
                root="D:\\repos\\sme-automation-inspection-internal"
            )
        }[self.model_index[index]]
        model.build()

        while True:
            if self.b_exit:
                break

            try:
                detections_bundle = self.detection_queue.get(timeout=1)
            except queue.Empty:
                pass
            else:
                (result_bundle, patches_bundle) = detections_bundle
                patches = patches_bundle.obj
                patches = [patch for _, patch in patches.values()]
                result = model.predict(patches)

                if result is not None:
                    scores = result["pred_scores"].cpu().numpy()
                    labels = result["pred_labels"].cpu().long().numpy()
                    print(self.model_index[index], list(patches_bundle.obj.keys()), scores, labels)

                del detections_bundle

        del model

    def detect_object(self, index):
        detector = {
            "kamera-atas": SpringMetalDetector(
                path="D:\\repos\\sme-automation-inspection-internal\\runs\\segment\\train\\weights\\best.pt",
                pre_processor=lambda x: resize_image(x, width=640),
                camera_name="kamera-atas",
                output_patch_shape=(680, 560),
                distance_thresholds=(0.4, 0.5)
            ),
            "kamera-samping": SpringMetalDetector(
                path="D:\\repos\\sme-automation-inspection-internal\\tools\\yolov8.pt",
                pre_processor=lambda x: resize_image(x, width=640),
                camera_name="kamera-samping",
                output_patch_shape=(680, 320),
                distance_thresholds=(0.4, 0.5)
            ),
        }[self.model_index[index]]

        detector.build()

        while True:
            if self.b_exit:
                break

            try:
                frame_bundle = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                frame = frame_bundle.obj
                result = detector.predict(frame)
                result_bundle = TimeBundle(result)
                result = detector.track(result)
                patches = detector.object_post_process(frame, result)
                patches_bundle = TimeBundle(patches)
                self.detection_queue.put((result_bundle, patches_bundle))
                self.result_queue.put(result)
                del frame_bundle

        del detector

    def thread_show(self, root, panel):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            imgtk = ImageTk.PhotoImage(image=frame, master=root)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            root.obr = imgtk

    def save_video(self, frame, size, filename):
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

        self.video_writer.write(frame)

    def get_frame(self):
        return self.frame_queue.get()

    # 存jpg图像
    def Save_jpg(self, buf_cache):
        if (None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX3()
        stParam.enImageType = MV_Image_Jpeg  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx3(stParam)

        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            tkinter.messagebox.showinfo('show info', 'save jpg success!')
        except Exception as e:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % e)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    # 存BMP图像
    def Save_Bmp(self, buf_cache):
        if (0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX3()
        stParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx3(stParam)
        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save bmp fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tkinter.messagebox.showinfo('show info', 'save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    # Mono图像转为python数组
    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    # 彩色图像转为python数组
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
