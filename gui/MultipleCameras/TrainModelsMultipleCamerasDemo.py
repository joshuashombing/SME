# -- coding: utf-8 --
import sys
import time
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk

sys.path.append("../MvImport")
# from MvCameraControl_class import *
from CamOperation_class_v1 import *
from PIL import Image, ImageTk
import cv2
from customtkinter import CTkLabel, CTkButton, CTkEntry, CTkRadioButton

from engine_v2 import AIEngine
from config import DEBUG


class GUI:

    def __init__(self, ai_engines):
        self.ai_engines = ai_engines
        self.window = tk.Tk()
        self.window_title = 'SME Spring Sheet Metal Inspection | Train Models Mode'
        self.window.iconbitmap("satnusa.ico")
        self.window.title(self.window_title)
        self.window_w = 690
        self.window_h = 1062
        self.window.geometry(f'{self.window_h}x{self.window_w}')
        self.window.resizable(False, False)  # Fixed sized

        # Handling event closing
        self.window.protocol("WM_DELETE_WINDOW", self.event_closing)

        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.obj_cam_operation = []
        self.b_is_run = False
        self.nOpenDevSuccess = 0
        self.devList = None
        self.triggercheck_val = tk.IntVar()
        self.model_val = tk.StringVar()
        self.page = Frame(self.window, height=0, width=0)
        self.page.pack(expand=True, fill=BOTH)

        self.default_exposure_time = 2000
        self.default_gain = 19.9963
        self.default_frame_rate = 9.5000

        # Header
        self.header_line = tk.Label(self.window, width=145, bg="#004040")
        self.header_line.place(x=21, y=25)
        self.header_label_thinner = tk.Label(self.window, width=145)
        self.header_label_thinner.place(x=21, y=23)
        self.header_label = tk.Label(self.window, text=self.window_title, font=("Roboto", 21), fg="#333")
        self.header_label.place(x=16, y=1)

        self.initialization_camera_label = tk.Label(self.window, font=("Roboto", 10), text='Initialization Camera',
                                                    fg="#333")
        self.initialization_camera_label.place(x=18, y=64)

        self.colect_img_data_activity_label = tk.Label(self.window, font=("Roboto", 10), text='Collect Images Data Activity',
                                                  fg="#333")
        self.colect_img_data_activity_label.place(x=308, y=64)

        self.label_total_devices = tk.Label(self.window, font=("Roboto", 8), text='Number of camera\t\t: None',
                                            fg="#004040")
        self.label_total_devices.place(x=18, y=90)

        self.button_style = ttk.Style()
        self.button_style.configure('TButton', foreground="#333")

        self.btn_enum_devices = CTkButton(master=self.window,
                                          width=120,
                                          height=30,
                                          border_width=0,
                                          corner_radius=4,
                                          text="Check Devices",
                                          fg_color="#c3c3c3",
                                          text_color="black",
                                          font=("Roboto", 12),
                                          command=self.enum_devices)
        self.btn_enum_devices.place(x=20, y=120)

        self.open_close_button_is_clicked = False
        self.open_close_button_style = ttk.Style()
        self.open_close_button_style.configure('OpenClose.TButton')
        self.btn_open_close_device_is_open = True
        self.btn_open_close_device = CTkButton(master=self.window,
                                               width=120,
                                               height=30,
                                               border_width=0,
                                               corner_radius=4,
                                               text="Open Device",
                                               fg_color="#1a8a42",
                                               text_color="white",
                                               font=("Roboto", 12),
                                               command=self.open_close_device)
        self.btn_open_close_device.place(x=151, y=120)

        self.radio_btn_good_or_defect_variable = tkinter.IntVar(value=0)

        self.radio_btn_good_or_defect_1 = CTkRadioButton(master=self.window, text="Good Images Data", fg_color="#158237", text_color="#004040",
                                                         variable=self.radio_btn_good_or_defect_variable, value=1)
        self.radio_btn_good_or_defect_2 = CTkRadioButton(master=self.window, text="Defect Images Data", fg_color="#BB1D2A", text_color="#004040",
                                                         variable=self.radio_btn_good_or_defect_variable, value=2)

        self.radio_btn_good_or_defect_1.place(x=314, y=95)
        self.radio_btn_good_or_defect_2.place(x=314, y=125)

        self.radio_btn_top_or_side_variable = tkinter.IntVar(value=0)

        self.radio_btn_top_or_side_1 = CTkRadioButton(master=self.window, text="Top Camera", fg_color="#004040", text_color="#004040",
                                                         variable=self.radio_btn_top_or_side_variable, value=1)
        self.radio_btn_top_or_side_2 = CTkRadioButton(master=self.window, text="Side Camera", fg_color="#004040", text_color="#004040",
                                                         variable=self.radio_btn_top_or_side_variable, value=2)

        self.radio_btn_top_or_side_1.place(x=474, y=95)
        self.radio_btn_top_or_side_2.place(x=474, y=125)

        self.label_status = tk.Label(self.window, font=("Roboto", 8), text='Collect Images Status:', fg="#004040")
        self.label_status.place(x=599, y=90)

        self.collect_data_status = self.collect_data_status_label(text="STOP ", font=("Roboto", 10),
                                                              corner_radius=4, fg_color="#CD202E",
                                                              text_color="white", height=20,
                                                              x=724, y=90)

        self.btn_start_training = CTkButton(master=self.window,
                                                 width=120,
                                                 height=30,
                                                 border_width=0,
                                                 corner_radius=4,
                                                 text="Start Train Models",
                                                 fg_color="#c3c3c3",
                                                 text_color="black",
                                                 font=("Roboto", 12),
                                                 command=self.start_training)
        self.btn_start_training.place(x=684, y=120)

        self.btn_start_stop_grabbing_style = ttk.Style()
        self.btn_start_stop_grabbing_style.configure('OpenClose.TButton')
        self.btn_start_stop_grabbing_is_grabbing = False
        self.btn_start_stop_grabbing = CTkButton(master=self.window,
                                                 width=160,
                                                 height=30,
                                                 border_width=0,
                                                 corner_radius=4,
                                                 text="Start Collect Images Data",
                                                 fg_color="#1a8a42",
                                                 text_color="white",
                                                 font=("Roboto", 12),
                                                 command=self.start_stop_grabbing)
        self.btn_start_stop_grabbing.place(x=601, y=120)

        self.train_activity_label = tk.Label(self.window, font=("Roboto", 10), text='Select Camera',
                                                      fg="#333")

        self.train_activity_label.place(x=465, y=64)

        # Create two bordered frames
        self.frame1 = self.create_bordered_frame(21, 188, "#A0A0A0", 1, 500, 470)
        self.frame2 = self.create_bordered_frame(541, 188, "#A0A0A0", 1, 500, 470)

        self.img_no_camera = tk.PhotoImage(file="no_image.png").subsample(4,4)  # Menyesuaikan ukuran gambar menjadi 50x50
        self.no_camera_label1 = tk.Label(self.frame1, image=self.img_no_camera)
        self.no_camera_label1.place(x=115, y=130)

        self.no_camera_label2 = tk.Label(self.frame2, image=self.img_no_camera)
        self.no_camera_label2.place(x=115, y=130)

        self.camera_status = [tk.Label(self.frame1, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold")),
                              tk.Label(self.frame2, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold"))]

        self.camera_status[0].place(x=160, y=380)
        self.camera_status[1].place(x=160, y=380)

        self.camera_label_side = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera Side Train Models',
                                          fg="#333")
        self.camera_label_side.place(x=30, y=175)
        self.camera_label_top = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera Top Train Models',
                                         fg="#333")
        self.camera_label_top.place(x=551, y=175)

        # Place the labels inside the bordered frames
        self.label_current_exposure_value = [tk.Label(self.frame1,
                                                      text="Current Exposure Value\t: None",
                                                      font=("Roboto", 8), fg="#004040"),
                                             tk.Label(self.frame2,
                                                      text="Current Exposure Value\t: None",
                                                      font=("Roboto", 8), fg="#004040")]
        self.label_current_exposure_value[0].place(x=10, y=10)

        self.label_current_exposure_value[1].place(x=10, y=10)

        self.label_exposure_time_cam1 = tk.Label(self.frame1, text='Exposure Time Camera\t\t:', font=("Roboto", 8),
                                                 fg="#004040")
        self.label_exposure_time_cam1.place(x=10, y=30)
        self.label_exposure_time_cam2 = tk.Label(self.frame2, text='Exposure Time Camera\t\t:', font=("Roboto", 8),
                                                 fg="#004040")
        self.label_exposure_time_cam2.place(x=10, y=30)

        self.text_exposure_time = [CTkEntry(master=self.frame1,
                                            placeholder_text="Insert Value",
                                            width=110,
                                            height=20,
                                            border_width=1,
                                            corner_radius=4,
                                            font=("Roboto", 11), ),
                                   CTkEntry(master=self.frame2,
                                            placeholder_text="Insert Value",
                                            width=110,
                                            height=20,
                                            border_width=1,
                                            corner_radius=4,
                                            font=("Roboto", 11), )]

        self.text_exposure_time[0].place(x=160, y=30)
        self.text_exposure_time[1].place(x=160, y=30)

        self.btn_set_param_cam1 = CTkButton(master=self.frame1,
                                            width=80,
                                            height=20,
                                            border_width=0,
                                            corner_radius=4,
                                            text="Save Value",
                                            fg_color="#004040",
                                            text_color="white",
                                            font=("Roboto", 12),
                                            command=lambda: self.set_parameter_new(0))
        self.btn_set_param_cam1.place(x=275, y=30)

        self.btn_set_param_cam2 = CTkButton(master=self.frame2,
                                            width=80,
                                            height=20,
                                            border_width=0,
                                            corner_radius=4,
                                            text="Save Value",
                                            fg_color="#004040",
                                            text_color="white",
                                            font=("Roboto", 12),
                                            command=lambda: self.set_parameter_new(1))
        self.btn_set_param_cam2.place(x=275, y=30)

        self.result_pos_bg = ((8, 60), (8, 60))
        self.result_pos_text = ((210, 65), (230, 65))

        # self.framex = -250
        # self.framey = 100
        # self.panel = Label(self.page)
        # self.panel.place(x=300 + self.framex, y=10 + self.framey, height=500, width=500)
        #
        # self.panel1 = Label(self.page)
        # self.panel1.place(x=810 + self.framex, y=10 + self.framey, height=500, width=500)
        #
        # self.panel2 = Label(self.page)
        # self.panel2.place(x=300 + self.framex, y=520 + self.framey, height=500, width=500)
        #
        # self.panel3 = Label(self.page)
        # self.panel3.place(x=810 + self.framex, y=520 + self.framey, height=500, width=500)

        self.radio_continuous = tk.Radiobutton(self.window, text='Continuous', variable=self.model_val,
                                               value='continuous', width=15,
                                               height=1, command=self.set_triggermode)
        # self.radio_continuous.place(x=20, y=400)
        self.radio_trigger = tk.Radiobutton(self.window, text='Trigger Mode', variable=self.model_val,
                                            value='triggermode', width=15,
                                            height=1, command=self.set_triggermode)
        # self.radio_trigger.place(x=160, y=400)
        self.model_val.set(1)

        self.checkbtn_trigger_software = tk.Checkbutton(self.window, text='Tigger by Software',
                                                        variable=self.triggercheck_val, onvalue=1,
                                                        offvalue=0)
        # self.checkbtn_trigger_software.place(x=20, y=450)

        self.btn_trigger_once = ttk.Button(self.window, text='Trigger Once', width=15, command=self.trigger_once)
        # self.btn_trigger_once.place(x=160, y=450)

        self.label_gain = tk.Label(self.window, text='Gain', width=15, height=1)
        # self.label_gain.place(x=20, y=250)
        self.text_gain = tk.Text(self.window, width=10, height=1)
        # self.text_gain.place(x=160, y=250)

        self.label_frame_rate = tk.Label(self.window, text='Frame Rate', width=15, height=1)
        # self.label_frame_rate.place(x=20, y=290)
        self.text_frame_rate = tk.Text(self.window, width=10, height=1)
        # self.text_frame_rate.place(x=160, y=290)

        self.btn_get_parameter = ttk.Button(self.window, text='Get Parameter', width=15, command=self.get_parameter)
        # self.btn_get_parameter.place(x=20, y=330)
        self.btn_set_parameter = ttk.Button(self.window, text='Set Parameter', width=15, command=self.set_parameter)
        # self.btn_set_parameter.place(x=90, y=330)

        # self.label_image1 = tk.Label(self.frame1)
        # self.label_image1.place(x=7, y=110)
        #
        # self.label_image2 = tk.Label(self.frame2)
        # self.label_image2.place(x=7, y=110)

        # self.load_and_display_image("assets/top_object.png", (480, 340), self.label_image1)
        # self.load_and_display_image("assets/side_object.png", (480, 340), self.label_image2)

        # self.video_file = "D:\\maftuh\\Projects\\SME\\anomalib\\sample\\dented\\atas\\Video_20240420173419630.avi"
        #
        # self.video_frame(cv2.VideoCapture(self.video_file), self.label_image1, (480, 340))
        # self.video_frame(cv2.VideoCapture(self.video_file), self.label_image2, (480, 340))

        self.obj_cam_operation = []
        # self.ai_engines.start_ai_engines()

        self.training_status = 0  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train

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

    def collect_data_status_label(self, text, font, corner_radius, fg_color, text_color, height, x, y):
        label = CTkLabel(self.window, text=text, font=font, corner_radius=corner_radius,
                         fg_color=fg_color, text_color=text_color, height=height)
        label.place(x=x, y=y)
        return label

    def video_frame(self, cap, label, size):
        ret, frame_image = cap.read()
        if ret:
            frame_image = cv2.resize(frame_image, size)
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_image)
            frame_image = ImageTk.PhotoImage(frame_image)
            label.config(image=frame_image)
            label.image = frame_image
            label.after(10, self.video_frame, cap, label, size)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            label.after(10, self.video_frame, cap, label, size)

    def load_and_display_image(self, filename, size, label):
        image = cv2.imread(filename)
        image = cv2.resize(image, size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image

    # def show_train_models(self, result_label):
    #     if self.ai_engine.good_event.is_set():
    #         self.latest_signal_change = time.time()
    #         self.ai_engine.good_event.clear()
    #         # show good signal
    #     if self.ai_engine.defect_event.is_set():
    #         self.latest_signal_change = time.time()
    #         self.ai_engine.defect_event.clear()
    #         # show ng signal
    #
    #     if (time.time() - self.latest_signal_change > 2
    #             and not self.ai_engine.good_event.is_set()
    #             and not self.ai_engine.defect_event.is_set()):
    #         # show waiting signal
    #         pass

    def show_result(self, relay: int, mode: str):
        pass

    def create_bordered_frame(self, x, y, color, border_width, width, height):
        frame = tk.Frame(self.window, width=width, height=height)
        frame.place(x=x, y=y)

        vertical_frame_left = tk.Frame(frame, width=border_width, height=height, relief="solid", bg=color)
        vertical_frame_left.place(x=0, y=0)

        vertical_frame_right = tk.Frame(frame, width=border_width, height=height, relief="solid", bg=color)
        vertical_frame_right.place(x=width - border_width, y=0)

        horizontal_frame_top = tk.Frame(frame, width=width, height=border_width, relief="solid", bg=color)
        horizontal_frame_top.place(x=0, y=0)

        horizontal_frame_bottom = tk.Frame(frame, width=width, height=border_width, relief="solid", bg=color)
        horizontal_frame_bottom.place(x=0, y=height - border_width)

        return frame

    def add_hint_text(self, text_widget, hint_text):
        text_widget.tag_configure("hint", font=("Roboto", 7, "italic"), foreground="gray")
        text_widget.insert("1.0", hint_text, "hint")
        text_widget.bind("<FocusIn>", lambda event: self.on_focus_in(text_widget, hint_text))
        text_widget.bind("<FocusOut>", lambda event: self.on_focus_out(text_widget, hint_text))

    def on_focus_in(self, text_widget, hint_text):
        if text_widget.get("1.0", "end-1c") == hint_text:
            text_widget.delete("1.0", "end-1c")
            text_widget.tag_remove("hint", "1.0", "end")

    def on_focus_out(self, text_widget, hint_text):
        if not text_widget.get("1.0", "end-1c"):
            text_widget.insert("1.0", hint_text, "hint")

    def event_closing(self):
        if self.training_status == 2:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Cannot close app!\nPlease wait until Train Models Procces Done')
        else:
            if tkinter.messagebox.askokcancel(f'Close | {self.window_title}', "Do you want to close the application?"):
                if self.b_is_run:
                    self.b_is_run = False
                    if self.btn_open_close_device_is_open:
                        self.close_device()
                self.ai_engines.stop_ai_engines()
                self.window.destroy()

    def enum_devices(self):
        if self.training_status == 2:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please wait until Train Models Procces Done')
        else:
            self.deviceList = MV_CC_DEVICE_INFO_LIST()
            self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
            if ret != 0:
                tkinter.messagebox.showerror(f'Error | {self.window_title}', 'enum devices fail! ret = ' + ToHexStr(ret))
                return

            self.label_total_devices.config(text=f'Number of camera           : {self.deviceList.nDeviceNum}')

            if self.deviceList.nDeviceNum == 0:
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'find no device!')
                return
            else:
                print("Find %d devices!" % self.deviceList.nDeviceNum)

            self.devList = []
            for i in range(0, self.deviceList.nDeviceNum):
                mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                    print("\ngige device: [%d]" % i)
                    strModeName = ""
                    for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                        if per == 0:
                            break
                        strModeName = strModeName + chr(per)
                    print("device model name: %s" % strModeName)

                    nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                    nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                    nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                    nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                    print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                    self.devList.append(
                        "Gige[" + str(i) + "]:" + str(nip1) + "." + str(nip2) + "." + str(nip3) + "." + str(nip4))
                elif mvcc_dev_info.nself.tlayerType == MV_USB_DEVICE:
                    print("\nu3v device: [%d]" % i)
                    strModeName = ""
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                        if per == 0:
                            break
                        strModeName = strModeName + chr(per)
                    print("device model name: %s" % strModeName)

                    strSerialNumber = ""
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                        if per == 0:
                            break
                        strSerialNumber = strSerialNumber + chr(per)
                    print("user serial number: %s" % strSerialNumber)
                    self.devList.append("USB[" + str(i) + "]" + str(strSerialNumber))

    def open_close_device(self):
        if self.training_status == 2:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please wait until Train Models Procces Done')
        else:
            if self.deviceList.nDeviceNum == 0 and not DEBUG:
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Find no device!')
            else:
                if self.btn_open_close_device_is_open:
                    self.open_device()
                    self.btn_open_close_device.configure(text="Close Device", fg_color="#CD202E")
                    self.btn_open_close_device_is_open = False
                else:
                    self.close_device()
                    self.btn_open_close_device.configure(text="Open Device", fg_color="#004040")
                    self.btn_open_close_device_is_open = True

    def open_device(self):
        if DEBUG:
            self._dummy_open_device()
        else:
            self._open_device()

    def _open_device(self):
        self.nOpenDevSuccess = 0
        if True == self.b_is_run:
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Camera is Running!')
            return
        self.obj_cam_operation = []
        for i in range(0, self.deviceList.nDeviceNum):
            camObj = MvCamera()
            strName = str(self.devList[i])
            self.obj_cam_operation.append(CameraOperation(camObj, self.ai_engines.ai_engines[i], self.deviceList, i))
            ret = self.obj_cam_operation[self.nOpenDevSuccess].Open_device()
            if 0 != ret:
                self.obj_cam_operation.pop()
                print("open cam %d fail ret[0x%x]" % (i, ret))
                continue
            else:
                print(str(self.devList[i]))
                self.nOpenDevSuccess = self.nOpenDevSuccess + 1
                self.model_val.set('continuous')
                print("self.nOpenDevSuccess = ", self.nOpenDevSuccess)
                self.b_is_run = True
            if 4 == self.nOpenDevSuccess:
                break

    def _dummy_open_device(self):
        self.nOpenDevSuccess = 2
        if self.b_is_run:
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Camera is Running!')
            return
        self.obj_cam_operation = []
        for i in range(2):
            self.obj_cam_operation.append(CameraOperation(None, self.ai_engines.ai_engines[i], self.deviceList, i, dummy=True))

    def start_stop_grabbing(self):
        if self.training_status == 2:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please wait until Train Models Procces Done')
        else:
            if self.btn_open_close_device_is_open:
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please open the device first!')
            else:
                if self.nOpenDevSuccess == 2 or DEBUG:
                    if self.btn_start_stop_grabbing_is_grabbing:
                        self.stop_grabbing()
                        self.btn_start_stop_grabbing.configure(text="Start Collect Images Data", fg_color="#1a8a42")
                        self.btn_start_stop_grabbing_is_grabbing = False
                    else:
                        self.start_grabbing()
                        self.btn_start_stop_grabbing.configure(text="Stop Collect Images Data", fg_color="#CD202E")
                        self.btn_start_stop_grabbing_is_grabbing = True
                else:
                    tkinter.messagebox.showinfo(f'Info | {self.window_title}', f'Found only {self.nOpenDevSuccess} device!')

    def start_grabbing(self):
        for device in range(0, self.nOpenDevSuccess):
            self.set_parameter_auto(device, self.default_exposure_time, self.default_gain, self.default_frame_rate)
            self.label_current_exposure_value[device].configure(text=f"Current Exposure Value\t: {self.default_exposure_time}")
        lock = threading.Lock()  # 申请一把锁
        ret = 0
        for i in range(0, self.nOpenDevSuccess):
            if 0 == i:
                self.label_image1 = tk.Label(self.frame1, image=self.obj_cam_operation[i].current_frame)
                self.label_image1.place(x=7, y=102)
                ret = self.obj_cam_operation[i].Start_grabbing(i, self.frame1, self.label_image1, lock,
                                                               self.show_result)
            elif 1 == i:
                self.label_image2 = tk.Label(self.frame2, image=self.obj_cam_operation[i].current_frame)
                self.label_image2.place(x=7, y=102)
                ret = self.obj_cam_operation[i].Start_grabbing(i, self.frame2, self.label_image2, lock,
                                                               self.show_result)
            # elif 2 == i:
            #     ret = self.obj_cam_operation[i].Start_grabbing(i, self.window, None, lock, None)
            # elif 3 == i:
            #     ret = self.obj_cam_operation[i].Start_grabbing(i, self.window, None, lock, None)
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera:' + str(i) + ',start grabbing fail! ret = ' + self.To_hex_str(ret))

            else:
                self.collect_data_status.configure(text="RUNNING ", fg_color="#1DB74E")

    def stop_grabbing(self):
        for i in range(0, self.nOpenDevSuccess):
            self.label_current_exposure_value[i].configure(text="Current Exposure Value\t: None")
            ret = self.obj_cam_operation[i].Stop_grabbing()
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera:' + str(i) + 'stop grabbing fail!ret = ' + self.To_hex_str(ret))
        print("cam stop grab ok ")

        self.training_status = 1  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train

        self.collect_data_status.configure(text="STOP ", fg_color="#BB1D2A")

    def close_device(self):
        if self.btn_start_stop_grabbing_is_grabbing:
            self.start_stop_grabbing()

        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Close_device()
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera:' + str(i) + 'close deivce fail!ret = ' + self.To_hex_str(ret))
                self.b_is_run = True
                return
        self.b_is_run = False
        print("cam close ok ")
        # 清除文本框的数值
        self.text_frame_rate.delete(1.0, tk.END)
        self.text_gain.delete(1.0, tk.END)

    def set_triggermode(self):
        strMode = self.model_val.get()
        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Set_trigger_mode(strMode)
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera:' + str(i) + 'set ' + strMode + ' fail! ret = ' + self.To_hex_str(ret))

    def trigger_once(self):
        nCommand = self.triggercheck_val.get()
        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Trigger_once(nCommand)
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera:' + str(i) + 'set triggersoftware fail!ret = ' + self.To_hex_str(ret))

    def get_parameter(self):
        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Get_parameter()
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera' + str(i) + 'get parameter fail!ret = ' + self.To_hex_str(ret))
            self.text_frame_rate.delete(1.0, tk.END)
            self.text_frame_rate.insert(1.0, self.obj_cam_operation[i].frame_rate)
            self.text_exposure_time[0].delete(1.0, tk.END)
            self.text_exposure_time[0].insert(1.0, self.obj_cam_operation[i].exposure_time)
            self.text_gain.delete(1.0, tk.END)
            self.text_gain.insert(1.0, self.obj_cam_operation[i].gain)

    def set_parameter_new(self, cam: int):
        if self.training_status == 2:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please wait until Train Models Procces Done')
        else:
            if self.btn_open_close_device_is_open:
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please open the device first!')
            else:
                str_num = self.text_exposure_time[cam].get()
                self.default_exposure_time = int(str_num if str_num else self.default_exposure_time)
                self.label_current_exposure_value[cam].configure(text=f"Current Exposure Value\t: {self.default_exposure_time}")
                self.set_parameter_auto(cam, self.default_exposure_time, self.default_gain, self.default_frame_rate)

    def set_parameter(self):
        for i in range(0, self.nOpenDevSuccess):
            self.obj_cam_operation[i].exposure_time = self.text_exposure_time[0].get(1.0, tk.END)
            self.obj_cam_operation[i].exposure_time = self.obj_cam_operation[i].exposure_time.rstrip("\n")
            self.obj_cam_operation[i].gain = self.text_gain.get(1.0, tk.END)
            self.obj_cam_operation[i].gain = self.obj_cam_operation[i].gain.rstrip("\n")
            self.obj_cam_operation[i].frame_rate = self.text_frame_rate.get(1.0, tk.END)
            self.obj_cam_operation[i].frame_rate = self.obj_cam_operation[i].frame_rate.rstrip("\n")
            ret = self.obj_cam_operation[i].Set_parameter(self.obj_cam_operation[i].frame_rate,
                                                          self.obj_cam_operation[i].exposure_time,
                                                          self.obj_cam_operation[i].gain)
            if 0 != ret:
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             'Camera' + str(i) + 'set parameter fail!ret = ' + self.To_hex_str(ret))

    def set_parameter_auto(self, cam_num, default_exposure_time, default_gain, default_frame_rate):
        i = cam_num
        self.obj_cam_operation[i].exposure_time = default_exposure_time
        self.obj_cam_operation[i].gain = default_gain
        self.obj_cam_operation[i].frame_rate = default_frame_rate
        ret = self.obj_cam_operation[i].Set_parameter(self.obj_cam_operation[i].frame_rate,
                                                      self.obj_cam_operation[i].exposure_time,
                                                      self.obj_cam_operation[i].gain)
        if 0 != ret:
            tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                         'Camera' + str(i) + 'set parameter fail!ret = ' + self.To_hex_str(ret))

    def start_training(self):
        if self.training_status != 1:  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please start collect images data first!')
        elif self.radio_btn_good_or_defect_variable.get() == 0 or self.radio_btn_top_or_side_variable.get() == 0:
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please select radio button first!')
        else:
            self.training_status = 2  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            self.collect_data_status.configure(text="TRAINING THE MODELS ", fg_color="#1DB74E")

            self.training()

            self.training_status = 3  # 0 -> Init, 1 -> Done Collect Data, 2 -> Training, 3 -> Done Train
            self.collect_data_status.configure(text="STOP ", fg_color="#BB1D2A")

    def training(self):
        print("Start training...")


class GUIAIEngine:
    def __init__(self, number_of_camera):
        self.ai_engines = []
        self.ai_engines_processes = []

        self.number_of_camera = number_of_camera

    def stop_ai_engines(self):
        for engine in self.ai_engines:
            engine.stop()

        for p in self.ai_engines_processes:
            p.join()

        for engine in self.ai_engines:
            engine.clear()

    def start_ai_engines(self):
        for i in range(self.number_of_camera):
            engine = AIEngine(camera_id=i)
            p = engine.start()
            time.sleep(0.2)
            self.ai_engines_processes.extend(p)
            self.ai_engines.append(engine)


if __name__ == "__main__":
    ai_engines = GUIAIEngine(number_of_camera=2)
    ai_engines.start_ai_engines()

    gui = GUI(ai_engines=ai_engines)
    gui.window.mainloop()
