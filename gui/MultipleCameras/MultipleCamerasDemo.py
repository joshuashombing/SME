import os
import warnings

from anomalib.utils.loggers.logger import setup_logger
from config import DEBUG, RESULT_DIR

logger = setup_logger("AutoInspection", log_dir=RESULT_DIR / "logs")

warnings.filterwarnings('ignore')

# -- coding: utf-8 --
import sys
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk
from tkinter import filedialog

sys.path.append("../MvImport")
# from MvCameraControl_class import *
from CamOperation_class_v1 import *
from PIL import Image, ImageTk
import cv2

import customtkinter as ctk
from customtkinter import CTkLabel, CTkButton, CTkEntry, CTkImage, CTkCheckBox

from engine_v2 import AIEngine, start_spawn_method
from relay import Relay


class AIEngineGUI:
    def __init__(self):
        self.camera_ids = {
            "kamera-samping": 0,
            "kamera-atas": 1
        }
        self.ai_engines = {}
        self.ai_engines_processes = []
        self.relay = Relay()
        self._initialize()

    def _initialize(self):
        for camera_id in self.camera_ids.values():
            self.ai_engines[camera_id] = AIEngine(camera_id=camera_id, relay=self.relay)

    def start_ai_engines(self):
        self.relay.open()
        start_spawn_method()
        for camera_id, engine in self.ai_engines.items():
            if engine is not None:
                engine.start()

    @property
    def ai_engines_started(self):
        return all([engine.is_model_built for engine in self.ai_engines.values() if engine is not None])

    def stop_ai_engines(self):
        for camera_id, engine in self.ai_engines.items():
            if engine is not None:
                engine.stop()

        self.relay.close()


class MainGUI(AIEngineGUI):

    def __init__(self):
        super().__init__()

        self.window = tk.Tk()
        self.window_title = 'SME Spring Sheet Metal Inspection'
        self.window.iconbitmap("satnusa.ico")
        self.window.title(self.window_title)
        self.window_width = 1062
        self.window_height = 690
        self.assumption_taskbar_height = 72  # Windows 10
        self.assumption_taskbar_height_fullscreen = 63  # Windows 10
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight() - self.assumption_taskbar_height
        self.window_default_width = self.window_width
        self.window_default_height = self.window_height
        self.window_delta_width = 0
        self.window_delta_height = 0
        self.window_prev_delta_width = 0
        self.window_prev_delta_height = 0
        self.window_delta_width_height_is_zero = False
        self.window.geometry(f'{self.window_width}x{self.window_height}')
        self.window.resizable(False, False)
        # self.window.minsize(self.window_width, self.window_height)
        self.window.bind("<Configure>", self.windows_is_on_resize)
        self.cam_frame_size_default = (482, 363)
        self.cam_frame_size = [self.cam_frame_size_default[0], self.cam_frame_size_default[1]]

        # Handling event closing
        self.window.protocol("WM_DELETE_WINDOW", self.event_closing)

        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.obj_cam_operation = []
        self.b_is_run = False
        self.nOpenDevSuccess = 0
        self.devList = None
        self.triggercheck_val = tk.IntVar()
        self.page = Frame(self.window, height=0, width=0)
        self.page.pack(expand=True, fill=BOTH)

        self.default_exposure_time = [2000, 2000]
        self.default_gain = [19.9963, 19.9963]
        self.default_frame_rate = [9.5000, 9.5000]

        self.ok_counter_camera1 = 0
        self.ng_counter_camera1 = 0
        self.ok_counter_camera2 = 0
        self.ng_counter_camera2 = 0

        # Header
        self.header_line = tk.Label(self.window, width=145, bg="#004040")
        self.header_line.place(x=21, y=25)
        self.header_label_thinner = tk.Label(self.window, width=145)
        self.header_label_thinner.place(x=21, y=23)
        self.header_label = tk.Label(self.window, text=self.window_title, font=("Roboto", 21), fg="#333")
        self.header_label.place(x=16, y=1)

        self.label_total_devices = tk.Label(self.window, font=("Roboto", 11), text='Number of Camera\t: 0',
                                            fg="#004040")
        self.label_total_devices.place(x=18, y=90)

        self.button_style = ttk.Style()
        self.button_style.configure('TButton', foreground="#333")

        self.btn_enum_devices = CTkButton(master=self.window,
                                          width=120,
                                          height=30,
                                          border_width=0,
                                          corner_radius=4,
                                          text="Check Camera",
                                          fg_color="#c3c3c3",
                                          text_color="black",
                                          font=("Roboto", 12),
                                          command=self.enum_devices)
        self.btn_enum_devices.place(x=175, y=87)

        self.open_close_button_is_clicked = False
        self.open_close_button_style = ttk.Style()
        self.open_close_button_style.configure('OpenClose.TButton')
        self.btn_open_close_device_is_open = True
        self.btn_open_close_device = CTkButton(master=self.window,
                                               width=120,
                                               height=30,
                                               border_width=0,
                                               corner_radius=4,
                                               text="Open Camera",
                                               fg_color="#1a8a42",
                                               text_color="white",
                                               font=("Roboto", 12),
                                               command=self.open_close_device)
        self.btn_open_close_device.place(x=305, y=87)

        # self.btn_start_stop_grabbing_style = ttk.Style()
        # self.btn_start_stop_grabbing_style.configure('OpenClose.TButton')
        # self.btn_start_stop_grabbing_is_grabbing = False
        # self.btn_start_stop_grabbing_is_ever_grabbing = False
        # self.btn_start_stop_grabbing = CTkButton(master=self.window,
        #                                          width=120,
        #                                          height=30,
        #                                          border_width=0,
        #                                          corner_radius=4,
        #                                          text="Start Inspection",
        #                                          fg_color="#1a8a42",
        #                                          text_color="white",
        #                                          font=("Roboto", 12),
        #                                          command=self.start_stop_grabbing)
        # # self.btn_start_stop_grabbing.config(state="disable")
        # self.btn_start_stop_grabbing.place(x=311, y=20)

        # self.btn_show_inspection_result = CTkButton(master=self.window,
        #                                             width=120,
        #                                             height=30,
        #                                             border_width=0,
        #                                             corner_radius=4,
        #                                             text="Show Results",
        #                                             fg_color="#c3c3c3",
        #                                             text_color="black",
        #                                             font=("Roboto", 12),
        #                                             image=CTkImage(Image.open("show_result-icon.png")),
        #                                             command=self.show_folder_result)
        # self.btn_show_inspection_result.place(x=659, y=87)

        # self.btn_full_screen = CTkButton(master=self.window,
        #                                  width=120,
        #                                  height=30,
        #                                  border_width=0,
        #                                  corner_radius=4,
        #                                  text="Full Screen",
        #                                  fg_color="#c3c3c3",
        #                                  text_color="black",
        #                                  font=("Roboto", 12),
        #                                  image=CTkImage(Image.open("fullscreen-icon.png")),
        #                                  command=self.full_screen)
        # self.btn_full_screen.place(x=790, y=87)

        # self.btn_setting = CTkButton(master=self.window,
        #                              width=120,
        #                              height=30,
        #                              border_width=0,
        #                              corner_radius=4,
        #                              text="Setting",
        #                              fg_color="#c3c3c3",
        #                              text_color="black",
        #                              font=("Roboto", 12),
        #                              image=CTkImage(Image.open("setting-icon.png")),
        #                              command=self.setting)
        #
        # self.btn_setting.place(x=921, y=87)

        self.window_is_full_screen = False

        # Camera
        self.number_camera_defined = 2  # Set default value -> 2

        # Create two bordered frames
        self.frame2 = tk.Frame(self.window, width=500, height=520)
        self.frame2.place(x=541, y=148)

        self.frame2_vertical_frame_left = tk.Frame(self.frame2, width=1, height=520, relief="solid", bg="#A0A0A0")
        self.frame2_vertical_frame_left.place(x=0, y=0)

        self.frame2_vertical_frame_right = tk.Frame(self.frame2, width=1, height=520, relief="solid", bg="#A0A0A0")
        self.frame2_vertical_frame_right.place(x=500 - 1, y=0)

        self.frame2_horizontal_frame_top = tk.Frame(self.frame2, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame2_horizontal_frame_top.place(x=0, y=0)

        self.frame2_horizontal_frame_bottom = tk.Frame(self.frame2, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame2_horizontal_frame_bottom.place(x=0, y=520 - 1)

        self.frame1 = tk.Frame(self.window, width=500, height=520)
        self.frame1.place(x=21, y=148)

        self.frame1_vertical_frame_left = tk.Frame(self.frame1, width=1, height=520, relief="solid", bg="#A0A0A0")
        self.frame1_vertical_frame_left.place(x=0, y=0)

        self.frame1_vertical_frame_right = tk.Frame(self.frame1, width=1, height=520, relief="solid", bg="#A0A0A0")
        self.frame1_vertical_frame_right.place(x=500 - 1, y=0)

        self.frame1_horizontal_frame_top = tk.Frame(self.frame1, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame1_horizontal_frame_top.place(x=0, y=0)

        self.frame1_horizontal_frame_bottom = tk.Frame(self.frame1, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame1_horizontal_frame_bottom.place(x=0, y=520 - 1)

        self.label_image1 = tk.Label(self.frame1)
        self.label_image2 = tk.Label(self.frame2)
        self.label_image_pos_default = (7, 149)
        self.label_image_pos = [self.label_image_pos_default[0], self.label_image_pos_default[1]]

        self.img_no_camera = tk.PhotoImage(file="no_image.png").subsample(4,
                                                                          4)  # Menyesuaikan ukuran gambar menjadi 50x50
        self.no_camera_label1 = tk.Label(self.frame1, image=self.img_no_camera)
        self.no_camera_label1.place(x=115, y=155)

        self.no_camera_label2 = tk.Label(self.frame2, image=self.img_no_camera)
        self.no_camera_label2.place(x=115, y=155)

        self.camera_status = [
            tk.Label(self.frame1, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold")),
            tk.Label(self.frame2, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold"))]

        self.camera_status[0].place(x=160, y=380)
        self.camera_status[1].place(x=160, y=380)

        self.camera_label_side = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera 2 ',
                                          fg="#333")
        self.camera_label_side.place(x=551, y=135)
        self.camera_reset_camera2 = CTkButton(master=self.window,
                                              width=80,
                                              height=30,
                                              border_width=0,
                                              corner_radius=4,
                                              text="Reset",
                                              fg_color="#FEFEFE",
                                              text_color="black",
                                              font=("Roboto", 12),
                                              command=lambda: self.reset_counter(0),
                                              image=CTkImage(Image.open("reset-icon.png"))
                                              )
        self.camera_reset_camera2.place(x=952, y=135)

        self.camera_label_top = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera 1 ',
                                         fg="#333")
        self.camera_label_top.place(x=30, y=135)
        self.camera_reset_camera1 = CTkButton(master=self.window,
                                              width=80,
                                              height=30,
                                              border_width=0,
                                              corner_radius=4,
                                              text="Reset",
                                              fg_color="#FEFEFE",
                                              text_color="black",
                                              font=("Roboto", 12),
                                              command=lambda: self.reset_counter(1),
                                              image=CTkImage(Image.open("reset-icon.png"))
                                              )
        self.camera_reset_camera1.place(x=432, y=135)

        self.label_current_exposure_value = [tk.Label(self.frame2,
                                                      text="Current Exposure Value\t: None",
                                                      font=("Roboto", 8), fg="#004040"),
                                             tk.Label(self.frame1,
                                                      text="Current Exposure Value\t: None",
                                                      font=("Roboto", 8), fg="#004040")]

        # self.label_ok = [tk.Label(self.frame2, text="4500")]

        # Place the labels inside the bordered frames
        self.label_exposure_camera2_ok_value = tk.Label(self.frame2, width=8, text="0", font=("Roboto", 14),
                                                        fg="#1DB74E", bg="#FEFEFE")
        self.label_exposure_camera2_ok_value.place(x=10, y=54)

        self.label_exposure_camera2_ok_status = tk.Label(self.frame2, width=8, text="OK", font=("Roboto", 14),
                                                         fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera2_ok_status.place(x=10, y=25)

        self.label_exposure_camera2_ng_value = tk.Label(self.frame2, width=12, text="0", font=("Roboto", 14),
                                                        fg="#CD202E", bg="#FEFEFE")
        self.label_exposure_camera2_ng_value.place(x=110, y=54)

        self.label_exposure_camera2_ng_status = tk.Label(self.frame2, width=12, text="NG", font=("Roboto", 14),
                                                         fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera2_ng_status.place(x=110, y=25)

        # self.label_percentage_camera2 = tk.Label(self.frame2, width=3, text='15%', font=("Roboto", 11), fg="#CD202E", bg="#FEFEFE")
        # self.label_percentage_camera2.place(x=210, y=42)
        self.label_exposure_camera2_total_status = tk.Label(self.frame2, width=10, text="INPUT", font=("Roboto", 14),
                                                            fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera2_total_status.place(x=255, y=25)

        self.label_exposure_camera2_total_value = tk.Label(self.frame2, width=10, text="0", font=("Roboto", 14),
                                                           fg="#004040", bg="#FEFEFE")
        self.label_exposure_camera2_total_value.place(x=255, y=54)

        

        self.inspection_activity_label_camera_2 = tk.Label(self.frame2, font=("Roboto", 10), text='Inspection Activity',
                                                           fg="#333")
        self.inspection_activity_label_camera_2.place(x=375, y=20)

        # Label untuk Kamera 1
        self.label_exposure_camera1_ok_value = tk.Label(self.frame1, width=8, text="0", font=("Roboto", 14),
                                                        fg="#1DB74E", bg="#FEFEFE")
        self.label_exposure_camera1_ok_value.place(x=10, y=54)

        self.label_exposure_camera1_ok_status = tk.Label(self.frame1, width=8, text="OK", font=("Roboto", 14),
                                                         fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera1_ok_status.place(x=10, y=25)

        self.label_exposure_camera1_ng_value = tk.Label(self.frame1, width=12, text="0", font=("Roboto", 14),
                                                        fg="#CD202E", bg="#FEFEFE")
        self.label_exposure_camera1_ng_value.place(x=110, y=54)

        self.label_exposure_camera1_ng_status = tk.Label(self.frame1, width=12, text="NG", font=("Roboto", 14),
                                                         fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera1_ng_status.place(x=110, y=25)

        # self.label_percentage_camera1 = tk.Label(self.frame1, width=3, text='15%', font=("Roboto", 11), fg="#CD202E", bg="#FEFEFE")
        # self.label_percentage_camera1.place(x=210, y=42)
        # #
        self.label_exposure_camera1_total_value = tk.Label(self.frame1, width=10, text="0", font=("Roboto", 14),
                                                           fg="#004040", bg="#FEFEFE")
        self.label_exposure_camera1_total_value.place(x=255, y=54)

        self.label_exposure_camera1_total_status = tk.Label(self.frame1, width=10, text="INPUT", font=("Roboto", 14),
                                                            fg="#000000", bg="#FEFEFE")
        self.label_exposure_camera1_total_status.place(x=255, y=25)

        self.inspection_activity_label_camera_1 = tk.Label(self.frame1, font=("Roboto", 10), text='Inspection Activity',
                                                           fg="#333")
        self.inspection_activity_label_camera_1.place(x=375, y=20)

        self.label_status_camera2 = tk.Label(self.frame2, font=("Roboto", 8), text='Status :', fg="#004040")
        self.label_status_camera2.place(x=375, y=41)

        self.label_status_camera1 = tk.Label(self.frame1, font=("Roboto", 8), text='Status :', fg="#004040")
        self.label_status_camera1.place(x=375, y=41)

        self.inspection_status_frame1 = self.inspection_status_label(
            self.frame1,
            text="STOP",
            font=("Roboto", 10),
            corner_radius=4,
            fg_color="#CD202E",
            text_color="white",
            height=20,
            x=430,
            y=41
        )

        self.inspection_status_frame2 = self.inspection_status_label(
            self.frame2,
            text="STOP",
            font=("Roboto", 10),
            corner_radius=4,
            fg_color="#CD202E",
            text_color="white",
            height=20,
            x=430,
            y=41
        )

        # self.frames = [self.frame1, self.frame2]
        # self.labels = [self.label_image1, self.label_image2]
        self.inspection_status_frames = [self.inspection_status_frame2, self.inspection_status_frame1]
        self.btn_start_stop_grabbing_buttons = []
        self.btn_start_stop_grabbing_is_grabbing = [False, False]

        # Style and Button initialization
        self.btn_start_stop_grabbing_style_camera1 = ttk.Style()
        self.btn_start_stop_grabbing_style_camera1.configure('OpenClose.TButton')

        self.btn_start_stop_grabbing_style_camera2 = ttk.Style()
        self.btn_start_stop_grabbing_style_camera2.configure('OpenClose.TButton')

        self.btn_start_stop_grabbing_camera2 = CTkButton(master=self.frame2,
                                                         width=0,
                                                         border_width=0,
                                                         corner_radius=4,
                                                         text="Start Inspection",
                                                         fg_color="#1a8a42",
                                                         text_color="white",
                                                         font=("Roboto", 10),
                                                         image=CTkImage(Image.open("shutdown.png")),
                                                         command=lambda: self.start_stop_grabbing(0))
        self.btn_start_stop_grabbing_camera2.place(x=378, y=65)
        self.btn_start_stop_grabbing_buttons.append(self.btn_start_stop_grabbing_camera2)

        self.btn_start_stop_grabbing_camera1 = CTkButton(master=self.frame1,
                                                         width=0,
                                                         border_width=0,
                                                         corner_radius=4,
                                                         text="Start Inspection",
                                                         fg_color="#1a8a42",
                                                         text_color="white",
                                                         font=("Roboto", 10),
                                                         image=CTkImage(Image.open("shutdown.png")),
                                                         command=lambda: self.start_stop_grabbing(1))
        self.btn_start_stop_grabbing_camera1.place(x=378, y=65)
        self.btn_start_stop_grabbing_buttons.append(self.btn_start_stop_grabbing_camera1)
        self.label_percentage = [
            tk.Label(self.frame2, width=6, text='0%', font=("Roboto", 11), fg="#CD202E", bg="#FEFEFE"),
            tk.Label(self.frame1, width=6, text='0%', font=("Roboto", 11), fg="#CD202E", bg="#FEFEFE")
        ]

        self.label_percentage[0].place(x=193, y=27)
        self.label_percentage[1].place(x=193, y=27)

        # self.text_exposure_time = [
        #     CTkEntry(master=self.frame2,
        #              placeholder_text="Insert Value",
        #              width=116,
        #              height=30,
        #              justify="center",
        #              border_color='#FEFEFE',  # Set border color
        #              corner_radius=0,  # Set corner radius to 0 for no rounded corners
        #              bg_color="#FEFEFE",  # Set background color to transparent
        #              fg_color="#FEFEFE",  # Set foreground color to transparent
        #              font=("Roboto", 11),
        #              text_color="#000000"),
        #     CTkEntry(master=self.frame1,
        #              placeholder_text="Insert Value",
        #              width=116,
        #              height=30,
        #              justify="center",
        #              border_color='#FEFEFE',  # Set border color
        #              corner_radius=0,  # Set corner radius to 0 for no rounded corners
        #              bg_color="#FEFEFE",  # Set background color to transparent
        #              fg_color="#FEFEFE",  # Set foreground color to transparent
        #              font=("Roboto", 11),
        #              text_color="#000000")
        # ]

        self.label_ok_count_values = [self.label_exposure_camera2_ok_value, self.label_exposure_camera1_ok_value]
        self.label_ng_count_values = [self.label_exposure_camera2_ng_value, self.label_exposure_camera1_ng_value]
        self.label_total_count_values = [self.label_exposure_camera2_total_value, self.label_exposure_camera1_total_value]

        # self.text_exposure_time[0].bind("<KeyRelease>", lambda event: self.on_input_percentage_change(0, event))
        # self.text_exposure_time[1].bind("<KeyRelease>", lambda event: self.on_input_percentage_change(1, event))

        # self.text_exposure_time[0].place(x=255, y=53)
        # self.text_exposure_time[1].place(x=255, y=53)

        # self.btn_set_param_cam1 = CTkButton(master=self.frame2,
        #                                     width=80,
        #                                     height=20,
        #                                     border_width=0,
        #                                     corner_radius=4,
        #                                     text="Save Value",
        #                                     fg_color="#004040",
        #                                     text_color="white",
        #                                     font=("Roboto", 12),
        #                                     command=lambda: self.set_parameter_new(0))
        # self.btn_set_param_cam1.place(x=275, y=60)

        # self.btn_set_param_cam2 = CTkButton(master=self.frame1,
        #                                     width=80,
        #                                     height=20,
        #                                     border_width=0,
        #                                     corner_radius=4,
        #                                     text="Save Value",
        #                                     fg_color="#004040",
        #                                     text_color="white",
        #                                     font=("Roboto", 12),
        #                                     command=lambda: self.set_parameter_new(1))
        # self.btn_set_param_cam2.place(x=275, y=30)

        # camera_inspection_status_label
        self.camera_inspection_status_label_canvas_1 = tk.Canvas(self.frame1, background=None, highlightthickness=0,
                                                                 width=490, height=45)
        self.camera_inspection_status_label_canvas_1.place(relx=0.5, rely=0.23, anchor="center")
        self.camera_inspection_status_label_canvas_1_rectangle = self.camera_inspection_status_label_canvas_1.create_rectangle(
            3, 3, 485, 37, dash=(5, 1), outline="#1DB74E", fill="")
        self.camera_inspection_status_label_label_1 = tk.Label(self.camera_inspection_status_label_canvas_1, bg=None,
                                                               fg="#004040", text='', font=("Roboto", 12, "bold"))
        self.camera_inspection_status_label_label_1.place(relx=0.5, rely=0.545, anchor="center")

        self.camera_inspection_status_label_canvas_2 = tk.Canvas(self.frame2, background=None, highlightthickness=0,
                                                                 width=490, height=45)
        self.camera_inspection_status_label_canvas_2.place(relx=0.5, rely=0.23, anchor="center")
        self.camera_inspection_status_label_canvas_2_rectangle = self.camera_inspection_status_label_canvas_2.create_rectangle(
            3, 3, 485, 37, dash=(5, 1), outline="#1DB74E", fill="")
        self.camera_inspection_status_label_label_2 = tk.Label(self.camera_inspection_status_label_canvas_2, bg=None,
                                                               fg="#004040", text='', font=("Roboto", 12, "bold"))
        self.camera_inspection_status_label_label_2.place(relx=0.5, rely=0.545, anchor="center")

        # # Label 1
        # self.camera_inspection_status_label_1 = tk.Label(self.frame2, bg="red", fg="white", width=47, height=1, text="NG", font=("Roboto", 12, "bold"))
        # self.camera_inspection_status_label_1.place(relx=0.5, anchor="center", y=105)

        # # Label 2
        # self.camera_inspection_status_label_2 = tk.Label(self.frame1, bg="#158237", fg="white", width=47, height=1, text="GOOD", font=("Roboto", 12, "bold"))
        # self.camera_inspection_status_label_2.place(relx=0.5, anchor="center", y=105)

        self.result_bg = [
            tk.Label(self.frame1, background="#1DB74E", text='', font=("Roboto", 16, "bold"), padx=240, pady=6),
            tk.Label(self.frame2, background="#1DB74E", text='', font=("Roboto", 16, "bold"), padx=240, pady=6),
        ]

        self.result_text = [
            tk.Label(self.frame1, background="#1DB74E", text='', font=("Roboto", 16, "bold"), foreground='white'),
            tk.Label(self.frame2, background="#1DB74E", text='', font=("Roboto", 16, "bold"), foreground='white')
        ]

        self.result_pos_bg = ((8, 100), (8, 100))
        self.result_pos_text = [[230, 105], [230, 105]]

        self.show_result(1, "empty")
        self.show_result(2, "empty")
        
    def update_counter_result(self, index):
        """
        Updates the OK/NG count labels for the specified engine index.

        Parameters:
            index (int): The index of the AI engine/camera (0 or 1).

        Returns:
            None
        """
        # Get the associated counter for the AI engine at the given index
        counter = self.ai_engines[index].counter

        # Update the OK count label
        self.label_ok_count_values[index].config(text=str(counter.num_good))

        # Update the NG count label
        self.label_ng_count_values[index].config(text=str(counter.num_defect))

        # Update the total count label (if applicable)
        if hasattr(self, 'label_total_count_values'):
            self.label_total_count_values[index].config(text=str(counter.total))

        # Optionally, update the NG percentage
        self.calculate_ng_percentage(index)



    def calculate_ng_percentage(self, index):
        count_ng = self.label_ng_count_values[index].cget("text")
        count_total = self.label_total_count_values[index].cget("text")
        # print("Count:", count)
        # print("Input:", content)

        if count_total > 0:
            percentage = (count_ng / count_total) * 100
        else:
            percentage = 0

        # percentage = 100 * percentage
        # # print("percentage", percentage)
        self.label_percentage[index].configure(text=f"{percentage:.2f}%")

    # def on_input_percentage_change(self, index, event):
    #     # Get the content of the entry widget
    #     content = self.text_exposure_time[index].get()

    #     # Check if the content is numeric
    #     if content.isdigit() or content == "":
    #         # print("Valid numeric input:", content)
    #         self.calculate_ng_percentage(index, content)
    #     else:
    #         # If the input is not numeric, delete the last character
    #         self.text_exposure_time[index].delete(len(content) - 1, ctk.END)

    def full_screen(self):
        self.windows_is_on_resize(event=None)
        if self.window_is_full_screen == False:
            self.btn_full_screen.configure(text="Exit Full Screen")
            self.window_is_full_screen = True
        else:
            self.btn_full_screen.configure(text="Full Screen")
            self.window_is_full_screen = False

        self.window.attributes("-fullscreen", self.window_is_full_screen)

    def windows_is_on_resize(self, event):
        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        # self.window_width = event.width
        # self.window_height = event.height

        self.window_delta_width = self.window_width - self.window_default_width
        self.window_delta_height = self.window_height - self.window_default_height

        if self.window_delta_width >= 0:
            if (self.window_prev_delta_width != self.window_delta_width or
                    self.window_prev_delta_height != self.window_delta_height):
                self.window_prev_delta_width = self.window_delta_width
                self.window_prev_delta_height = self.window_delta_height
                self.windows_resized(self.window_delta_width, self.window_delta_height)

    def windows_resized(self, delta_width, delta_height):
        self.screen_height = self.window.winfo_screenheight() - self.assumption_taskbar_height
        self.window_width_barrier = self.screen_height - self.window_default_height

        half_delta_width = (delta_width // 2)
        small_delta_width = (delta_width // 69)
        test_delta_width = (delta_width // 19)
        quarter_delta_width = (delta_width // 4)
        delta_width_ = (delta_width // 7)
        half_delta_height = (delta_height // 2)

        ng_label_offset = (small_delta_width * 12)
        total_label_offset = (small_delta_width * 24)

        # Header
        self.header_line.configure(width=145 + delta_width_)
        self.header_label_thinner.configure(width=145 + delta_width_)

        # Frame
        self.frame2.configure(width=500 + half_delta_width, height=520 + delta_height)
        self.frame2.place(x=541 + half_delta_width, y=148)
        self.frame1.configure(width=500 + half_delta_width, height=520 + delta_height)
        self.camera_label_side.place(x=551 + half_delta_width, y=135)

        self.frame2_vertical_frame_left.configure(height=520 + delta_height)
        self.frame2_vertical_frame_right.configure(height=520 + delta_height)
        self.frame2_vertical_frame_right.place(x=500 + half_delta_width - 1, y=0)
        self.frame2_horizontal_frame_top.configure(width=500 + half_delta_width)
        self.frame2_horizontal_frame_bottom.configure(width=500 + half_delta_width)
        self.frame2_horizontal_frame_bottom.place(x=0, y=520 - 1 + delta_height)

        self.frame1_vertical_frame_left.configure(height=520 + delta_height)
        self.frame1_vertical_frame_right.configure(height=520 + delta_height)
        self.frame1_vertical_frame_right.place(x=500 + half_delta_width - 1, y=0)
        self.frame1_horizontal_frame_top.configure(width=500 + half_delta_width)
        self.frame1_horizontal_frame_bottom.configure(width=500 + half_delta_width)
        self.frame1_horizontal_frame_bottom.place(x=0, y=520 - 1 + delta_height)

        self.no_camera_label1.place(x=115 + quarter_delta_width, y=130 + half_delta_height)
        self.no_camera_label2.place(x=115 + quarter_delta_width, y=130 + half_delta_height)
        self.camera_status[0].place(x=160 + quarter_delta_width, y=380 + half_delta_height)
        self.camera_status[1].place(x=160 + quarter_delta_width, y=380 + half_delta_height)

        self.camera_reset_camera1.place(x=432 + half_delta_width, y=135)
        self.camera_reset_camera2.place(x=952 + delta_width, y=135)

        # self.camera_inspection_status_label_1.configure(width=47+test_delta_width)
        # self.camera_inspection_status_label_2.configure(width=47+test_delta_width)

        # # self.camera_inspection_status_label_1.place(y=105)
        # self.camera_inspection_status_label_2.place(y=105)

        self.inspection_status_frame1.place(x=430 + half_delta_width, y=41)
        self.inspection_status_frame2.place(x=430 + half_delta_width, y=41)

        self.inspection_activity_label_camera_1.place(x=375 + half_delta_width, y=20)
        self.inspection_activity_label_camera_2.place(x=375 + half_delta_width, y=20)

        self.btn_start_stop_grabbing_camera1.place(x=378 + half_delta_width, y=65)
        self.btn_start_stop_grabbing_camera2.place(x=378 + half_delta_width, y=65)

        self.label_status_camera1.place(x=375 + half_delta_width, y=41)
        self.label_status_camera2.place(x=375 + half_delta_width, y=41)

        self.label_exposure_camera2_ok_value.configure(width=10 + small_delta_width)
        self.label_exposure_camera2_ng_value.configure(width=10 + small_delta_width)
        self.label_exposure_camera2_total_value.configure(width=10 + small_delta_width)

        self.label_exposure_camera2_ok_value.place(x=10)
        self.label_exposure_camera2_ng_value.place(x=130 + ng_label_offset)
        self.label_exposure_camera2_total_value.place(x=250 + total_label_offset)

        self.label_exposure_camera2_ok_status.configure(width=10 + small_delta_width)
        self.label_exposure_camera2_ng_status.configure(width=10 + small_delta_width)
        self.label_exposure_camera2_total_status.configure(width=10 + small_delta_width)

        self.label_exposure_camera2_ok_status.place(x=10)
        self.label_exposure_camera2_ng_status.place(x=130 + ng_label_offset)
        self.label_exposure_camera2_total_status.place(x=250 + total_label_offset)

        # Kamera 1
        self.label_exposure_camera1_ok_value.configure(width=10 + small_delta_width)
        self.label_exposure_camera1_ng_value.configure(width=10 + small_delta_width)
        self.label_exposure_camera1_total_value.configure(width=10 + small_delta_width)

        self.label_exposure_camera1_ok_value.place(x=10)
        self.label_exposure_camera1_ng_value.place(x=130 + ng_label_offset)
        self.label_exposure_camera1_total_value.place(x=250 + total_label_offset)

        self.label_exposure_camera1_ok_status.configure(width=10 + small_delta_width)
        self.label_exposure_camera1_ng_status.configure(width=10 + small_delta_width)
        self.label_exposure_camera1_total_status.configure(width=10 + small_delta_width)

        self.label_exposure_camera1_ok_status.place(x=10)
        self.label_exposure_camera1_ng_status.place(x=130 + ng_label_offset)
        self.label_exposure_camera1_total_status.place(x=250 + total_label_offset)

        self.btn_show_inspection_result.place(x=659 + delta_width, y=87)
        self.btn_full_screen.place(x=790 + delta_width, y=87)
        self.btn_setting.place(x=921 + delta_width, y=87)

        # # Inspection result
        self.camera_inspection_status_label_canvas_1.configure(width=490 + half_delta_width)
        self.camera_inspection_status_label_canvas_1.coords(self.camera_inspection_status_label_canvas_1_rectangle, 3,
                                                            3, 485 + half_delta_width, 37)
        self.camera_inspection_status_label_canvas_1.place(relx=0.5, rely=0.23 - (delta_height / 4300), anchor="center")

        self.camera_inspection_status_label_canvas_2.configure(width=490 + half_delta_width)
        self.camera_inspection_status_label_canvas_2.coords(self.camera_inspection_status_label_canvas_2_rectangle, 3,
                                                            3, 485 + half_delta_width, 37)
        self.camera_inspection_status_label_canvas_2.place(relx=0.5, rely=0.23 - (delta_height / 4300), anchor="center")

        # Images
        self.result_pos_text = [[210 + quarter_delta_width, 94], [230 + quarter_delta_width, 94]]

        self.result_bg[0].configure(padx=240 + quarter_delta_width)
        self.result_bg[1].configure(padx=240 + quarter_delta_width)
        self.result_bg[0].place(x=8, y=99 + (delta_height / 80))
        self.result_bg[1].place(x=8, y=99 + (delta_height / 80))

        cam_delta_width = self.cam_frame_size_default[
            0]  # Default val from CamOperation cam_delta_width = 482 and cam_delta_height = 363
        cam_delta_height = None
        x_label_image = self.label_image_pos_default[0]
        y_label_image = self.label_image_pos_default[1]

        if self.window_delta_width == 0:
            y_label_image += half_delta_height
        elif self.window_delta_height == 0:
            x_label_image += quarter_delta_width
        elif self.window_delta_width < self.window_delta_height:
            cam_delta_width += half_delta_width
            y_adjustment = half_delta_height - (quarter_delta_width * 2 // 3)
            y_label_image += y_adjustment
        elif self.window_delta_height <= self.window_delta_width:
            if self.window_delta_width >= self.window_width_barrier and self.window_delta_height >= self.window_width_barrier:
                cam_delta_width += half_delta_width
                y_adjustment = half_delta_height - (quarter_delta_width * 0.78)
                y_label_image += y_adjustment
            else:
                cam_delta_width = None
                cam_delta_height = self.cam_frame_size_default[1] + self.window_delta_height
                x_adjustment = quarter_delta_width - (self.window_delta_height * 2 // 3)
                x_label_image += x_adjustment

        self.label_image_pos = [x_label_image, y_label_image]
        if self.btn_start_stop_grabbing_is_grabbing:
            if self.label_image1 is not None:
                self.label_image1.place(x=x_label_image, y=y_label_image)
            if self.label_image2 is not None:
                self.label_image2.place(x=x_label_image, y=y_label_image)

        self.cam_frame_size = [cam_delta_width, cam_delta_height]

        for obj_cam in self.obj_cam_operation:
            obj_cam.set_frame_size(cam_delta_width, cam_delta_height)

    def to_hex_str(self, num):
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

    def show_folder_result(self):
        path = RESULT_DIR / "images"

        # Get the list of all files and folders in the directory
        items = os.listdir(path)

        # Filter only folders
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

        # Use the sorted function to sort folders based on modification time
        latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(path, f)))

        path = os.path.join(path, latest_folder)

        os.startfile(path)

    def inspection_status_label(self, frame, text, font, corner_radius, fg_color, text_color, height, x, y):
        label = CTkLabel(frame, text=text, font=font, corner_radius=corner_radius, fg_color=fg_color,
                         text_color=text_color, height=height)
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

    def show_result(self, relay: int, mode: str):
        """
        Displays the result on the GUI.

        Parameters:
            relay (int): The relay number, either 1 or 2. For the 'empty' mode, you can set it to 3 for both relays.
            mode (str): The mode of operation - 'good', 'defect', or 'empty'.

        Returns:
            None
        """
        if mode == "empty":
            if relay == 3:
                # Hide both result widgets
                for id in range[2]:
                    self.result_bg[id].place(x=-1000, y=-1000)
                    self.result_text[id].place(x=-1000, y=-1000)
                    # self.result_bg[1].place(x=-1000, y=-1000)
                    # self.result_text[1].place(x=-1000, y=-1000)
            elif relay == 1:
                # Hide the left result widget
                self.result_bg[1].place(x=-1000, y=-1000)
                self.result_text[1].place(x=-1000, y=-1000)
            else:
                # Hide the right result widget
                self.result_bg[0].place(x=-1000, y=-1000)
                self.result_text[0].place(x=-1000, y=-1000)
        else:
            result_index = 0 if relay == 2 else 1
            bg_color = "#158237" if mode == "good" else "#BB1D2A"
            text = "Good" if mode == "good" else "NG "

            # Update the background and text for the result widget
            self.result_bg[result_index].config(background=bg_color, text="")
            self.result_text[result_index].config(background=bg_color, text=text)

            # Place the widgets at the correct positions

            self.result_bg[result_index].place(x=self.result_pos_bg[result_index][0],
                                               y=self.result_pos_bg[result_index][1])
            self.result_text[result_index].place(x=self.result_pos_text[result_index][0],
                                                 y=self.result_pos_text[result_index][1])
        # for camera_id in self.ai_engines:
        #     counter = self.ai_engines[camera_id].counter
        #     print(camera_id)
        #     print(counter)

        if relay == 2:
            self.update_counter_result(1)
        else:
            self.update_counter_result(0)

        # else:
        #
        #     counter = self.ai_engines[0].counter
        #     self.label_exposure_camera2_ok_value.config(text=str(counter.num_good))
        #     self.label_exposure_camera2_ng_value.config(text=str(counter.num_defect))
        #     self.label_exposure_camera2_total_value.config(text=str(counter.total))

        # print("Camera id", camera_id)
        # print("Number of good", counter.num_good)
        # print("Number of defect", counter.num_defect)

        # Counter
        # if relay == 1:
        #     if mode == "good":
        #         self.ok_counter_camera1 += 1
        #     elif mode == "defect":
        #         self.ng_counter_camera1 += 1
        #     self.label_exposure_camera1_ok_value.config(text=str(self.ok_counter_camera1))
        #     self.label_exposure_camera1_ng_value.config(text=str(self.ng_counter_camera1))
        #     total_camera_1 = self.ok_counter_camera1 + self.ng_counter_camera1
        #     self.label_exposure_camera1_total_value.config(text=str(total_camera_1))
        # elif relay == 2:
        #     if mode == "good":
        #         self.ok_counter_camera2 += 1
        #     elif mode == "defect":
        #         self.ng_counter_camera2 += 1
        #     self.label_exposure_camera1_ok_value.config(text=str(self.ok_counter_camera2))
        #     self.label_exposure_camera2_ng_value.config(text=str(self.ng_counter_camera2))
        #     total_camera_2 = self.ok_counter_camera1 + self.ng_counter_camera2
        #     self.label_exposure_camera2_total_value(text=str(total_camera_2))

    def setting(self):
        # Membuat pop-up window
        self.popup = tk.Toplevel(self.window)
        self.popup.title("Settings")
        self.popup.iconbitmap("satnusa.ico")

        popup_width = 300
        popup_height = 200

        window_x = self.window.winfo_rootx()
        window_y = self.window.winfo_rooty()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        x = window_x + (window_width - popup_width) // 2
        y = window_y + (window_height - popup_height) // 2

        self.popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

        frame = tk.Frame(self.popup)
        frame.pack(fill="both", padx=10, pady=10)

        self.save_camera_var = tk.BooleanVar()
        self.save_camera_checkbox = CTkCheckBox(master=frame,
                                                text="Save Camera",
                                                variable=self.save_camera_var,
                                                onvalue=True,
                                                offvalue=False)
        self.save_camera_checkbox.pack(anchor="w", padx=5, pady=5)
        self.save_image_inspection_var = tk.BooleanVar()
        self.save_image_inspection_checkbox = CTkCheckBox(master=frame,
                                                          text="Save Image Inspection",
                                                          variable=self.save_image_inspection_var,
                                                          onvalue=True,
                                                          offvalue=False)
        self.save_image_inspection_checkbox.pack(anchor="w", padx=5, pady=5)  #

        browse_frame = tk.Frame(frame)
        browse_frame.pack(anchor="w", padx=5, pady=5)

        self.browse_button = CTkButton(master=browse_frame,
                                       text="Browse File",
                                       command=self.browse_file, fg_color="red")
        self.browse_button.pack(side="left")

        self.file_label = tk.Label(browse_frame, text="No file chosen", fg="black")
        self.file_label.pack(side="left", padx=(5, 0))

        button_frame = tk.Frame(self.popup)
        button_frame.pack(side="bottom", fill="x", pady=(10, 10))

        self.submit_button = CTkButton(master=button_frame,
                                       text="Submit",
                                       width=1,
                                       command=self.submit,
                                       fg_color="red",
                                       text_color="white")
        self.submit_button.pack(side="right", padx=(5, 10))

        self.cancel_button = CTkButton(master=button_frame,
                                       text="Cancel",
                                       width=1,
                                       command=self.popup.destroy,
                                       fg_color="transparent",
                                       text_color="red")
        self.cancel_button.pack(side="right", padx=5)

    def submit(self):
        print("Submit clicked")
        self.popup.destroy()

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select a File",
                                               filetypes=(("Image Files", "*.jpg*"),
                                                          ("Image Files", "*.png*")))
        if file_path:
            self.file_label.config(text=file_path)
            print(f"Selected file: {file_path}")

    # def event_closing(self, restart=False, message=None):
    #     if self.btn_start_stop_grabbing_is_grabbing:

    #         # Return True when in condition stop grabbing
    #         close = self.start_stop_grabbing()
    #         if close:
    #             self.event_closing()
    #     else:
    #         if restart:
    #             if message is None:
    #                 message = "Do you want to restart the application?"

    #             if tkinter.messagebox.askokcancel(f'Close | {self.window_title}', message):
    #                 self.close_app()
    #                 logger.info("Restart the App")
    #                 os.execl(sys.executable, sys.executable, *sys.argv)
    #         else:
    #             if message is None:
    #                 message = "Do you want to close the application?"

    #             if tkinter.messagebox.askokcancel(f'Close | {self.window_title}', message):
    #                 self.close_app()
    #                 logger.info("Close the App")

    def event_closing(self, restart=False, message=None):
        if any(self.btn_start_stop_grabbing_is_grabbing):  # Memeriksa apakah ada kamera yang sedang grabbing
            # Hentikan grabbing untuk semua kamera
            for camera_index in range(len(self.btn_start_stop_grabbing_is_grabbing)):
                if self.btn_start_stop_grabbing_is_grabbing[camera_index]:
                    self.stop_grabbing(camera_index)

            # Tunggu sampai semua kamera berhenti grabbing sebelum menutup aplikasi
            self.window.after(100, lambda: self.event_closing(restart, message))
        else:
            if restart:
                if message is None:
                    message = "Do you want to restart the application?"

                if tkinter.messagebox.askokcancel(f'Close | {self.window_title}', message):
                    self.close_app()
                    logger.info("Restart the App")
                    os.execl(sys.executable, sys.executable, *sys.argv)
            else:
                if message is None:
                    message = "Do you want to close the application?"

                if tkinter.messagebox.askokcancel(f'Close | {self.window_title}', message):
                    self.close_app()
                logger.info("Close the App")

    def close_app(self):
        if self.b_is_run:
            self.b_is_run = False
            if self.btn_open_close_device_is_open:
                self.close_device()
        self.stop_ai_engines()
        self.window.destroy()

    def enum_devices(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if ret != 0:
            logger.error(f'Enum cameras fail! ret = "{self.to_hex_str(ret)}"')
            tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                         f'Enum cameras fail! ret = "{self.to_hex_str(ret)}"')
            return

        self.label_total_devices.config(text=f'Number of Camera : {self.deviceList.nDeviceNum}')

        if self.deviceList.nDeviceNum == 0:
            logger.info('Find no camera!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Find no camera!')
            return
        else:
            logger.info("Find %d cameras!" % self.deviceList.nDeviceNum)

        self.devList = []
        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                logger.info("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                logger.info("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                self.devList.append(
                    "Gige[" + str(i) + "]:" + str(nip1) + "." + str(nip2) + "." + str(nip3) + "." + str(nip4))
            elif mvcc_dev_info.nself.tlayerType == MV_USB_DEVICE:
                logger.info("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                logger.info("user serial number: %s" % strSerialNumber)
                self.devList.append("USB[" + str(i) + "]" + str(strSerialNumber))

    def open_close_device(self):
        if self.deviceList.nDeviceNum == 0 and not DEBUG:
            logger.info('Find no camera!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Find no camera!')
        else:
            if self.btn_open_close_device_is_open:
                self.open_device()
                self.btn_open_close_device.configure(text="Close Camera", fg_color="#CD202E")
                self.btn_open_close_device_is_open = False
            else:
                self.close_device()
                self.btn_open_close_device.configure(text="Open Camera", fg_color="#004040")
                self.btn_open_close_device_is_open = True

    def open_device(self):
        if DEBUG:
            self._dummy_open_device()
        else:
            self._open_device()

        self.start_ai_engines()

    def _open_device(self):
        self.nOpenDevSuccess = 0
        if self.b_is_run:
            logger.info('Camera is Running!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Camera is Running!')
            return

        for i in range(0, self.deviceList.nDeviceNum):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            # if self.number_camera_defined == 1 and i == 0:
            #     continue

            camObj = MvCamera()
            strName = str(self.devList[i])
            engine = self.ai_engines.get(i)

            if engine is not None:
                engine.set_show_warning_relay(self.show_warning_relay)

            self.obj_cam_operation.append(CameraOperation(camObj, engine, self.deviceList, i))
            ret = self.obj_cam_operation[self.nOpenDevSuccess].Open_device()
            if 0 != ret:
                self.obj_cam_operation.pop()
                logger.error("open cam %d fail ret[0x%x]" % (i, ret))
                continue
            else:
                logger.info(str(self.devList[i]))
                self.nOpenDevSuccess = self.nOpenDevSuccess + 1
                logger.info(f"self.nOpenDevSuccess = {self.nOpenDevSuccess}")
                self.b_is_run = True
            if 4 == self.nOpenDevSuccess:
                break

    def _dummy_open_device(self):
        self.nOpenDevSuccess = 2
        if self.b_is_run:
            logger.info('Camera is Running!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Camera is Running!')
            return
        self.obj_cam_operation = []
        for i in range(2):
            engine = self.ai_engines.get(i)

            if engine is not None:
                engine.set_show_warning_relay(self.show_warning_relay)

            self.obj_cam_operation.append(CameraOperation(None, engine, self.deviceList, i, dummy=True))

    def show_warning_relay(self):
        logger.error("The blower is not working. Please check the connection.")
        tkinter.messagebox.showerror(
            f'Error | {self.window_title}',
            "The blower is not working. Please check the connection!."
        )

    def start_stop_grabbing(self, camera_index):
        def wait_until_model_ready():
            self.btn_start_stop_grabbing_buttons[camera_index].configure(text="Starting...", fg_color="#c3c3c3",
                                                                         text_color="black")
            # self.btn_start_stop_grabbing_buttons[camera_index].configure(text="Starting...", fg_color="#c3c3c3",
            #                                                   text_color="black")

            while True:
                if self.ai_engines_started:
                    self.start_grabbing(camera_index)
                    break
                time.sleep(0.1)

        if self.btn_open_close_device_is_open:
            logger.info('Please open the camera first!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please open the camera first!')
        else:
            if self.nOpenDevSuccess == self.number_camera_defined or DEBUG:
                if self.btn_start_stop_grabbing_is_grabbing[camera_index]:
                    if tkinter.messagebox.askokcancel(f'Close | {self.window_title}',
                                                      "Do you want to Stop Inspection?"):
                        self.stop_grabbing(camera_index)
                        return True
                    else:
                        return False
                else:
                    if self.ai_engines_started:
                        self.start_grabbing(camera_index)
                    else:
                        threading.Thread(target=wait_until_model_ready).start()
            else:
                logger.info(f'Found only {self.nOpenDevSuccess} device!')
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', f'Found only {self.nOpenDevSuccess} device!')

    def start_grabbing(self, camera_index):
        if not self.btn_start_stop_grabbing_is_grabbing[camera_index]:
            self.btn_start_stop_grabbing_is_grabbing[camera_index] = True
            self.inspection_status_frames[camera_index].configure(text="RUNNING", fg_color="#1DB74E", width=50)

            self.windows_is_on_resize(event=None)  # Memastikan ukuran jendela benar
            for i, obj_cam in enumerate(self.obj_cam_operation):
                obj_cam.set_frame_size(self.cam_frame_size[0], self.cam_frame_size[1])

            lock = threading.Lock()
            ret = 0
            # print(self.obj_cam_operation)

            if camera_index == 0:
                # camera_index = 0
                self.label_image2 = tk.Label(self.frame2, image=self.obj_cam_operation[camera_index].current_frame)
                self.label_image2.place(x=self.label_image_pos[0], y=self.label_image_pos[1])
                ret = self.obj_cam_operation[camera_index].Start_grabbing(camera_index, self.frame2, self.label_image2,
                                                                          lock, self.show_result)
            elif camera_index == 1:
                # camera_index = 1
                self.label_image1 = tk.Label(self.frame1, image=self.obj_cam_operation[camera_index].current_frame)
                self.label_image1.place(x=self.label_image_pos[0], y=self.label_image_pos[1])
                ret = self.obj_cam_operation[camera_index].Start_grabbing(camera_index, self.frame1, self.label_image1,
                                                                          lock, self.show_result)

            if ret != 0:
                logger.error(f'Camera: {camera_index}, start grabbing fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {camera_index}, start grabbing fail! ret = "{self.to_hex_str(ret)}"')
                self.btn_start_stop_grabbing_is_grabbing[camera_index] = False
            else:
                logger.info(f'Camera: {camera_index} started grabbing successfully.')
                self.btn_start_stop_grabbing_buttons[camera_index].configure(text="Stop Inspection", fg_color="#CD202E",
                                                                             text_color="white", width=100)
                self.btn_start_stop_grabbing_is_ever_grabbing = True

    # def stop_grabbing(self, camera_index):
    #     if self.btn_start_stop_grabbing_is_grabbing[camera_index]:
    #         self.btn_start_stop_grabbing_is_grabbing[camera_index] = False
    #         self.inspection_status_frames[camera_index].configure(text="STOP", fg_color="#BB1D2A")

    #         ret = self.obj_cam_operation[camera_index].Stop_grabbing()
    #         if ret != 0:
    #             logger.error(f'Camera: {camera_index}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
    #             tkinter.messagebox.showerror(f'Error | {self.window_title}', f'Camera: {camera_index}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
    #         else:
    #             logger.info(f'Camera: {camera_index} stopped grabbing successfully.')

    #         self.btn_start_stop_grabbing_buttons[camera_index].configure(text="Start Inspection", fg_color="#1a8a42", text_color="white")
    #         self.btn_start_stop_grabbing_is_ever_grabbing = True

    #         if camera_index == 0:
    #             self.label_image1.place(x=-1000, y=-1000)
    #             self.label_image1.image = None  # Clear image reference
    #         elif camera_index == 1:
    #             self.label_image2.place(x=-1000, y=-1000)
    #             self.label_image2.image = None  # Clear image reference

    def stop_grabbing(self, camera_index):
        if self.btn_start_stop_grabbing_is_grabbing[camera_index]:
            self.btn_start_stop_grabbing_is_grabbing[camera_index] = False
            self.inspection_status_frames[camera_index].configure(text="STOP", fg_color="#BB1D2A")

            ret = self.obj_cam_operation[camera_index].Stop_grabbing()
            if ret != 0:
                logger.error(f'Camera: {camera_index}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {camera_index}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
            else:
                logger.info(f'Camera: {camera_index} stopped grabbing successfully.')

            self.btn_start_stop_grabbing_buttons[camera_index].configure(text="Start Inspection", fg_color="#1a8a42",
                                                                         text_color="white")
            self.btn_start_stop_grabbing_is_ever_grabbing = True

            # Reset label image dan hapus referensi gambar
            if camera_index == 1:
                self.label_image1.place_forget()  # Hilangkan label dari tampilan
                self.label_image1.image = None  # Hapus referensi gambar
                self.label_image1 = None  # Hapus label
            elif camera_index == 0:
                self.label_image2.place_forget()  # Hilangkan label dari tampilan
                self.label_image2.image = None  # Hapus referensi gambar
                self.label_image2 = None  # Hapus label

    def close_device(self):
        if any(self.btn_start_stop_grabbing_is_grabbing):
            # Hentikan grabbing untuk semua kamera yang sedang aktif
            for i in range(self.nOpenDevSuccess):
                if self.btn_start_stop_grabbing_is_grabbing[i]:
                    self.stop_grabbing(i)  # Hentikan grabbing untuk kamera i

        # Hentikan AI engines jika diperlukan
        self.stop_ai_engines()

        # Tutup setiap perangkat kamera
        for i in range(self.nOpenDevSuccess):
            # if self.number_camera_defined == 1 and i == 0:
            #     continue  # Skip if only one camera is defined and it's the top camera

            ret = self.obj_cam_operation[i].Close_device()
            if ret != 0:
                logger.error(f'Camera: {str(i)}, close camera fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {str(i)}, close camera fail! ret = "{self.to_hex_str(ret)}"')
                self.b_is_run = True
                return

        self.b_is_run = False
        logger.info("Camera close ok")

    def reset_counter(self, camera_index):
        if camera_index == 1:
            counter = self.ai_engines[1].counter
            self.on_input_percentage_change(1, None)
        else:
            counter = self.ai_engines[0].counter
            self.on_input_percentage_change(0, None)
        counter.clear()

    def set_parameter_new(self, cam: int):
        # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
        # if self.number_camera_defined == 1 and cam == 0:
        #     logger.info(f'Camera {cam} [side] is set to not open!')
        #     tkinter.messagebox.showinfo(f'Info | {self.window_title}', f'Camera {cam} [side] is set to not open!')
        #
        # else:
        if self.btn_open_close_device_is_open:
            logger.info('Please open the camera first!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please open the camera first!')
        else:
            str_num = self.text_exposure_time[cam].get()
            self.default_exposure_time[cam] = int(str_num if str_num else self.default_exposure_time[cam])
            self.label_current_exposure_value[cam].configure(
                text=f"Current Exposure Value\t: {self.default_exposure_time[cam]}")
            self.set_parameter_auto(cam, self.default_exposure_time[cam], self.default_gain[cam],
                                    self.default_frame_rate[cam])

    def set_parameter_auto(self, cam_num, default_exposure_time, default_gain, default_frame_rate):
        i = cam_num
        self.obj_cam_operation[i].exposure_time = default_exposure_time
        self.obj_cam_operation[i].gain = default_gain
        self.obj_cam_operation[i].frame_rate = default_frame_rate
        ret = self.obj_cam_operation[i].Set_parameter(self.obj_cam_operation[i].frame_rate,
                                                      self.obj_cam_operation[i].exposure_time,
                                                      self.obj_cam_operation[i].gain)
        if 0 != ret:
            logger.error(f'Camera: {str(i)}, set parameter fail! ret = "{self.to_hex_str(ret)}"')
            tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                         f'Camera: {str(i)}, set parameter fail! ret = "{self.to_hex_str(ret)}"')


if __name__ == "__main__":
    logger.info("Start opening application")
    gui = MainGUI()
    gui.window.mainloop()
