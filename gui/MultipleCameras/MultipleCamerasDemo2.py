import os
import warnings

from anomalib.utils.loggers.logger import setup_logger
from config import DEBUG, RESULT_DIR

logger = setup_logger("AutoInspection", log_dir=RESULT_DIR/"logs")

warnings.filterwarnings('ignore')

# -- coding: utf-8 --
import sys
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk

sys.path.append("../MvImport")
# from MvCameraControl_class import *
from CamOperation_class_v1 import *
from PIL import Image, ImageTk
import cv2
from customtkinter import CTkLabel, CTkButton, CTkEntry

from engine_v2 import AIEngine, start_spawn_method
from relay import Relay


class AIEngineGUI:
    def __init__(self):
        self.camera_ids = {
            # "kamera-samping": 0,
            # "kamera-atas": 1
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
        self.window.minsize(self.window_width, self.window_height)
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

        self.inspection_activity_label = tk.Label(self.window, font=("Roboto", 10), text='Inspection Activity',
                                                  fg="#333")
        self.inspection_activity_label.place(x=308, y=64)

        self.label_total_devices = tk.Label(self.window, font=("Roboto", 8), text='Number of Camera\t\t: None',
                                            fg="#004040")
        self.label_total_devices.place(x=18, y=90)

        self.label_status = tk.Label(self.window, font=("Roboto", 8), text='Status\t:', fg="#004040")
        self.label_status.place(x=308, y=90)

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
                                               text="Open Camera",
                                               fg_color="#1a8a42",
                                               text_color="white",
                                               font=("Roboto", 12),
                                               command=self.open_close_device)
        self.btn_open_close_device.place(x=151, y=120)

        self.btn_start_stop_grabbing_style = ttk.Style()
        self.btn_start_stop_grabbing_style.configure('OpenClose.TButton')
        self.btn_start_stop_grabbing_is_grabbing = False
        self.btn_start_stop_grabbing_is_ever_grabbing = False
        self.btn_start_stop_grabbing = CTkButton(master=self.window,
                                                 width=120,
                                                 height=30,
                                                 border_width=0,
                                                 corner_radius=4,
                                                 text="Start Inspection",
                                                 fg_color="#1a8a42",
                                                 text_color="white",
                                                 font=("Roboto", 12),
                                                 command=self.start_stop_grabbing)
        # self.btn_start_stop_grabbing.config(state="disable")
        self.btn_start_stop_grabbing.place(x=311, y=120)

        self.btn_show_inspection_result = CTkButton(master=self.window,
                                                    width=120,
                                                    height=30,
                                                    border_width=0,
                                                    corner_radius=4,
                                                    text="Show Results",
                                                    fg_color="#c3c3c3",
                                                    text_color="black",
                                                    font=("Roboto", 12),
                                                    command=self.show_folder_result)
        self.btn_show_inspection_result.place(x=442, y=120)

        self.btn_full_screen = CTkButton(master=self.window,
                                                    width=120,
                                                    height=30,
                                                    border_width=0,
                                                    corner_radius=4,
                                                    text="Full Screen",
                                                    fg_color="#c3c3c3",
                                                    text_color="black",
                                                    font=("Roboto", 12),
                                                    command=self.full_screen)
        self.btn_full_screen.place(x=573, y=120)

        self.window_is_full_screen = False

        # Camera
        self.number_camera_defined = 1  # Set default value -> 2

        # Create two bordered frames
        self.frame1 = tk.Frame(self.window, width=500, height=470)
        self.frame1.place(x=541, y=188)

        self.frame1_vertical_frame_left = tk.Frame(self.frame1, width=1, height=470, relief="solid", bg="#A0A0A0")
        self.frame1_vertical_frame_left.place(x=0, y=0)

        self.frame1_vertical_frame_right = tk.Frame(self.frame1, width=1, height=470, relief="solid", bg="#A0A0A0")
        self.frame1_vertical_frame_right.place(x=500 - 1, y=0)

        self.frame1_horizontal_frame_top = tk.Frame(self.frame1, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame1_horizontal_frame_top.place(x=0, y=0)

        self.frame1_horizontal_frame_bottom = tk.Frame(self.frame1, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame1_horizontal_frame_bottom.place(x=0, y=470 - 1)

        self.frame2 = tk.Frame(self.window, width=500, height=470)
        self.frame2.place(x=21, y=188)

        self.frame2_vertical_frame_left = tk.Frame(self.frame2, width=1, height=470, relief="solid", bg="#A0A0A0")
        self.frame2_vertical_frame_left.place(x=0, y=0)

        self.frame2_vertical_frame_right = tk.Frame(self.frame2, width=1, height=470, relief="solid", bg="#A0A0A0")
        self.frame2_vertical_frame_right.place(x=500 - 1, y=0)

        self.frame2_horizontal_frame_top = tk.Frame(self.frame2, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame2_horizontal_frame_top.place(x=0, y=0)

        self.frame2_horizontal_frame_bottom = tk.Frame(self.frame2, width=500, height=1, relief="solid", bg="#A0A0A0")
        self.frame2_horizontal_frame_bottom.place(x=0, y=470 - 1)

        self.label_image1 = tk.Label(self.frame1)
        self.label_image2 = tk.Label(self.frame2)
        self.label_image_pos_default = (7, 102)
        self.label_image_pos = [self.label_image_pos_default[0], self.label_image_pos_default[1]]

        self.img_no_camera = tk.PhotoImage(file="no_image.png").subsample(4, 4)  # Menyesuaikan ukuran gambar menjadi 50x50
        self.no_camera_label1 = tk.Label(self.frame1, image=self.img_no_camera)
        self.no_camera_label1.place(x=115, y=130)

        self.no_camera_label2 = tk.Label(self.frame2, image=self.img_no_camera)
        self.no_camera_label2.place(x=115, y=130)

        self.camera_status = [
            tk.Label(self.frame1, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold")),
            tk.Label(self.frame2, foreground='#004040', text='No Camera Scan', font=("Roboto", 16, "bold"))]

        self.camera_status[0].place(x=160, y=380)
        self.camera_status[1].place(x=160, y=380)

        self.camera_label_side = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera Side Inspection ',
                                          fg="#333")
        self.camera_label_side.place(x=551, y=175)
        self.camera_label_top = tk.Label(self.window, font=("Roboto", 12, "bold"), text='Camera Top Inspection ',
                                         fg="#333")
        self.camera_label_top.place(x=30, y=175)
        

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

        # camera_inspection_status_label
        self.camera_inspection_status_label_canvas_1 = tk.Canvas(self.frame1, background=None, highlightthickness=0, width=490, height=45)
        self.camera_inspection_status_label_canvas_1.place(relx=0.5, rely=0.175, anchor="center")
        self.camera_inspection_status_label_canvas_1_rectangle = self.camera_inspection_status_label_canvas_1.create_rectangle(3, 3, 485, 37, dash=(5, 1), outline="#1DB74E", fill="")
        self.camera_inspection_status_label_label_1 = tk.Label(self.camera_inspection_status_label_canvas_1, bg=None, fg="#004040", text='', font=("Roboto", 12, "bold"))
        self.camera_inspection_status_label_label_1.place(relx=0.5, rely=0.5, anchor="center")

        self.camera_inspection_status_label_canvas_2 = tk.Canvas(self.frame2, background=None, highlightthickness=0, width=490, height=45)
        self.camera_inspection_status_label_canvas_2.place(relx=0.5, rely=0.175, anchor="center")
        self.camera_inspection_status_label_canvas_2_rectangle = self.camera_inspection_status_label_canvas_2.create_rectangle(3, 3, 485, 37, dash=(5, 1), outline="#1DB74E", fill="")
        self.camera_inspection_status_label_label_2 = tk.Label(self.camera_inspection_status_label_canvas_2, bg=None, fg="#004040", text='', font=("Roboto", 12, "bold"))
        self.camera_inspection_status_label_label_2.place(relx=0.5, rely=0.5, anchor="center")

        self.result_bg = [
            tk.Label(self.frame1, background="#1DB74E", text='', font=("Roboto", 16, "bold"), padx=240, pady=6),
            tk.Label(self.frame2, background="#1DB74E", text='', font=("Roboto", 16, "bold"), padx=240, pady=6),
        ]

        self.result_text = [
            tk.Label(self.frame1, background="#1DB74E", text='', font=("Roboto", 16, "bold"), foreground='white'),
            tk.Label(self.frame2, background="#1DB74E", text='', font=("Roboto", 16, "bold"), foreground='white')
        ]

        self.result_pos_bg = ((8, 60), (8, 60))
        self.result_pos_text = [[210, 65], [230, 65]]

        self.inspection_status = self.inspection_status_label(text="STOP ", font=("Roboto", 10),
                                                              corner_radius=4, fg_color="#CD202E",
                                                              text_color="white", height=20,
                                                              x=370, y=90)

        self.show_result(2, "defect")

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

        half_delta_width = (delta_width//2)
        quarter_delta_width = (delta_width//4)
        delta_width_ = (delta_width//7)
        half_delta_height = (delta_height//2)

        # Header
        self.header_line.configure(width=145+delta_width_)
        self.header_label_thinner.configure(width=145+delta_width_)

        # Frame
        self.frame1.configure(width=500+half_delta_width, height=470+delta_height)
        self.frame1.place(x=541+half_delta_width, y=188)
        self.frame2.configure(width=500+half_delta_width, height=470+delta_height)
        self.camera_label_side.place(x=551+half_delta_width, y=175)

        self.frame1_vertical_frame_left.configure(height=470+delta_height)
        self.frame1_vertical_frame_right.configure(height=470+delta_height)
        self.frame1_vertical_frame_right.place(x=500+half_delta_width-1, y=0)
        self.frame1_horizontal_frame_top.configure(width=500+half_delta_width)
        self.frame1_horizontal_frame_bottom.configure(width=500+half_delta_width)
        self.frame1_horizontal_frame_bottom.place(x=0, y=470-1+delta_height)

        self.frame2_vertical_frame_left.configure(height=470+delta_height)
        self.frame2_vertical_frame_right.configure(height=470+delta_height)
        self.frame2_vertical_frame_right.place(x=500+half_delta_width-1, y=0)
        self.frame2_horizontal_frame_top.configure(width=500+half_delta_width)
        self.frame2_horizontal_frame_bottom.configure(width=500+half_delta_width)
        self.frame2_horizontal_frame_bottom.place(x=0, y=470-1+delta_height)

        self.no_camera_label1.place(x=115+quarter_delta_width, y=130+half_delta_height)
        self.no_camera_label2.place(x=115+quarter_delta_width, y=130+half_delta_height)
        self.camera_status[0].place(x=160+quarter_delta_width, y=380+half_delta_height)
        self.camera_status[1].place(x=160+quarter_delta_width, y=380+half_delta_height)

        # Inspection result
        self.camera_inspection_status_label_canvas_1.configure(width=490+half_delta_width)
        self.camera_inspection_status_label_canvas_1.coords(self.camera_inspection_status_label_canvas_1_rectangle, 3, 3, 485+half_delta_width, 37)
        self.camera_inspection_status_label_canvas_1.place(relx=0.5, rely=0.175-(delta_height/4300), anchor="center")

        self.camera_inspection_status_label_canvas_2.configure(width=490+half_delta_width)
        self.camera_inspection_status_label_canvas_2.coords(self.camera_inspection_status_label_canvas_2_rectangle, 3, 3, 485+half_delta_width, 37)
        self.camera_inspection_status_label_canvas_2.place(relx=0.5, rely=0.175-(delta_height/4300), anchor="center")

        # Images
        self.result_pos_text = [[210 + quarter_delta_width, 65], [230 + quarter_delta_width, 65]]

        self.result_bg[0].configure(padx=240 + quarter_delta_width)
        self.result_bg[1].configure(padx=240 + quarter_delta_width)

        cam_delta_width = self.cam_frame_size_default[0]  # Default val from CamOperation cam_delta_width = 482 and cam_delta_height = 363
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
            self.label_image1.place(x=x_label_image, y=y_label_image)
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

    def inspection_status_label(self, text, font, corner_radius, fg_color, text_color, height, x, y):
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

    def show_result(self, relay: int, mode: str):
        """
        Displays the result on the GUI.

        Parameters:
            relay (int): The relay number, either 1 or 2. For the 'empty' mode, you can set it to 3 for both relays.
            mode (str): The mode of operation - 'good', 'defect', or 'empty'.

        Returns:
            None

        Description:
            This method updates the GUI to display the result for a given relay and mode.
            - For 'empty' mode:
                - If relay is 3, hides both result widgets.
                - If relay is 1, hides the left result widget.
                - If relay is 2, hides the right result widget.
            - For 'good' mode:
                - If relay is 1, displays 'Good' with a green background in the left result widget.
                - If relay is 2, displays 'Good' with a green background in the right result widget.
            - For 'defect' mode:
                - If relay is 1, displays 'Defect' with a red background in the left result widget.
                - If relay is 2, displays 'Defect' with a red background in the right result widget.
        """

        if mode == "empty":
            if relay == 3:
                self.result_bg[0].place(x=-1000, y=-1000)
                self.result_text[0].place(x=-1000, y=-1000)
                self.result_bg[1].place(x=-1000, y=-1000)
                self.result_text[1].place(x=-1000, y=-1000)
            elif relay == 1:
                self.result_bg[0].place(x=-1000, y=-1000)
                self.result_text[0].place(x=-1000, y=-1000)
            else:
                self.result_bg[1].place(x=-1000, y=-1000)
                self.result_text[1].place(x=-1000, y=-1000)
        else:
            result_index = 0 if relay == 1 else 1
            bg_color = "#158237" if mode == "good" else "#BB1D2A"
            text = "Good" if mode == "good" else "  NG "

            self.result_bg[result_index].config(background=bg_color, text="")
            self.result_text[result_index].config(background=bg_color, text=text)
            self.result_bg[result_index].place(x=self.result_pos_bg[result_index][0],
                                               y=self.result_pos_bg[result_index][1])
            self.result_text[result_index].place(x=self.result_pos_text[result_index][0],
                                                 y=self.result_pos_text[result_index][1])

    def event_closing(self, restart=False, message=None):
        if self.btn_start_stop_grabbing_is_grabbing:

            # Return True when in condition stop grabbing
            close = self.start_stop_grabbing()
            if close:
                self.event_closing()
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
            tkinter.messagebox.showerror(f'Error | {self.window_title}', f'Enum cameras fail! ret = "{self.to_hex_str(ret)}"')
            return

        self.label_total_devices.config(text=f'Number of Camera           : {self.deviceList.nDeviceNum}')

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
            if self.number_camera_defined == 1 and i == 0:
                continue

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

    def start_stop_grabbing(self):
        def wait_until_model_ready():
            self.btn_start_stop_grabbing.configure(text="Starting...", fg_color="#c3c3c3", text_color="black")

            while True:
                if self.ai_engines_started:
                    self.start_grabbing()
                    # self.btn_start_stop_grabbing.configure(text="Stop Inspection", fg_color="#CD202E", text_color="white")
                    # self.btn_start_stop_grabbing_is_grabbing = True
                    # self.btn_start_stop_grabbing_is_ever_grabbing = True
                    break
                time.sleep(0.1)

        if self.btn_open_close_device_is_open:
            logger.info('Please open the camera first!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', 'Please open the camera first!')
        else:
            if self.nOpenDevSuccess == self.number_camera_defined or DEBUG:
                # if self.btn_start_stop_grabbing_is_ever_grabbing:
                #     self.event_closing(restart=True, message="Stopping inspection means you have to restart. Want to continue?")
                if self.btn_start_stop_grabbing_is_grabbing:
                    if tkinter.messagebox.askokcancel(f'Close | {self.window_title}',
                                                      "Do you want to Stop Inspection?"):
                        self.stop_grabbing()
                        # Return True when in condition stop grabbing
                        return True
                    else:
                        return False
                else:
                    if self.ai_engines_started:
                        self.start_grabbing()
                        # self.btn_start_stop_grabbing.configure(text="Stop Inspection", fg_color="#CD202E", text_color="white")
                        # self.btn_start_stop_grabbing_is_grabbing = True
                        # self.btn_start_stop_grabbing_is_ever_grabbing = True
                    else:
                        threading.Thread(target=lambda: wait_until_model_ready()).start()
                        # tkinter.messagebox.showinfo(f'Info | {self.window_title}',
                        #                             "Please wait! The AI model is being built and is not ready yet.")
            else:
                logger.info(f'Found only {self.nOpenDevSuccess} device!')
                tkinter.messagebox.showinfo(f'Info | {self.window_title}', f'Found only {self.nOpenDevSuccess} device!')

    def start_grabbing(self):
        self.windows_is_on_resize(event=None)
        for i, obj_cam in enumerate(self.obj_cam_operation):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            if self.number_camera_defined == 1 and i == 0:
                continue

            obj_cam.set_frame_size(self.cam_frame_size[0], self.cam_frame_size[1])

        for i, device in enumerate(range(0, self.nOpenDevSuccess)):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            if self.number_camera_defined == 1 and i == 0:
                continue

            self.set_parameter_auto(device, self.default_exposure_time[device], self.default_gain[device],
                                    self.default_frame_rate[device])
            self.label_current_exposure_value[device].configure(
                text=f"Current Exposure Value\t: {self.default_exposure_time[device]}")
        lock = threading.Lock()  # 申请一把锁
        ret = 0
        for i in range(0, self.nOpenDevSuccess):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            if self.number_camera_defined == 1 and i == 0:
                continue

            if 0 == i:
                self.label_image1 = tk.Label(self.frame1, image=self.obj_cam_operation[i].current_frame)
                self.label_image1.place(x=self.label_image_pos[0], y=self.label_image_pos[1])
                ret = self.obj_cam_operation[i].Start_grabbing(
                    i, self.frame1, self.label_image1, lock,  self.show_result
                )
            elif 1 == i:
                self.label_image2 = tk.Label(self.frame2, image=self.obj_cam_operation[i].current_frame)
                self.label_image2.place(x=self.label_image_pos[0], y=self.label_image_pos[1])
                ret = self.obj_cam_operation[i].Start_grabbing(
                    i, self.frame2, self.label_image2, lock,  self.show_result
                )
            # elif 2 == i:
            #     ret = self.obj_cam_operation[i].Start_grabbing(i, self.window, None, lock, None)
            # elif 3 == i:
            #     ret = self.obj_cam_operation[i].Start_grabbing(i, self.window, None, lock, None)
            if 0 != ret:
                logger.error(f'Camera: {str(i)}, start grabbing fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {str(i)}, start grabbing fail! ret = "{self.to_hex_str(ret)}"')

            else:
                self.inspection_status.configure(text="RUNNING ", fg_color="#1DB74E")

        if ret == 0:
            self.btn_start_stop_grabbing.configure(text="Stop Inspection", fg_color="#CD202E", text_color="white")
            self.btn_start_stop_grabbing_is_grabbing = True
            self.btn_start_stop_grabbing_is_ever_grabbing = True

    def stop_grabbing(self):
        for i in range(0, self.nOpenDevSuccess):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            if self.number_camera_defined == 1 and i == 0:
                continue

            self.label_current_exposure_value[i].configure(text="Current Exposure Value\t: None")
            ret = self.obj_cam_operation[i].Stop_grabbing()
            if 0 != ret:
                logger.error(f'Camera: {str(i)}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {str(i)}, stop grabbing fail! ret = "{self.to_hex_str(ret)}"')
        logger.info("cam stop grab ok ")

        self.inspection_status.configure(text="STOP ", fg_color="#BB1D2A")
        self.btn_start_stop_grabbing.configure(text="Start Inspection", fg_color="#1a8a42", text_color="white")
        self.btn_start_stop_grabbing_is_grabbing = False
        self.btn_start_stop_grabbing_is_ever_grabbing = True

        self.label_image1.place(x=-1000, y=-1000)
        self.label_image2.place(x=-1000, y=-1000)

    def close_device(self):
        if self.btn_start_stop_grabbing_is_grabbing:
            self.start_stop_grabbing()

        self.stop_ai_engines()
        for i in range(0, self.nOpenDevSuccess):

            # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
            if self.number_camera_defined == 1 and i == 0:
                continue

            ret = self.obj_cam_operation[i].Close_device()
            if 0 != ret:
                logger.error(f'Camera: {str(i)}, close camera fail! ret = "{self.to_hex_str(ret)}"')
                tkinter.messagebox.showerror(f'Error | {self.window_title}',
                                             f'Camera: {str(i)}, close camera fail! ret = "{self.to_hex_str(ret)}"')
                self.b_is_run = True
                return
        self.b_is_run = False
        logger.info("cam close ok ")

    def set_parameter_new(self, cam: int):
        # Assuming the top camera is at index 1, the first looping set continue when self.number_camera_fine is 1
        if self.number_camera_defined == 1 and cam == 0:
            logger.info(f'Camera {cam} [side] is set to not open!')
            tkinter.messagebox.showinfo(f'Info | {self.window_title}', f'Camera {cam} [side] is set to not open!')

        else:
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
