# -- coding: utf-8 --
import sys
from tkinter import * 
from tkinter.messagebox import *
import _tkinter
import tkinter.messagebox
import tkinter as tk
import sys, os
from tkinter import ttk

from CamOperationRGB_class import *
from MvCameraControl_class import *

from PIL import Image,ImageTk


class CameraApp:
    def __init__(self, window):
        # Inisialisasi objek kamera dan variabel lainnya
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.obj_cam_operation = []
        self.b_is_run = False
        self.nOpenDevSuccess = 0
        self.devList = []
        # self.text_number_of_devices = None
        self.text_number_of_devices = tk.Text(window, width=10, height=1)
        self.model_val = tk.StringVar()
        self.triggercheck_val = tk.IntVar()    
        self.panel = None
        self.panel1 = None
        self.panel2 = None
        self.panel3 = None
        self.create_gui(window)

    def create_gui(self, window):
        window.title('MultipleCamerasDemo')
        window.geometry('1330x1020')

        model_val = tk.StringVar()
        triggercheck_val = tk.IntVar()

        page = Frame(window, height=400, width=60)
        page.pack(expand=True, fill=BOTH)

        panel = Label(page)
        panel.place(x=300, y=10, height=500, width=500)

        panel1 = Label(page)
        panel1.place(x=810, y=10, height=500, width=500)

        panel2 = Label(page)
        panel2.place(x=300, y=520, height=500, width=500)

        panel3 = Label(page)
        panel3.place(x=810, y=520, height=500, width=500)

        text_number_of_devices = tk.Text(window, width=10, height=1)
        text_number_of_devices.place(x=200, y=20)
        label_total_devices = tk.Label(window, text='Jumlah kamera:', width=25, height=1)
        label_total_devices.place(x=20, y=20)

        label_exposure_time = tk.Label(window, text='Waktu Paparan', width=15, height=1)
        label_exposure_time.place(x=20, y=350)
        text_exposure_time = tk.Text(window, width=15, height=1)
        text_exposure_time.place(x=160, y=350)

        label_gain = tk.Label(window, text='Penguatan', width=15, height=1)
        label_gain.place(x=20, y=400)
        text_gain = tk.Text(window, width=15, height=1)
        text_gain.place(x=160, y=400)

        label_frame_rate = tk.Label(window, text='Frekuensi Frame', width=15, height=1)
        label_frame_rate.place(x=20, y=450)
        text_frame_rate = tk.Text(window, width=15, height=1)
        text_frame_rate.place(x=160, y=450)

        btn_enum_devices = tk.Button(window, text='Inisialisasi Kamera', width=35, height=1, command=self.enum_devices)
        btn_enum_devices.place(x=20, y=50)

        btn_open_device = tk.Button(window, text='Buka Perangkat', width=15, height=1, command=self.open_device)
        btn_open_device.place(x=20, y=100)

        btn_close_device = tk.Button(window, text='Tutup Perangkat', width=15, height=1, command=self.close_device)
        btn_close_device.place(x=160, y=100)

        # radio_continuous = tk.Radiobutton(window, text='Continuous', variable=model_val, value='continuous', width=15,
        #                                 height=1, command=self.set_triggermode)
        radio_continuous = tk.Radiobutton(window, text='Continuous', variable=model_val, value='continuous', width=15,
                                        height=1, command=self.set_trigger_mode)
        radio_continuous.place(x=20, y=150)

        radio_trigger = tk.Radiobutton(window, text='Mode Pemicu', variable=model_val, value='triggermode', width=15,
                                        height=1, command=self.set_trigger_mode)
        # radio_trigger = tk.Radiobutton(window, text='Mode Pemicu', variable=model_val, value='triggermode', width=15,
        #                             height=1, command=self.set_triggermode)
        radio_trigger.place(x=160, y=150)
        model_val.set(1)

        btn_start_grabbing = tk.Button(window, text='Mulai Mengambil', width=15, height=1, command=self.start_grabbing)
        btn_start_grabbing.place(x=20, y=200)

        btn_stop_grabbing = tk.Button(window, text='Berhenti Mengambil', width=15, height=1, command=self.stop_grabbing)
        btn_stop_grabbing.place(x=160, y=200)

        checkbtn_trigger_software = tk.Checkbutton(window, text='Pemicu oleh Perangkat Lunak', variable=triggercheck_val,
                                                onvalue=1, offvalue=0)
        checkbtn_trigger_software.place(x=20, y=250)

        btn_trigger_once = tk.Button(window, text='Pemicu Sekali', width=15, height=1, command=self.trigger_once)
        btn_trigger_once.place(x=160, y=250)

        btn_get_parameter = tk.Button(window, text='Dapatkan Parameter', width=15, height=1, command=self.get_parameter)
        btn_get_parameter.place(x=20, y=500)

        btn_set_parameter = tk.Button(window, text='Atur Parameter', width=15, height=1, command=self.set_parameter)
        btn_set_parameter.place(x=160, y=500)


    def enum_devices(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)

        if ret != 0:
            showerror('Kesalahan', f'Gagal mengenali perangkat! ret[0x{ret:x}]')
            return

        self.text_number_of_devices.delete(1.0, END)
        self.text_number_of_devices.insert(1.0, str(self.deviceList.nDeviceNum))

        if self.deviceList.nDeviceNum == 0:
            showinfo('Informasi', 'Tidak ditemukan perangkat!')
            return
        else:
            print(f"Menemukan {self.deviceList.nDeviceNum} perangkat!")

        self.devList = []
        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print(f"\ngige device: [{i}]")
                strModeName = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if per != 0)
                print(f"Nama model perangkat: {strModeName}")

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print(f"IP saat ini: {nip1}.{nip2}.{nip3}.{nip4}")
                self.devList.append(f"Gige[{i}]: {nip1}.{nip2}.{nip3}.{nip4}")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print(f"\nu3v device: [{i}]")
                strModeName = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if per != 0)
                print(f"Nama model perangkat: {strModeName}")

                strSerialNumber = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if per != 0)
                print(f"Nomor serial pengguna: {strSerialNumber}")
                self.devList.append(f"USB[{i}]: {strSerialNumber}")

    def open_device(self):
        if self.b_is_run:
            showinfo('Info', 'Kamera sudah berjalan!')
            return

        self.obj_cam_operation = []
        for i in range(0, self.deviceList.nDeviceNum):
            camObj = MvCamera()
            strName = str(self.devList[i])
            camera_operation = CameraOperation(camObj, self.deviceList, i)
            self.obj_cam_operation.append(camera_operation)

            ret = camera_operation.Open_device()
            if ret != 0:
                self.obj_cam_operation.pop()
                showerror('Error', f'Buka kamera {i} gagal! ret = 0x{ret:x}')
            else:
                print(str(self.devList[i]))
                self.nOpenDevSuccess += 1
                self.model_val.set('continuous')
                print("nOpenDevSuccess = ", self.nOpenDevSuccess)
                self.b_is_run = True

            if self.nOpenDevSuccess == 4:
                break

    def start_grabbing(self):
        global nOpenDevSuccess
        lock = threading.Lock()  # Membuat penguncian

        # Iterasi melalui obj_cam_operation
        for i in range(0, self.nOpenDevSuccess):
            panel = None

            # Memilih panel berdasarkan indeks kamera
            if i == 0:
                panel = self.panel
            elif i == 1:
                panel = self.panel1
            elif i == 2:
                panel = self.panel2
            elif i == 3:
                panel = self.panel3

            # Memanggil metode Start_grabbing dari objek kamera yang sesuai
            # ret = self.obj_cam_operation[i].Start_grabbing(i, panel, lock)
            ret = self.obj_cam_operation[i].Start_grabbing(panel, lock)
            
            # Menampilkan pesan kesalahan jika metode gagal
            if ret != 0:
                showerror('Error', f'Kamera: {i}, gagal memulai pengambilan gambar! ret = {To_hex_str(ret)}')

    def stop_grabbing(self):
        global nOpenDevSuccess
        global obj_cam_operation

        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Stop_grabbing()
            if ret != 0:
                tkinter.messagebox.showerror('Tampilkan Kesalahan', f'Kamera: {i} gagal menghentikan pengambilan gambar! ret = {To_hex_str(ret)}')

        print("Pengambilan gambar kamera dihentikan dengan sukses")

    def close_device(self):
        global b_is_run
        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Close_device()
            if ret != 0:
                tkinter.messagebox.showerror('Tampilkan error', 'Kamera: ' + str(i) + ' gagal menutup perangkat! ret = ' + To_hex_str(ret))
                self.b_is_run = True
                return
        self.b_is_run = False
        print("Kamera ditutup dengan baik.")
        
        # Membersihkan nilai teks di kotak teks
        self.text_frame_rate.delete(1.0, tk.END)
        self.text_exposure_time.delete(1.0, tk.END)
        self.text_gain.delete(1.0, tk.END)

    def set_trigger_mode(self):
        global obj_cam_operation
        global nOpenDevSuccess
        strMode = self.model_val.get()  # Perubahan di sini
        for i in range(0, nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Set_trigger_mode(strMode)
            if 0 != ret:
                tkinter.messagebox.showerror('show error', f'camera: {i} set {strMode} fail! ret = {To_hex_str(ret)}')

    def set_trigger_mode(self):
        global obj_cam_operation
        global nOpenDevSuccess
        strMode = self.model_val.get()
        
        for i in range(0, self.nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Set_trigger_mode(strMode)
            if ret != 0:
                tkinter.messagebox.showerror('Kesalahan', f'Kamera {i} gagal mengatur mode pemicu! ret = {To_hex_str(ret)}')

    # def set_trigger_mode(self):
    #     global obj_cam_operation
    #     global nOpenDevSuccess
    #     strMode = self.model_val.get()
        
    #     for i in range(0, self.nOpenDevSuccess):
    #         ret = self.obj_cam_operation[i].Set_trigger_mode(strMode)
    #         if ret != 0:
    #             tkinter.messagebox.showerror('Kesalahan', f'Kamera {i} gagal mengatur mode pemicu! ret = {To_hex_str(ret)}')

    def trigger_once(self):
        global triggercheck_val
        global nOpenDevSuccess

        nCommand = triggercheck_val.get()

        for i in range(0, nOpenDevSuccess):
            ret = self.obj_cam_operation[i].Trigger_once(nCommand)
            if ret != 0:
                tkinter.messagebox.showerror('Kesalahan', f'Kamera: {i}, gagal menetapkan trigger sekali! ret = {To_hex_str(ret)}')


    def get_parameter(self):
        try:
            # Mendapatkan parameter dari objek kamera yang pertama (indeks 0)
            if self.nOpenDevSuccess > 0:
                ret = self.obj_cam_operation[0].Get_parameter()
                if ret != 0:
                    showerror('Error', 'Gagal mendapatkan parameter kamera! ret = ' + To_hex_str(ret))
                    return
                # Menampilkan nilai parameter di GUI
                self.text_frame_rate.delete(1.0, END)
                self.text_frame_rate.insert(1.0, self.obj_cam_operation[0].frame_rate)

                self.text_exposure_time.delete(1.0, END)
                self.text_exposure_time.insert(1.0, self.obj_cam_operation[0].exposure_time)

                self.text_gain.delete(1.0, END)
                self.text_gain.insert(1.0, self.obj_cam_operation[0].gain)
            else:
                showinfo('Info', 'Tidak ada kamera yang terbuka.')
        except Exception as e:
            showerror('Error', 'Terjadi kesalahan: ' + str(e))

    def set_parameter(self):
        for i in range(0, self.nOpenDevSuccess):
            # Mendapatkan nilai dari teks di GUI
            exposure_time = self.text_exposure_time.get(1.0, END).rstrip("\n")
            gain = self.text_gain.get(1.0, END).rstrip("\n")
            frame_rate = self.text_frame_rate.get(1.0, END).rstrip("\n")

            # Mengatur parameter pada objek kamera
            self.obj_cam_operation[i].exposure_time = exposure_time
            self.obj_cam_operation[i].gain = gain
            self.obj_cam_operation[i].frame_rate = frame_rate

            # Memanggil metode Set_parameter pada objek kamera
            ret = self.obj_cam_operation[i].Set_parameter(frame_rate, exposure_time, gain)

            # Menampilkan pesan kesalahan jika ada
            if ret != 0:
                showerror('show error', f'Camera {i} set parameter fail! ret = {To_hex_str(ret)}')


if __name__ == "__main__":
    window = Tk()
    window.title('MultipleCamerasDemo')
    window.geometry('1330x1020')
    
    # Membuat objek aplikasi
    app = CameraApp(window)

    # Menjalankan GUI
    window.mainloop()