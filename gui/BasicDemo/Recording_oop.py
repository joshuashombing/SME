# -- coding: utf-8 --
import sys
sys.path.append("../MvImport")


import sys, os
import threading
import msvcrt
from datetime import datetime
from ctypes import *
from MvCameraControl_class import *


class Recording:
    def __init__(self, deviceList):
        self.cam = MvCamera()
        self.deviceList = deviceList
        self.g_bExit = False

    def work_thread(self, cam=0, pData=0, nDataSize=0):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        stInputFrameInfo = MV_CC_INPUT_FRAME_INFO()
        memset(byref(stInputFrameInfo), 0, sizeof(MV_CC_INPUT_FRAME_INFO))

        while True:
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)

            if None != stOutFrame.pBufAddr and 0 == ret:
                print(f"get one frame: Width[{stOutFrame.stFrameInfo.nWidth}], Height[{stOutFrame.stFrameInfo.nHeight}], nFrameNum[{stOutFrame.stFrameInfo.nFrameNum}]")
                stInputFrameInfo.pData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
                stInputFrameInfo.nDataLen = stOutFrame.stFrameInfo.nFrameLen
                ret = cam.MV_CC_InputOneFrame(stInputFrameInfo)
                if ret != 0:
                    print(f"input one frame fail! nRet [0x{ret:x}]")
                nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print(f"no data[0x{ret:x}]")

            if self.g_bExit:
                break

    def start_recording(self, nConnectionNum):
        if int(nConnectionNum) >= self.deviceList.nDeviceNum:
            print("Input salah!")
            sys.exit()

        stDeviceList = cast(self.deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)

        if ret != 0:
            print(f"Gagal membuat handle! ret[0x{ret:x}]")
            sys.exit()

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

        if ret != 0:
            print(f"Gagal membuka perangkat! ret[0x{ret:x}]")
            sys.exit()

        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()

            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print(f"Peringatan: Gagal mengatur Ukuran Paket! ret[0x{ret:x}]")
            else:
                print(f"Peringatan: Gagal mendapatkan Ukuran Paket! ret[0x{nPacketSize:x}]")

        stBool = c_bool(False)
        ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        stRecordPar = MV_CC_RECORD_PARAM()
        memset(byref(stRecordPar), 0, sizeof(MV_CC_RECORD_PARAM))

        ret = self.cam.MV_CC_GetIntValue("Width", stParam)
        if ret != 0:
            print(f"Gagal mendapatkan lebar! nRet [0x{ret:x}]")
            sys.exit()
        stRecordPar.nWidth = stParam.nCurValue

        ret = self.cam.MV_CC_GetIntValue("Height", stParam)
        if ret != 0:
            print(f"Gagal mendapatkan tinggi! nRet [0x{ret:x}]")
            sys.exit()
        stRecordPar.nHeight = stParam.nCurValue

        stEnumValue = MVCC_ENUMVALUE()
        memset(byref(stEnumValue), 0 ,sizeof(MVCC_ENUMVALUE))
        ret = self.cam.MV_CC_GetEnumValue("PixelFormat", stEnumValue)
        if ret != 0:
            print(f"Gagal mendapatkan Format Pixel! nRet [0x{ret:x}]")
            sys.exit()
        stRecordPar.enPixelType = MvGvspPixelType(stEnumValue.nCurValue)

        stFloatValue = MVCC_FLOATVALUE()
        memset(byref(stFloatValue), 0 ,sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", stFloatValue)
        if ret != 0:
            print(f"Gagal mendapatkan Nilai ResultingFrameRate! nRet [0x{ret:x}]")
            sys.exit()
        stRecordPar.fFrameRate = stFloatValue.fCurValue

        stRecordPar.nBitRate = 10000
        stRecordPar.enRecordFmtType = MV_FormatType_AVI
        # stRecordPar.enRecordFmtType = MV_FormatType_Undefined
        # Buat folder jika belum tersedia
        save_folder = "data/save/recording"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generate nama file dengan timestamp
        current_time = datetime.now()
        file_name = f"recording_{current_time.strftime('%Y%m%d_%H%M%S')}.avi"
        stRecordPar.strFilePath = os.path.join(save_folder, file_name).encode('ascii')

        nRet = self.cam.MV_CC_StartRecord(stRecordPar)
        if nRet != 0:
            print(f"Mulai Rekam gagal! nRet [0x{nRet:x}]")
            sys.exit()

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"Mulai pengambilan gambar gagal! ret[0x{ret:x}]")
            sys.exit()

        try:
            self.hThreadHandle = threading.Thread(target=self.work_thread, args=(self.cam, None, None))
            self.hThreadHandle.start()
        except:
            print("error: unable to start thread")

        print("Tekan tombol untuk menghentikan pengambilan gambar.")
        msvcrt.getch()
        self.g_bExit = True
        self.hThreadHandle.join()

    def stop_recording(self):
        self.g_bExit = True

        if self.hThreadHandle is not None:
            self.hThreadHandle.join()

        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"Gagal menghentikan pengambilan gambar! ret[0x{ret:x}]")
            sys.exit()

        ret = self.cam.MV_CC_StopRecord()
        if ret != 0:
            print(f"Gagal menghentikan Rekam! ret[0x{ret:x}]")
            sys.exit()

        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print(f"Gagal menutup perangkat! ret[0x{ret:x}]")
            sys.exit()

        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print(f"Gagal menghancurkan handle! ret[0x{ret:x}]")
            sys.exit()


if __name__ == "__main__":
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

    if ret != 0:
        print(f"enum devices fail! ret[0x{ret:x}]")
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print(f"Find {deviceList.nDeviceNum} devices!")

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = input("please input the number of the device to connect:")

    recording = Recording(deviceList)
    recording.start_recording(nConnectionNum)

    print("press a key to stop grabbing.")
    msvcrt.getch()

    recording.stop_recording()
