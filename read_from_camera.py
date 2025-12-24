import numpy as np
import cv2
from ctypes import *
from MvImport.MvCameraControl_class import *

# 安全解码字符串
def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ['gbk', 'utf-8', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode('latin-1', errors='replace')

# Bayer 格式对应的 OpenCV 去马赛克代码（针对你的相机是 BayerRG8）
def get_bayer_cv_code(pixel_type):
    if pixel_type == 0x01080009:  # BayerRG8
        return cv2.COLOR_BAYER_BG2BGR   # OpenCV 中 BayerRG8 对应 COLOR_BAYER_BG2BGR
    # 如果以后遇到其他 Bayer，可继续扩展
    elif pixel_type == 0x01080008:  # BayerGR8
        return cv2.COLOR_BAYER_GB2BGR
    elif pixel_type == 0x0108000A:  # BayerGB8
        return cv2.COLOR_BAYER_GR2BGR
    elif pixel_type == 0x0108000B:  # BayerBG8
        return cv2.COLOR_BAYER_RG2BGR
    return None

def read_from_camera(device_index: int = 0) -> np.ndarray:
    cam = MvCamera()

    # 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0 or device_index >= deviceList.nDeviceNum:
        raise RuntimeError("未找到相机或索引错误")

    stDeviceInfo = cast(deviceList.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents
    print(f"选中设备 [{device_index}]: {decoding_char(stDeviceInfo.SpecialInfo.stUsb3VInfo.chModelName)} "
          f"(序列号: {decoding_char(stDeviceInfo.SpecialInfo.stUsb3VInfo.chSerialNumber)})")

    # 创建并打开句柄
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0: raise RuntimeError(f"创建句柄失败: 0x{ret:x}")
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0: raise RuntimeError(f"打开设备失败: 0x{ret:x}")

    try:
        # 设置连续模式
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0: print(f"警告: 设置 TriggerMode 失败 (0x{ret:x})")

        # 必须先开始取流
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始取流失败: 0x{ret:x}")

        # 获取 PayloadSize
        stPayloadSize = MVCC_INTVALUE()
        memset(byref(stPayloadSize), 0, sizeof(stPayloadSize))
        ret = cam.MV_CC_GetIntValue("PayloadSize", stPayloadSize)
        if ret != 0: raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:x}")
        nPayloadSize = stPayloadSize.nCurValue

        # 抓取一帧
        data_buf = (c_ubyte * nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        print("正在抓取图像（已开始取流）...")
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
        if ret != 0:
            raise RuntimeError(f"抓取图像失败，错误码: 0x{ret:x}")

        print(f"成功获取一帧！分辨率: {stFrameInfo.nWidth}x{stFrameInfo.nHeight}，像素格式: 0x{stFrameInfo.enPixelType:08x}")

        # 转为 numpy
        img_buffer = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
        pixel_type = stFrameInfo.enPixelType
        height = stFrameInfo.nHeight
        width = stFrameInfo.nWidth

        # 处理不同格式
        if pixel_type == PixelType_Gvsp_Mono8:  # 黑白相机常见
            img = img_buffer.reshape(height, width)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif pixel_type in [0x01080008, 0x01080009, 0x0108000A, 0x0108000B]:  # Bayer8 系列（你的就是 0x01080009 = BayerRG8）
            img = img_buffer.reshape(height, width)  # uint8 单通道
            cv_code = get_bayer_cv_code(pixel_type)
            if cv_code:
                img = cv2.cvtColor(img, cv_code)
            else:
                raise RuntimeError("不支持的 Bayer8 格式")

        elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
            img = img_buffer.reshape(height, width, 3)
            if pixel_type == PixelType_Gvsp_RGB8_Packed:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        else:
            raise RuntimeError(f"暂不支持的像素格式: 0x{pixel_type:08x}（可在 MVS 客户端查看并修改为 BayerRG8 或 RGB8）")

        return img.copy()

    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    try:
        frame = read_from_camera(0)
        cv2.imshow("Camera Frame - MV-CE060-10UC", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("图像显示成功！")
    except Exception as e:
        print(f"错误: {e}")