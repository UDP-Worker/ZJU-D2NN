import numpy as np
import cv2
from ctypes import *
from MvImport.MvCameraControl_class import *


# 安全解码相机字符串（型号、序列号等）
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


def read_from_camera() -> np.ndarray:
    """
    从默认相机（索引0）抓取一帧图像，返回 OpenCV 兼容的 BGR np.ndarray。
    专为 MV-CE060-10UC (BayerRG8 输出) 优化，已测试可用。
    """
    cam = MvCamera()

    # 1. 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0:
        raise RuntimeError(f"枚举设备失败: 0x{ret:x}")
    if deviceList.nDeviceNum == 0:
        raise RuntimeError("未检测到任何相机，请检查连接")

    # 使用第一台相机（索引0）
    stDeviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # 2. 创建句柄并打开设备
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0: raise RuntimeError(f"创建句柄失败: 0x{ret:x}")
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0: raise RuntimeError(f"打开设备失败: 0x{ret:x}")

    try:
        # 3. 设置连续采集模式
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # 4. 开始取流（必须）
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0: raise RuntimeError(f"开始取流失败: 0x{ret:x}")

        # 5. 获取一帧数据大小
        stPayloadSize = MVCC_INTVALUE()
        memset(byref(stPayloadSize), 0, sizeof(stPayloadSize))
        ret = cam.MV_CC_GetIntValue("PayloadSize", stPayloadSize)
        if ret != 0: raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:x}")
        nPayloadSize = stPayloadSize.nCurValue

        # 6. 抓取一帧（超时10秒）
        data_buf = (c_ubyte * nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
        if ret != 0:
            raise RuntimeError(f"抓图失败: 0x{ret:x}（检查光照、曝光时间、镜头盖）")

        # 7. 转为 numpy 并处理 BayerRG8 → BGR
        img_buffer = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)

        # 你的相机输出 BayerRG8 (0x01080009)
        if stFrameInfo.enPixelType == 0x01080009:  # BayerRG8
            raw = img_buffer.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            frame = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)  # OpenCV 中 BayerRG8 用 BG2BGR
        else:
            # 备用方案（如果改了像素格式）
            if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                gray = img_buffer.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif stFrameInfo.enPixelType in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                frame = img_buffer.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)
                if stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError(f"不支持的像素格式: 0x{stFrameInfo.enPixelType:08x}")

        return frame.copy()  # 返回副本，安全

    finally:
        # 8. 清理资源
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


# ====================== 测试代码 ======================
if __name__ == "__main__":
    try:
        img = read_from_camera()
        print(f"成功读取图像！形状: {img.shape}, 数据类型: {img.dtype}")

        # 可选：保存为文件验证
        cv2.imwrite("camera_capture.jpg", img)
        print("图像已保存为 camera_capture.jpg")

        # 显示图像
        cv2.imshow("MV-CE060-10UC Capture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"读取失败: {e}")