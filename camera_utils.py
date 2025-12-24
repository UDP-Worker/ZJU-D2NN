import sys
import os
from MvImport.MvCameraControl_class import *

if __name__ == "__main__":
    # 1. 初始化SDK
    MvCamera.MV_CC_Initialize()

    # 2. 进行设备发现，控制，图像采集等操作

    # 3. 反初始化SDK
    MvCamera.MV_CC_Finalize()

