import mss
import mss.tools
import numpy as np

with mss.mss() as sct:
    # 获取所有屏幕的信息
    monitors = sct.monitors
    # print(monitors)

    # 获取编号为2的屏幕的信息
    monitor = monitors[1]
    print(monitor)

    # 获取编号为2的屏幕的截图
    sct_img = sct.grab(monitor)
    # print(np.dtype(sct_img))

    # 将截图保存为PNG文件
    # mss.tools.to_png(sct_img.rgb, sct_img.size, output="screenshot.png")