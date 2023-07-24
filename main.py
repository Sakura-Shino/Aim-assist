import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from grabscreen import grab_screen
from cs_model import load_model
import mss

import torch
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.augmentations import letterbox

import time
import pynput
from pynput import keyboard
from mouse_controller import lock


x, y = (1920, 1080)
re_x, re_y = (1920, 1080)

# initial lock mode
lock_mode = False

def on_press(key):
    try:
        global lock_mode
        print('alphanumeric key {0} pressed'.format(
            key.char))
      
        if key.char == 'z':
            lock_mode = not lock_mode
            print('lock mode: ', 'on' if lock_mode else 'off')

    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        lock_mode = False
        # Stop listener
        # return False


# Collect events until released
# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

# initial mss model and monitor
sct=mss.mss()
mss_mode = True
monitors = sct.monitors # 获取所有屏幕的信息
print(monitors)
monitor_id = 2
monitor = monitors[monitor_id] # 获取编号为monitor_id的屏幕的信息
# monitor['top'] = 0
# monitor['left'] = 0
# monitor['width'] = x
# monitor['height'] = y

imgsz=(640, 640)

conf_thres=0.25  # confidence threshold
iou_thres=0.45 # NMS IOU threshold
max_det=1000 # maximum detections per image
agnostic_nms=False  # class-agnostic NMS
classes=None  # filter by class: --class 0, or --class 0 2 3

model= load_model()
stride, names, pt = model.stride, model.names, model.pt

# mouse
mouse = pynput.mouse.Controller()

while True:
    if mss_mode:
        im0 = np.array(sct.grab(monitor))[:, :, :3] # get raw pixels from the screen, save it to a Numpy array, [:, :, :3] BGRA to BGR
    else:
        im0 = grab_screen(region=(0, 0, x, y))
    im0 = cv2.resize(im0, (re_x, re_y)) # resize to 1920x1080, for better performance

    # operate on im0, use im instead
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    
    # if im is not None:
    #     break
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim im=im.unsqueeze(0)
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # print(pred)

    aims = []
    for i, det in enumerate(pred):  # detections per image
        s=''
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det): # xyxy, confidence, class
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                # print(line)
                aim=('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                # aim format: [id, x_center, y_center, width, hight]
                aims.append(aim)
                # print(aim)
        
    if len(aims) > 0:
        if lock_mode:
            lock(aims, mouse, monitor, x, y)
            # pass
        # lock(aims, mouse, x, y)
        for i, det in enumerate(aims):
            _, x_center, y_center, width, hight = det
            x_center, y_center, width, hight = float(x_center)*re_x, float(y_center)*re_y, float(width)*re_x, float(hight)*re_y
            top_left = (int(x_center-width/2), int(y_center-hight/2))
            bottom_right = (int(x_center+width/2), int(y_center+hight/2))
            color = (0, 255, 0)
            cv2.rectangle(im0, top_left, bottom_right, color, thickness = 3)
            

    cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('csgo-detect', re_x//2, re_y//2)
    cv2.imshow('csgo-detect',im0)

    # hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive')
    hwnd = win32gui.FindWindow(None, 'csgo-detect')
    CVRECT = win32gui.GetWindowRect(hwnd)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()