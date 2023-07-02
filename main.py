from grabscreen import grab_screen
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from cs_model import load_model
import torch
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.augmentations import letterbox


x, y = (1920, 1080)
re_x, re_y = (1920, 1080)

imgsz=(640, 640)

conf_thres=0.25  # confidence threshold
iou_thres=0.45 # NMS IOU threshold
max_det=1000 # maximum detections per image
agnostic_nms=False  # class-agnostic NMS
classes=None  # filter by class: --class 0, or --class 0 2 3

model= load_model()
stride, names, pt = model.stride, model.names, model.pt

while True:
    im0=grab_screen(region=(0,0,x,y))
    im0 = cv2.resize(im0, (re_x, re_y))

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
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim=('%g ' * len(line)).rstrip() % line
                # print(aim)
                aim = aim.split(' ')
                aims.append(aim)
        
    if len(aims) > 0:
        for i, det in enumerate(aims):
            _, x_center, y_center, width, hight = det
            x_center, y_center, width, hight = float(x_center)*re_x, float(y_center)*re_y, float(width)*re_x, float(hight)*re_y
            top_left = (int(x_center-width/2), int(y_center-hight/2))
            bottom_right = (int(x_center+width/2), int(y_center+hight/2))
            color = (0, 255, 0)
            cv2.rectangle(im0, top_left, bottom_right, color, thickness = 3)
            

    cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('csgo-detect', re_x//3, re_y//3)
    cv2.imshow('csgo-detect',im0)

    # hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive')
    hwnd = win32gui.FindWindow(None, 'csgo-detect')
    CVRECT = win32gui.GetWindowRect(hwnd)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()