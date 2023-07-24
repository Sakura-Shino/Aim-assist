import pynput
import mss

def lock(aims, mouse:pynput.mouse.Controller, monitor:dict, x, y):
    mouse_pos_x, mouse_pos_y = mouse.position
    # judge whether the mouse is in the monitor
    if mouse_pos_x < monitor['left'] or mouse_pos_x > monitor['left'] + monitor['width'] or mouse_pos_y < monitor['top'] or mouse_pos_y > monitor['top'] + monitor['height']:
        return
    else:
        # transform the mouse position to the monitor 2
        mouse_pos_x = mouse_pos_x - monitor['left'] + 0
        mouse_pos_y = mouse_pos_y - monitor['top'] + 0

    # choose the nearest aim
    dist_list = []
    for det in aims:
        _, x_c, y_c, _, _ = det
        dist = (x * float(x_c) - mouse_pos_x) ** 2 + (y * float(y_c) - mouse_pos_y) ** 2
        dist_list.append(dist)
    
    det = aims[dist_list.index(min(dist_list))]

    tag, x_center, y_center, width, hight = det
    tag = int(tag)
    x_center, width = x * float(x_center), x * float(width)
    y_center, hight = y * float(y_center), y * float(hight)
    # correct the position of the mouse on selected monitor
    x_center += monitor['left']
    y_center += monitor['top']
    tag = 0 # for test
    if tag == 0 or tag == 2:
        mouse.position = (x_center, y_center)
    elif tag == 1 or tag == 3:
        mouse.position = (x_center, y_center - 1 / 6 * hight)