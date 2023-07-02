import torch
import sys
sys.path.append('..')
from models.common import DetectMultiBackend
from utils.general import Profile, check_img_size
from utils.torch_utils import select_device

def load_model(device = '', 
               imgsz=(640, 640), 
               weights = '.\models\yolov5s.pt', 
               half = False
):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # half = device != 'cpu'

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    return model
