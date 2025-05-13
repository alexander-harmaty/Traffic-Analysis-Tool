import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Dynamically add UltraFast repo path to sys.path
ultrafast_repo_path = os.path.join(os.path.dirname(__file__), 'ultrafast_lane')
if ultrafast_repo_path not in sys.path:
    sys.path.insert(0, ultrafast_repo_path)

from model.model import parsingNet  # from cloned UltraFast repo

# Predefined row anchors for CULane (288 height)
CULANE_ROW_ANCHORS = [121, 131, 141, 150, 160, 170, 180, 189,
                      199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class UltraFastLaneDetector:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.row_anchors = CULANE_ROW_ANCHORS
        self.griding_num = 200
        self.num_lanes = 4
        self.cls_num_per_lane = len(self.row_anchors)

        self.model = parsingNet(
            pretrained=False,
            backbone='18',
            use_aux=False,
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, self.num_lanes)
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        model_state = self.model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items()
                               if k in model_state and v.size() == model_state[k].size()}
        model_state.update(filtered_checkpoint)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect_lanes(self, frame: np.ndarray):
        original_h, original_w = frame.shape[:2]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_transformed = self.img_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(img_transformed)[0].squeeze(0).cpu().numpy()

        prob = out[:-1, :, :]  # exclude background class (index 200)
        idx = np.arange(self.griding_num).reshape(-1, 1, 1)
        prob = softmax(prob, axis=0)
        loc = np.sum(prob * idx, axis=0)
        out_max = np.max(prob, axis=0)

        lanes = []
        for i in range(self.num_lanes):
            lane = []
            for r in range(self.cls_num_per_lane):
                if out_max[r, i] > 0.3:  # confidence threshold
                    x = int(loc[r, i] * original_w / self.griding_num)
                    y = int(self.row_anchors[r] * original_h / 288)
                    lane.append((x, y))
            if len(lane) > 2:
                lanes.append(lane)
        return lanes

def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def draw_lanes(frame: np.ndarray, lanes):
    for lane in lanes:
        for pt in lane:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    return frame
