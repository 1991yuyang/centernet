from model import CenterNet
import torch as t
from torch import nn
from torchvision import transforms as T
import os
from PIL import Image
import numpy as np
import cv2
from torchvision.ops import nms
from numpy import random as rd
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


img_size = (512, 512)  # (h, w)
num_classes = 20
R = 4
class_names = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
use_best_model = False
peak_value_count = 100
nms_iou_thresh = 0.01
pool_tool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
transformer = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])
color_r = rd.permutation(rd.choice(np.arange(0, 255), num_classes, replace=False)).reshape((-1, 1))
color_g = rd.permutation(rd.choice(np.arange(0, 255), num_classes, replace=False)).reshape((-1, 1))
color_b = rd.permutation(rd.choice(np.arange(0, 255), num_classes, replace=False)).reshape((-1, 1))
colors = np.concatenate([color_r, color_g, color_b], axis=1)


def load_model():
    model = CenterNet(num_classes)
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def load_one_img(img_pth):
    orig_img = Image.open(img_pth)
    d = transformer(orig_img).unsqueeze(0).cuda(0)
    return d, orig_img


def inference_one_img(model, data, orig_img):
    cv2_bgr_orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    w_ratio = img_size[1] / orig_img.size[0]
    h_ratio = img_size[0] / orig_img.size[1]
    with t.no_grad():
        output = model(data)[0].cpu().detach().numpy()  # [num_classes + 4, h, w]
        heatmap = output[:num_classes, :, :]
        offset = output[num_classes:num_classes + 2, :, :]
        size = output[num_classes + 2:num_classes + 4, :, :]
        size[0] = size[0] * img_size[1]
        size[1] = size[1] * img_size[0]
        size = size.astype(int)  # transfor scaled size to real size
        peak_value = pool_tool(t.from_numpy(heatmap)).numpy()
        bboxes_result = {}
        for class_idx in range(peak_value.shape[0]):
            peak_value_c = peak_value[class_idx]
            min_value = np.sort(peak_value_c.ravel())[::-1][peak_value_count - 1]
            peak_point_indexs = peak_value_c >= min_value
            x_offset = offset[0][peak_point_indexs]
            y_offset = offset[1][peak_point_indexs]
            w = size[0][peak_point_indexs]
            h = size[1][peak_point_indexs]
            y, x = np.where(peak_point_indexs)
            confs = peak_value_c[peak_point_indexs].tolist()
            x_tl = (((x + x_offset) * R - w / 2) / w_ratio).tolist()
            y_tl = (((y + y_offset) * R - h / 2) / h_ratio).tolist()
            x_br = (((x + x_offset) * R + w / 2) / w_ratio).tolist()
            y_br = (((y + y_offset) * R + h / 2) / h_ratio).tolist()
            keep_index = nms(t.from_numpy(np.array(list(zip(x_tl, y_tl, x_br, y_br)))), t.from_numpy(np.array(confs)), nms_iou_thresh).numpy().tolist()
            x_tl = np.array(x_tl)[keep_index].astype(int).tolist()
            y_tl = np.array(y_tl)[keep_index].astype(int).tolist()
            x_br = np.array(x_br)[keep_index].astype(int).tolist()
            y_br = np.array(y_br)[keep_index].astype(int).tolist()
            confs = np.array(confs)[keep_index].tolist()
            bboxs = list(zip(x_tl, y_tl, x_br, y_br, confs))
            point_color = colors[class_idx].tolist()
            class_name = class_names[class_idx]
            bboxes_result[class_name] = bboxs
            for bbx in bboxs:
                ptLeftTop = list(bbx[:2])
                ptRightBottom = list(bbx[2:4])
                conf = bbx[-1]
                thickness = 2
                lineType = 4
                cv2.rectangle(cv2_bgr_orig_img, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)
                t_size = cv2.getTextSize(class_name + ":%.2f" % (conf,), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
                textlbottom = ptLeftTop + np.array(list(t_size))
                cv2.rectangle(cv2_bgr_orig_img, tuple(ptLeftTop), tuple(textlbottom), point_color, 2)
                ptLeftTop[1] = int(ptLeftTop[1] + (t_size[1] / 2 + 4))
                cv2.putText(cv2_bgr_orig_img, class_name + ":%.2f" % (conf,), tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, point_color, 1)
    cv2.imshow("img", cv2_bgr_orig_img)
    cv2.waitKey()
    cv2.imwrite("result.png", cv2_bgr_orig_img)
    return bboxes_result


if __name__ == "__main__":
    img_pth = r"F:\data\VOCdevkit\VOC2012\voc\val\images\2008_000076.jpg"
    model = load_model()
    d, orig_img = load_one_img(img_pth)
    inference_one_img(model, d, orig_img)
