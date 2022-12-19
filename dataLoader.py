from utils import *
from torch.utils import data
import os
from torchvision import transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import torch as t

"""
data_root_dir
    train
        images
            1.jpg
            2.jpg
            3.jpg
            ......
        labels
            1.xml
            2.xml
            3.xml
            ......
    val
        images
            1.jpg
            2.jpg
            3.jpg
            ......
        labels
            1.xml
            2.xml
            3.xml
            ......
"""


class MySet(data.Dataset):

    def __init__(self, data_root_dir, is_train, img_size ,num_classes, class_names, R, sigma_iou_thresh):
        """

        :param data_root_dir: data root dir
        :param is_train: True for training, False for validating
        :param img_size: (H, W)
        :param num_classes: count of class
        :param class_names: name of classes, [class1, class2, ....]
        :param R: network stride
        :param sigma_iou_thresh: iou threshold for getting the gauss standard standard deviation sigma
        """
        if is_train:
            self.img_dir = os.path.join(data_root_dir, "train", "images")
            self.label_dir = os.path.join(data_root_dir, "train", "labels")
        else:
            self.img_dir = os.path.join(data_root_dir, "val", "images")
            self.label_dir = os.path.join(data_root_dir, "val", "labels")
        self.names = [name.split(".")[0] for name in os.listdir(self.img_dir)]
        self.img_size = img_size
        self.transformer = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])
        self.num_classes = num_classes
        self.class_names = class_names
        self.R = R
        self.iou_thresh = sigma_iou_thresh
        assert len(self.class_names) == self.num_classes, "element count of class_names should equal to num_classes"

    def __getitem__(self, index):
        name = self.names[index]
        img_pth = os.path.join(self.img_dir, "%s.jpg" % (name,))
        if not os.path.exists(img_pth):
            img_pth = os.path.join(self.img_dir, "%s.png" % (name,))
        label_pth = os.path.join(self.label_dir, "%s.xml" % (name,))
        img = Image.open(img_pth)
        original_size = img.size  # (original_w, original_h)
        w_ratio = self.img_size[1] / original_size[0]
        h_ratio = self.img_size[0] / original_size[1]
        data = self.transformer(img)
        tree = ET.parse(label_pth)
        object_nodes = tree.findall("object")
        label = t.zeros((self.num_classes + 4, self.img_size[0] // self.R, self.img_size[1] // self.R)).type(t.FloatTensor)  # label[:self.num_classes] is gauss heat map, label[self.num_classes] is x_offset, label[self.num_classes + 1] is y_offset, label[self.num_classes + 2] is bounding box width, label[self.num_classes + 3] is bounding box height, bounding box width and height between 0 and 1
        for obj in object_nodes:
            class_name = obj.find("name").text
            bbox_node = obj.find("bndbox")
            xmin = int(int(bbox_node.find("xmin").text) * w_ratio)
            ymin = int(int(bbox_node.find("ymin").text) * h_ratio)
            xmax = int(int(bbox_node.find("xmax").text) * w_ratio)
            ymax = int(int(bbox_node.find("ymax").text) * h_ratio)
            bbox_w = (xmax - xmin) / self.img_size[1]
            bbox_h = (ymax - ymin) / self.img_size[0]
            key_point_coord = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            sigma = get_sigma((xmin, ymin, xmax, ymax), self.iou_thresh, self.R)
            gauss_map_of_curr_obj, key_point_coord_stride, key_point_coord_stride_int = get_one_key_point_gauss(self.img_size, self.R, sigma, key_point_coord)
            class_index = self.class_names.index(class_name)
            gauss_map_of_curr_obj = t.from_numpy(gauss_map_of_curr_obj).unsqueeze(0).type(t.FloatTensor)
            label[class_index:class_index + 1, :, :] = t.max(t.cat([label[class_index:class_index + 1, :, :], gauss_map_of_curr_obj], dim=0), dim=0).values
            label[self.num_classes, key_point_coord_stride_int[1], key_point_coord_stride_int[0]] = key_point_coord_stride[0] - key_point_coord_stride_int[0]
            label[self.num_classes + 1, key_point_coord_stride_int[1], key_point_coord_stride_int[0]] = key_point_coord_stride[1] - key_point_coord_stride_int[1]
            label[self.num_classes + 2, key_point_coord_stride_int[1], key_point_coord_stride_int[0]] = bbox_w
            label[self.num_classes + 3, key_point_coord_stride_int[1], key_point_coord_stride_int[0]] = bbox_h
        return data, label

    def __len__(self):
        return len(self.names)


def make_loader(data_root_dir, is_train, img_size ,num_classes, class_names, R, sigma_iou_thresh, batch_size, num_workers):
    loader = data.DataLoader(MySet(data_root_dir, is_train, img_size ,num_classes, class_names, R, sigma_iou_thresh), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    data_root_dir = r"G:\fish_video\fish_data\data"
    is_train = True
    img_size = (512, 512)
    num_classes = 1
    class_names = ["fish"]
    R = 4
    iou_thresh = 0.7
    batch_size = 8
    s = make_loader(data_root_dir, is_train, img_size ,num_classes, class_names, R, iou_thresh, batch_size, num_workers=1)
    for data, label in s:
        print((label[0, 0, :, :] == 1).sum())
        print((t.norm(label[:, num_classes:num_classes + 2, :, :][0], p=1, dim=0) != 0).sum())
        print("====================")