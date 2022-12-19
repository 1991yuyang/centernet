import numpy as np


def get_one_key_point_gauss(img_size, R, sigma, key_point_coord):
    """

    :param img_size: image size, (H, W)
    :param R: whole network stride
    :param sigma: gauss standard deviation
    :param key_point_coord: key point coordinate, (x, y)
    :return:
    """
    h = img_size[0] // R
    w = img_size[1] // R
    y_range = np.arange(h).reshape((-1, 1))
    x_range = np.arange(w).reshape((1, -1))
    key_point_coord_stride = (key_point_coord[0] / R, key_point_coord[1] / R)
    key_point_coord_stride_int = (int(key_point_coord_stride[0]), int(key_point_coord_stride[1]))
    gauss_map_of_current_key_point = np.exp(-(np.power((x_range - key_point_coord_stride_int[0]), 2) + np.power((y_range - key_point_coord_stride_int[1]), 2)) / (2 * sigma ** 2))
    return gauss_map_of_current_key_point, key_point_coord_stride, key_point_coord_stride_int


def get_sigma(bbox, iou_thresh, R):
    """

    :param bbox: (x1, y1, x2, y2)
    :param iou_thresh: iou threshold
    :param R: whole network stride
    :return: sigma
    """
    h = (bbox[3] - bbox[1]) / R
    w = (bbox[2] - bbox[0]) / R
    a1 = 4 * iou_thresh
    b1 = 2 * iou_thresh * (h + w)
    c1 = (iou_thresh - 1) * (h * w)
    sigma1 = (np.sqrt(b1 ** 2 - 4 * a1 * c1) - b1) / (2 * a1)
    a2 = 4
    b2 = -2 * (h + w)
    c2 = (1 - iou_thresh) * (h * w)
    sigma2 = (-np.sqrt(b2 ** 2 - 4 * a2 * c2) - b2) / (2 * a2)
    a3 = 1
    b3 = -(w + h)
    c3 = (1 - iou_thresh) * (w * h) / (1 + iou_thresh)
    sigma3 = (-np.sqrt(b3 ** 2 - 4 * a3 * c3) - b3) / (2 * a3)
    sigma = np.min([sigma1, sigma2, sigma3])
    return sigma

if __name__ == "__main__":
    import cv2
    bbox = [100, 20, 300, 30]
    key_point_coord = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    iou_thresh = 0.7
    R = 4
    sigma = get_sigma(bbox, iou_thresh, R)
    print(sigma)
    gauss_map_of_current_key_point, key_point_coord_stride, key_point_coord_stride_int = get_one_key_point_gauss((512, 640), 4, sigma, key_point_coord)
    bgr_map = cv2.cvtColor((gauss_map_of_current_key_point * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imshow("gauss_map", bgr_map)
    cv2.waitKey()
