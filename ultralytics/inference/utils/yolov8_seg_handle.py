# coding = utf8
from osgeo import gdal
import numpy as np
import torch
import cv2
from shapely import Polygon, MultiPolygon, unary_union
from scipy.spatial.distance import pdist, squareform
import logging
from ultralytics import YOLO
from tqdm import tqdm
import random
import os
import time
from ultralytics.models.sam import Predictor as SAMPredictor
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 经纬度坐标转换为像素点坐标
def lon_lat_to_pixel(image_path, coordinates):
    dataset = gdal.Open(image_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    pixel_arr = []
    if not coordinates or len(coordinates) <= 1:
        pixel_arr.append([0, 0])
        pixel_arr.append([width, height])
    else:
        for coordinate in coordinates:
            coordinate_x = coordinate[0]
            coordinate_y = coordinate[1]
            location = [int((coordinate_x - geo_transform[0]) / geo_transform[1]),
                        int((coordinate_y - geo_transform[3]) / geo_transform[5])]
            pixel_arr.append(location)
    return pixel_arr

# 求像素点坐标最大值，最小值
def pixel_max_min(pixel_arr):
    pixel_x_min = -1
    pixel_x_max = -1
    pixel_y_min = -1
    pixel_y_max = -1
    for pixel in pixel_arr:
        pixel_x = pixel[0]
        pixel_y = pixel[1]
        if pixel_x_min == -1:
            pixel_x_min = pixel_x
        if pixel_y_min == -1:
            pixel_y_min = pixel_y
        if pixel_x_max == -1:
            pixel_x_max = pixel_x
        if pixel_y_max == -1:
            pixel_y_max = pixel_y

        if pixel_x > pixel_x_max:
            pixel_x_max = pixel_x
        if pixel_x < pixel_x_min:
            pixel_x_min = pixel_x
        if pixel_y > pixel_y_max:
            pixel_y_max = pixel_y
        if pixel_y < pixel_y_max:
            pixel_y_max = pixel_y
    return pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max

# 切割图片
def split_image(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        picture = cv2.imread(image_path)

        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                pic = picture[h0: h1, w0: w1, :]
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width/2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict

# 切割大图片
def split_image_large(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        # picture = cv2.imread(image_path)
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
        # 如果不是tif图片，用cv2读
        if num_bands != 3:
            split_images_dict = split_image(image_path, split_arr, pixel_arr)
            return split_images_dict
        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                # pic = picture[h0: h1, w0: w1, :]
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1-w0, h1-h0))
                # Assuming the image is RGB
                pic = np.zeros((h1-h0, w1-w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[2]  # Blue channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[0]  # Red channel
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width / 2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def model_predict(config_dict, mid_dict):
    logging.info('开始模型预测')
    model_path = config_dict['model_path']
    device = config_dict['device']
    conf_thres = config_dict['conf_thres']
    polygon_threshold = config_dict['polygon_threshold']

    split_images_dict = mid_dict['split_images_dict']

    # 预测分割图片，并存储所有分割图像在大图中的标签
    all_box_arr = []
    weight_arr = []
    label_arr = []
    polygon_arr = []
    mask_arr = []
    pics = [sub_dict['pic'] for sub_dict in split_images_dict.values()]
    model = YOLO(model_path)
    batch_size = 16
    for i in tqdm(range(0, len(pics), batch_size)):
        results = None
        sub_pics = pics[i:i+batch_size]
        results = model.predict(sub_pics, task='segment', save=False, conf=conf_thres,
                                device=device,
                                show_boxes=True,
                                save_crop=False)
        for j in range(i, min(i+batch_size, len(pics))):
            x0 = split_images_dict[j]['location'][0]
            y0 = split_images_dict[j]['location'][1]
            x1 = split_images_dict[j]['location'][2]
            y1 = split_images_dict[j]['location'][3]
            if len(list(results[j-i].boxes.data)) == 0:
                continue
            masks = results[j-i].masks.xy
            boxes = results[j-i].boxes.data
            dim0, dim1 = boxes.shape
            for row in range(dim0):
                arr = boxes[row]
                location = []
                location.append([int(arr[0] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[3] + y0)])
                location.append([int(arr[0] + x0), int(arr[3] + y0)])
                # 去除边缘部分
                if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
                        location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
                    continue
                weight = float(arr[-2])
                label = int(arr[-1])

                mask = np.array(masks[row]).astype(int)
                if mask.shape[0] <= 3:
                    continue
                points = Polygon(mask).buffer(0)
                if type(points) is MultiPolygon:
                    largest_polygon = max(points.geoms, key=lambda p: p.area)
                    points = largest_polygon
                exterior_coords_tuples = list(points.simplify(polygon_threshold, preserve_topology=True).exterior.coords)
                # Convert to a list of lists if required
                points = np.array([[int(t[0]), int(t[1])] for t in exterior_coords_tuples])
                points[:, 0] = points[:, 0] + x0
                points[:, 1] = points[:, 1] + y0

                mask[:, 0] = mask[:, 0] + x0
                mask[:, 1] = mask[:, 1] + y0

                all_box_arr.append(location)
                weight_arr.append(weight)
                label_arr.append(label)
                polygon_arr.append(points.tolist())
                mask_arr.append(mask.tolist())
        del results
        torch.cuda.empty_cache()

    predict_result = {'all_box_arr': all_box_arr, 'weight_arr': weight_arr, 'label_arr': label_arr,
                      'polygon_arr': polygon_arr, 'mask_arr': mask_arr}
    mid_dict['predict_result'] = predict_result
    logging.info('模型预测完成')
    # result = [all_box_arr, weight_arr, label_arr, polygon_arr, mask_arr]
    return mid_dict

def model_predict_bbox(config_dict, mid_dict):
    logging.info('开始模型预测')
    model_path = config_dict['model_path']
    device = config_dict['device']
    conf_thres = config_dict['conf_thres']
    polygon_threshold = config_dict['polygon_threshold']

    split_images_dict = mid_dict['split_images_dict']

    # 预测分割图片，并存储所有分割图像在大图中的标签
    all_box_arr = []
    weight_arr = []
    label_arr = []
    pics = [sub_dict['pic'] for sub_dict in split_images_dict.values()]
    model = YOLO(model_path)
    batch_size = 16
    for i in tqdm(range(0, len(pics), batch_size)):
        results = None
        sub_pics = pics[i:i+batch_size]
        results = model.predict(sub_pics, task='detect', save=False, conf=conf_thres,
                                device=device,
                                show_boxes=True,
                                save_crop=False)
        for j in range(i, min(i+batch_size, len(pics))):
            x0 = split_images_dict[j]['location'][0]
            y0 = split_images_dict[j]['location'][1]
            x1 = split_images_dict[j]['location'][2]
            y1 = split_images_dict[j]['location'][3]
            if len(list(results[j-i].boxes.data)) == 0:
                continue
            boxes = results[j-i].boxes.data

            dim0, dim1 = boxes.shape
            for row in range(dim0):
                arr = boxes[row]
                location = []
                location.append([int(arr[0] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[3] + y0)])
                location.append([int(arr[0] + x0), int(arr[3] + y0)])
                # 去除边缘部分
                if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
                        location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
                    continue
                weight = float(arr[-2])
                label = int(arr[-1])

                all_box_arr.append(location)
                weight_arr.append(weight)
                label_arr.append(label)
        del results
        torch.cuda.empty_cache()

    predict_result = {'all_box_arr': all_box_arr, 'weight_arr': weight_arr, 'label_arr': label_arr}
    mid_dict['predict_result'] = predict_result
    logging.info('模型预测完成')
    # result = [all_box_arr, weight_arr, label_arr, polygon_arr, mask_arr]
    return mid_dict

def apply_clahe_to_rgb_image(bgr_image, clip_limit=2, tile_grid_size=(8, 8)):
    # Convert the RGB image to Lab color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)

    # Split the Lab image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the L channel
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with the a and b channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert back to BGR color space
    bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)

    return bgr_image_clahe

# 切割图片
def split_image_augment_cv2(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        picture = cv2.imread(image_path)

        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                pic = picture[h0: h1, w0: w1, :]
                pic = apply_clahe_to_rgb_image(pic)
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    pic = picture[h0: h1, w0: w1, :]
                    pic = apply_clahe_to_rgb_image(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    pic = apply_clahe_to_rgb_image(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width/2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    pic = apply_clahe_to_rgb_image(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict
def split_image_with_augment(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        # picture = cv2.imread(image_path)
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
        # 如果不是tif图片，用cv2读
        if num_bands != 3:
            split_images_dict = split_image_augment_cv2(image_path, split_arr, pixel_arr)
            return split_images_dict
        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                # pic = picture[h0: h1, w0: w1, :]
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1-w0, h1-h0))
                # Assuming the image is RGB
                pic = np.zeros((h1-h0, w1-w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[2]  # Blue channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[0]  # Red channel
                pic = apply_clahe_to_rgb_image(pic)
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    pic = apply_clahe_to_rgb_image(pic)

                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    pic = apply_clahe_to_rgb_image(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width / 2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    pic = apply_clahe_to_rgb_image(pic)
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict
# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if (point_num < 3): return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)

def cal_area_2poly(data1, data2):
    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: Iterable container of points.
    :param alpha: Alpha value to influence the gooeyness of the border.
    Alpha controls how concave the output shape is. Larger values
    make the shape more concave.
    :return: list of simplices
    """
    from scipy.spatial import Delaunay
    from shapely.ops import polygonize
    from shapely.geometry import Polygon, LineString

    triangles = Delaunay(points).simplices
    alpha_edges = []

    for triangle in triangles:
        # Convert triangle points to a Polygon format
        triangle_polygon = Polygon(points[triangle])

        # We add the start point at the end to form a closed ring
        # If checking by perimeter:
        if triangle_polygon.length < alpha:
            for i in range(3):  # triangles have 3 edges
                alpha_edges.append((triangle[i], triangle[(i + 1) % 3]))

        # Uncomment below if checking by area instead:
        # if triangle_polygon.area < alpha:
        #     for i in range(3):  # triangles have 3 edges
        #         alpha_edges.append((triangle[i], triangle[(i + 1) % 3]))

    # Construct lines from the edges and polygonize the result
    lines = [LineString([points[edge[0]], points[edge[1]]]) for edge in alpha_edges]
    return list(polygonize(lines))


def compute_polygon(points, tolerance):
    # points = np.load("test.npy")
    alpha = 10000

    hulls = alpha_shape(points, alpha)
    # while len(hulls) != 1:
    #     alpha += 1000
    #     # print(alpha)
    #     hulls = alpha_shape(points, alpha)
    # print(len(hulls))

    # Plot
    # tolerance = 3
    simplified_hull = hulls[0].simplify(tolerance, preserve_topology=True)
    x, y = simplified_hull.exterior.xy
    polygon = []
    for i in range(len(list(x))):
        polygon.append([int(list(x)[i]), int(list(y)[i])])
    return polygon
    # plt.scatter(points[:, 0], points[:, 1], color='r')
    # plt.plot(x, y, '-', label='Simplified Convex Hull')
    # plt.fill(*hulls[0].exterior.xy, alpha=0.3)
    # plt.show()
# points = np.load("test.npy")
# print(compute_polygon(points, 3))

def towercrane_contour_generator(bbox, points, width):
    # Generate some random points
    # np.random.seed(42)  # for reproducibility
    # points = np.random.rand(20, 2) * 10
    # points = np.load("test.npy")
    # Calculate pairwise distances
    distances = squareform(pdist(points))
    farthest_points_idx = np.unravel_index(np.argmax(distances), distances.shape)
    p1, p2 = points[farthest_points_idx[0]], points[farthest_points_idx[1]]
    p1, p2 = intersection_points(bbox, p1, p2)
    # Distance between the two farthest points
    d = np.linalg.norm(p1 - p2)

    # Midpoint between the two farthest points
    midpoint = (p1 + p2) / 2

    # Direction vector
    dir_vector = (p2 - p1) / d

    # Width of the rectangle
    # width = 2.0  # you can adjust this manually

    # Construct the four vertices of the rectangle
    v1 = p1 + (width/2)*np.array([-dir_vector[1], dir_vector[0]])
    v2 = p1 + (width/2)*np.array([dir_vector[1], -dir_vector[0]])
    v3 = v2 + d*dir_vector
    v4 = v1 + d*dir_vector

    polygon = [list(v1), list(v2), list(v3), list(v4)]
    polygon = [[int(j) for j in i] for i in polygon]
    # return polygon
    # Plotting
    # plt.scatter(points[:, 0], points[:, 1], c='blue')
    # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro-')
    # plt.plot([v1[0], v2[0], v3[0], v4[0], v1[0]], [v1[1], v2[1], v3[1], v4[1], v1[1]], 'g-')
    # plt.axis('equal')
    # plt.show()
    return polygon

def intersection_points(bbox, p1, p2):
    # Calculate the slope and y-intercept for the line passing through a and b
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    x3 = p1[0]
    y3 = p1[1]
    x4 = p2[0]
    y4 = p2[1]
    m = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    c = y3 - m * x3

    intersections = []

    # With x = x1 (left edge)
    y = m * x1 + c
    if y1 <= y <= y2:
        intersections.append((x1, y))

    # With x = x2 (right edge)
    y = m * x2 + c
    if y1 <= y <= y2:
        intersections.append((x2, y))

    # With y = y1 (top edge)
    if m != 0:
        x = (y1 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y1))

    # With y = y2 (bottom edge)
    if m != 0:
        x = (y2 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y2))
    intersections = list(set(intersections))
    a = np.array([int(intersections[0][0]), int(intersections[0][1])])
    b = np.array([int(intersections[1][0]), int(intersections[1][1])])
    return a, b

# # Example usage
# x1, y1 = 0, 0
# x2, y2 = 10, 10
# x3, y3 = 1, 2
# x4, y4 = 4, 3
#
# print(intersection_points(x1, y1, x2, y2, x3, y3, x4, y4))

# bbox = [250,100,750,600]
# points = np.load("test.npy")
# width = 15
# points = contour_generator(bbox, points, width)
# print(points)

# 去重（融合Polygon的方法）
def after_handle_merge_polygon(config_dict, mid_dict):
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']
    overlap_percent = config_dict['overlap_percent']

    # Pre-calculate polygons and their areas
    polygons = [Polygon(p).buffer(0) for p in polygon_arr]
    areas = np.array([p.area for p in polygons])

    box_len = len(polygons)
    flags = np.zeros(box_len, dtype=int)

    for row1 in tqdm(range(box_len - 1)):
        if flags[row1]:
            continue
        for row2 in range(row1 + 1, box_len):
            if flags[row2]:
                continue
            if polygons[row1].intersects(polygons[row2]):
                inter_area = polygons[row1].intersection(polygons[row2]).area
                if inter_area / areas[row1] >= overlap_percent or inter_area / areas[row2] >= overlap_percent:
                    flags[row1] = 1
                    union_poly = polygons[row1].union(polygons[row2]).buffer(0)
                    if isinstance(union_poly, MultiPolygon):
                        union_poly = max(union_poly.geoms, key=lambda p: p.area)
                    polygons[row2] = union_poly
                    areas[row2] = union_poly.area
                    union_poly = list(union_poly.exterior.coords)
                    union_poly = np.array([[int(t[0]), int(t[1])] for t in union_poly])
                    # Update the polygon array directly if needed
                    x_min = np.min(union_poly, axis=0)[0]
                    x_max = np.max(union_poly, axis=0)[0]
                    y_min = np.min(union_poly, axis=0)[1]
                    y_max = np.max(union_poly, axis=0)[1]
                    all_box_arr[row2] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    polygon_arr[row2] = union_poly.tolist()
                    mask_arr[row2] = mask_arr[row2] + mask_arr[row1]
                    mask_arr[row1] = []

    # Build the result based on flags
    valid_indices = np.where(flags == 0)[0]
    mid_dict['res_result'] = {key[1]: [predict_result[key[0]][i] for i in valid_indices] for key in
                              [['all_box_arr', 'res_box'], ['weight_arr', 'res_weight'], ['label_arr', 'res_label'],
                               ['polygon_arr', 'res_polygon'], ['mask_arr', 'res_mask']]}
    logging.info('后处理完成')
    return mid_dict

# 去重（锚框去重方法）
def after_handle_bbox(config_dict, mid_dict, method='area'):
    import geopandas as gpd
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']

    overlap_percent = config_dict['overlap_percent']

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'polygon': polygon_arr,
        'weight': weight_arr,
        'label': label_arr,
        'mask': mask_arr,
        'box': all_box_arr,
        'geometry': [Polygon(p) for p in all_box_arr]
    })

    # Spatial self-join to find overlapping polygons
    joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    # print(joined_gdf.columns)
    # Initialize a set to keep track of processed indices
    processed_indices = set()

    for idx, row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
        row1 = idx
        row2 = row['index_right']

        if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
            continue

        poly1 = gdf.at[row1, 'geometry']
        poly2 = gdf.at[row2, 'geometry']
        area1 = poly1.area
        area2 = poly2.area
        over_area = poly1.intersection(poly2).area

        if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
            # 取大的
            if method == 'area':
                if area1 >= area2:
                    processed_indices.add(row2)
                else:
                    processed_indices.add(row1)
            else:
                if gdf['weight'].tolist()[row1] >= gdf['weight'].tolist()[row2]:
                    processed_indices.add(row2)
                else:
                    processed_indices.add(row1)
    # Remove processed (merged) polygons
    gdf = gdf.drop(index=list(processed_indices))
    # Reconstruct the result
    mid_dict['res_result'] = {
        'res_box': gdf['box'].tolist(),
        # Add other necessary fields
        'res_weight': gdf['weight'].tolist(),
        'res_label': gdf['label'].tolist(),
        'res_mask': gdf['mask'].tolist(),
        'res_polygon': gdf['polygon'].tolist()
    }
    logging.info('后处理完成')

    return mid_dict


# 去重（多边形去重方法）
def after_handle_polygon(config_dict, mid_dict, method='area'):
    import geopandas as gpd
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']

    overlap_percent = config_dict['overlap_percent']
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'box': all_box_arr,
        'weight': weight_arr,
        'label': label_arr,
        'mask': mask_arr,
        'geometry': [Polygon(p) for p in polygon_arr]
    })

    # Spatial self-join to find overlapping polygons
    joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    # print(joined_gdf.columns)
    # Initialize a set to keep track of processed indices
    processed_indices = set()

    for idx, row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
        row1 = idx
        row2 = row['index_right']

        if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
            continue

        poly1 = gdf.at[row1, 'geometry']
        poly2 = gdf.at[row2, 'geometry']
        area1 = poly1.area
        area2 = poly2.area
        over_area = poly1.intersection(poly2).area

        if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
            # 取大的
            if method == 'area':
                if area1 >= area2:
                    processed_indices.add(row2)
                else:
                    processed_indices.add(row1)
            else:
                if gdf['weight'].tolist()[row1] >= gdf['weight'].tolist()[row2]:
                    processed_indices.add(row2)
                else:
                    processed_indices.add(row1)
    # Remove processed (merged) polygons
    gdf = gdf.drop(index=list(processed_indices))
    # Reconstruct the result
    mid_dict['res_result'] = {
        'res_box': gdf['box'].tolist(),
        # Add other necessary fields
        'res_weight': gdf['weight'].tolist(),
        'res_label': gdf['label'].tolist(),
        'res_mask': gdf['mask'].tolist(),
        'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']]
    }
    logging.info('后处理完成')

    return mid_dict

def sam_handle(config_dict, mid_dict, method='bbox'):
    logging.info('开始SAM处理')
    orig_img = config_dict['image_path']
    # SAM Parameters
    conf = 0.25
    crop_n_layers = 0
    crop_overlap_ratio = 512 / 1500
    crop_downscale_factor = 1
    point_grids = None
    points_stride = 32
    points_batch_size = 64
    conf_thres = 0.88
    stability_score_thresh = 0.95
    stability_score_offset = 0.95
    crops_nms_thresh = 0.7

    # Create SAMPredictor
    overrides = dict(conf=conf, task='segment', save=False, mode='predict', imgsz=1024,
                     model="D:/Code/gitcode/yolov8/model_folder/SAM/sam_h.pt")
    predictor = SAMPredictor(overrides=overrides)

    predict_result = mid_dict['res_result']
    all_box_arr = predict_result['res_box']
    # weight_arr = predict_result['weight_arr']
    # label_arr = predict_result['label_arr']
    polygon_arr = predict_result['res_polygon']

    sam_boxes_list = []
    for i in range(len(all_box_arr)):
        one_poly = Polygon(polygon_arr[i])
        if not one_poly.is_valid:
            one_poly = one_poly.buffer(0)
        centroid = one_poly.centroid
        one_box = all_box_arr[i]
        sam_box = [one_box[0][0], one_box[0][1], one_box[2][0], one_box[2][1]]
        sam_boxes_list.append((sam_box, [[int(centroid.x), int(centroid.y)]]))
    # Set image
    sam_masks = []
    predictor.set_image(cv2.imread(orig_img))  # set with np.ndarray
    for sam_box, sam_point in sam_boxes_list:
        if method == 'bbox':
            results = predictor(points=None, labels=None, bboxes=sam_box, crop_n_layers=crop_n_layers,
                                crop_overlap_ratio=crop_overlap_ratio,
                                crop_downscale_factor=crop_downscale_factor, point_grids=point_grids,
                                points_stride=points_stride,
                                points_batch_size=points_batch_size, conf_thres=conf_thres,
                                stability_score_thresh=stability_score_thresh,
                                stability_score_offset=stability_score_offset,
                                crops_nms_thresh=crops_nms_thresh)
        else:
            results = predictor(points=sam_point, labels=[1], bboxes=None, crop_n_layers=crop_n_layers,
                                crop_overlap_ratio=crop_overlap_ratio,
                                crop_downscale_factor=crop_downscale_factor, point_grids=point_grids,
                                points_stride=points_stride,
                                points_batch_size=points_batch_size, conf_thres=conf_thres,
                                stability_score_thresh=stability_score_thresh,
                                stability_score_offset=stability_score_offset,
                                crops_nms_thresh=crops_nms_thresh)
        sam_mask = results[0].masks.xy[0]
        sam_mask.astype(int).tolist()
        sam_mask = Polygon(sam_mask).buffer(0)
        if type(sam_mask) is MultiPolygon:
            sam_mask = max(sam_mask.geoms, key=lambda p: p.area)
        tolerance = 0.5
        sam_mask = sam_mask.simplify(tolerance, preserve_topology=True)
        # Get the exterior coordinates of the polygon as a list of tuples
        exterior_coords_tuples = list(sam_mask.exterior.coords)
        # Convert to a list of lists if required
        sam_mask = [[int(t[0]), int(t[1])] for t in exterior_coords_tuples]
        sam_masks.append(sam_mask)

    # Reset image
    predictor.reset_image()
    mid_dict['res_result']['res_sam_mask'] = sam_masks
    return mid_dict

def show_images(config_dict, mid_dict):
    image_path = config_dict['image_path']
    model_path = config_dict['model_path']
    show_flag = config_dict['show_flag']
    class_dict = config_dict['class_dict']
    if not show_flag:
        return 0
    res_result = mid_dict['res_result']
    res_box = res_result['res_box']
    res_weight = res_result['res_weight']
    res_label = res_result['res_label']
    res_polygon = res_result['res_polygon']
    res_mask = res_result['res_mask']

    image_name = image_path.split('/')[-1]
    model_name = model_path.split('/')[-1].split('.')[0]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    show_path = path + '/out/' + image_name.replace('.' + suf, '_show')
    # 大图打上标签
    if not os.path.exists(show_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(show_path)
    image_name = image_path.split('/')[-1]
    suf = image_name.split('.')[1]
    try:
        img = cv2.imread(image_path)
    except:
        return 0
    box_len = len(res_box)

    # Start with the bounding boxes
    img_bbox = img.copy()  # Create a copy of the original image for bounding boxes

    for row in range(box_len):
        arr = res_box[row]
        label = class_dict[res_label[row]]
        bin_label = bin(res_label[row] + 1)[2:].zfill(3)
        color = [i * 255 for i in list(map(int, list(bin_label)))]
        confidence = res_weight[row]
        xyxy = [arr[0][0], arr[0][1], arr[2][0], arr[2][1]]

        plot_one_box(xyxy, img_bbox, label=f'{label} {round(confidence, 2)}', color=color, line_thickness=2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_bbox" + f'_{model_name}.' + suf)), img_bbox)

    # Now, for the polygons/contours
    img_poly = img.copy()  # Create another copy of the original image for polygons/contours

    for row in range(box_len):
        polygon = res_polygon[row]
        # label = class_dict[res_label[row]]
        bin_label = bin(res_label[row] + 1)[2:].zfill(3)
        color = [i * 255 for i in list(map(int, list(bin_label)))]
        polygon = np.array(polygon).reshape((-1, 1, 2))
        cv2.drawContours(img_poly, [polygon], -1, color, 2)
    print(os.path.join(show_path, image_name.replace('.' + suf, "_contour" + f'_{model_name}.' + suf)))
    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour" + f'_{model_name}.' + suf)), img_poly)

    img_mask = img.copy()  # Create another copy of the original image for polygons/contours

    for row in range(box_len):
        points = res_mask[row]
        for point in points:
            point = tuple(point)
            point = (int(point[0]), int(point[1]))
            cv2.circle(img_mask, point, radius=3, color=(255, 0, 0), thickness=-1)
    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_mask" + f'_{model_name}.' + suf)), img_mask)
    logger.info('生成图片完成')

def show_images_with_sam(config_dict, mid_dict):
    image_path = config_dict['image_path']
    model_path = config_dict['model_path']
    show_flag = config_dict['show_flag']
    if not show_flag:
        return 0
    res_result = mid_dict['res_result']
    res_box = res_result['res_box']
    res_weight = res_result['res_weight']
    res_label = res_result['res_label']
    res_polygon = res_result['res_polygon']
    res_mask = res_result['res_mask']
    res_sam_mask = res_result['res_sam_mask']

    image_name = image_path.split('/')[-1]
    model_name = model_path.split('/')[-1].split('.')[0]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    show_path = path + '/out/' + image_name.replace('.' + suf, '_show')
    # 大图打上标签
    if not os.path.exists(show_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(show_path)
    image_name = image_path.split('/')[-1]
    suf = image_name.split('.')[1]
    try:
        img = cv2.imread(image_path)
    except:
        return 0
    box_len = len(res_box)

    # Start with the bounding boxes
    img_bbox = img.copy()  # Create a copy of the original image for bounding boxes

    for row in range(box_len):
        arr = res_box[row]
        label = res_label[row]
        confidence = res_weight[row]
        xyxy = [arr[0][0], arr[0][1], arr[2][0], arr[2][1]]

        plot_one_box(xyxy, img_bbox, label=f'{label} {round(confidence, 2)}', color=(0, 0, 255), line_thickness=2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_bbox" + f'_{model_name}.' + suf)), img_bbox)

    # Now, for the polygons/contours
    img_contour = img.copy()  # Create another copy of the original image for polygons/contours

    for row in range(box_len):
        polygon = res_polygon[row]
        polygon = np.array(polygon).reshape((-1, 1, 2))
        cv2.drawContours(img_contour, [polygon], -1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour" + f'_{model_name}.' + suf)),
                img_contour)

    img_mask = img.copy()  # Create another copy of the original image for polygons/contours

    for row in range(box_len):
        points = res_mask[row]
        for point in points:
            point = tuple(point)
            point = (int(point[0]), int(point[1]))
            cv2.circle(img_mask, point, radius=5, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_mask" + f'_{model_name}.' + suf)), img_mask)

    img_sam_mask = img.copy()
    for row in range(len(res_sam_mask)):
        sam_mask = res_sam_mask[row]
        sam_mask = np.array(sam_mask).reshape((-1, 1, 2))
        colors = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(img_sam_mask, [sam_mask], -1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_sam_mask" + f'_{model_name}.' + suf)),
                img_sam_mask)
    logger.info('生成图片完成')

def create_geojson(config_dict, mid_dict):
    start_time = config_dict['start_time']
    image_path = config_dict['image_path']
    out_flag = config_dict['out_flag']
    out_file_path = config_dict['out_file_path']
    class_names = config_dict['class_dict']

    res_result = mid_dict['res_result']
    # res_box = res_result['res_box']
    res_weight = res_result['res_weight']
    res_label = res_result['res_label']
    res_polygon = res_result['res_polygon']
    if not out_flag:
        return mid_dict
    time_now = int(time.time())
    image_name = image_path.split('/')[-1]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    show_path = path + '/out/' + image_name.replace('.' + suf, '_predict.geojson')

    gdal.AllRegister()
    dataset = gdal.Open(image_path)
    adfGeoTransform = dataset.GetGeoTransform()

    res_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": []
    }
    date = image_name.split('.')[0][-8:]
    for index in range(len(res_polygon)):
        polygon = res_polygon[index]
        label = res_label[index]
        weight = res_weight[index]
        name = class_names[label]
        feature = {"type": "Feature",
                   "properties": {"Id": 0, "name": name, "date": date, "area": 0.0, "label": label, "result": 1,
                                  "XMMC": "", "HYMC": "", "weight": weight, "bz": 0},
                   "geometry": {"type": "Polygon", "coordinates": []}}
        coordinate = []
        for xy in polygon:
            location = [xy[0] * adfGeoTransform[1] + adfGeoTransform[0],
                        xy[1] * adfGeoTransform[5] + adfGeoTransform[3]]
            coordinate.append(location)
        coordinate.append(coordinate[0])
        feature['geometry']['coordinates'].append(coordinate)
        res_dict['features'].append(feature)

    end_time = time.time()
    consume_time = end_time - start_time
    res = str(res_dict).replace('\'', '"').replace('None', '"None"')
    res_dict['consume_time'] = consume_time
    mid_dict['res'] = res

    # 输出json文件， 默认不输出
    out_file = open(show_path, 'w', encoding='utf8')
    out_file.write(res)
    out_file.close()

    logging.info('图片路径：' + image_path + ' 总耗时（单位s）：' + str(consume_time))

    return mid_dict

from osgeo import gdal, osr


def reproject_dataset(src_ds, dst_srs):
    """Reproject a dataset to a new spatial reference system."""
    # Define target SRS
    dst_wkt = dst_srs.ExportToWkt()

    # Set up the transformation
    resampling_method = gdal.GRA_Bilinear
    error_threshold = 0.125  # Error threshold for transformation approximation
    warp_options = gdal.WarpOptions(resampleAlg=resampling_method, dstSRS=dst_wkt, errorThreshold=error_threshold)

    # Perform the reprojection
    reprojected_ds = gdal.Warp('test.tif', src_ds, options=warp_options, format='MEM')

    return reprojected_ds


def calculate_area(geotiff_path):
    # Open the GeoTIFF file
    src_ds = gdal.Open(geotiff_path)

    # Define the target SRS (e.g., UTM)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(32633)  # Example: UTM zone 33N. Change as needed.

    # Reproject the dataset
    reprojected_ds = reproject_dataset(src_ds, target_srs)

    # Get the GeoTransform and calculate pixel size
    geotransform = reprojected_ds.GetGeoTransform()
    pixelWidth = geotransform[1]
    pixelHeight = -geotransform[5]  # Pixel height is generally negative

    # Calculate area of each pixel
    pixelArea = pixelWidth * pixelHeight

    # Get raster size
    rasterXSize = reprojected_ds.RasterXSize
    rasterYSize = reprojected_ds.RasterYSize

    # Calculate total area
    # totalArea = rasterXSize * rasterYSize * pixelArea

    return pixelArea

