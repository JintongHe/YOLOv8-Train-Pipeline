# coding = utf8
from osgeo import gdal, osr
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
    # 打开影像文件以获取地理信息
    dataset = gdal.Open(image_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    pixel_arr = []  # 用于存储像素坐标的列表

    if not coordinates or len(coordinates) <= 1:
        # 如果输入坐标为空或只有一个坐标，则默认返回整个影像的像素范围
        pixel_arr.append([0, 0])  # 左上角像素坐标
        pixel_arr.append([width, height])  # 右下角像素坐标
    else:
        # 遍历输入的经纬度坐标
        for coordinate in coordinates:
            coordinate_x = coordinate[0]  # 经度
            coordinate_y = coordinate[1]  # 纬度
            # 使用地理转换信息将经纬度坐标转换为像素坐标
            pixel_x = int((coordinate_x - geo_transform[0]) / geo_transform[1])
            pixel_y = int((coordinate_y - geo_transform[3]) / geo_transform[5])
            # 将像素坐标添加到结果列表中
            pixel_arr.append([pixel_x, pixel_y])

    return pixel_arr  # 返回包含像素坐标的列表

# 求像素点坐标最大值，最小值
def pixel_max_min(pixel_arr):
    # 初始化像素坐标的最小和最大值，初始值设置为负一表示尚未找到有效值
    pixel_x_min = -1
    pixel_x_max = -1
    pixel_y_min = -1
    pixel_y_max = -1

    # 遍历像素坐标列表以查找最小和最大值
    for pixel in pixel_arr:
        pixel_x = pixel[0]
        pixel_y = pixel[1]

        # 如果是第一个像素，将其值分别赋给最小和最大值
        if pixel_x_min == -1:
            pixel_x_min = pixel_x
        if pixel_y_min == -1:
            pixel_y_min = pixel_y
        if pixel_x_max == -1:
            pixel_x_max = pixel_x
        if pixel_y_max == -1:
            pixel_y_max = pixel_y

        # 检查当前像素是否大于或小于已知的最大和最小值
        if pixel_x > pixel_x_max:
            pixel_x_max = pixel_x
        if pixel_x < pixel_x_min:
            pixel_x_min = pixel_x
        if pixel_y > pixel_y_max:
            pixel_y_max = pixel_y
        if pixel_y < pixel_y_min:
            pixel_y_min = pixel_y

    # 返回找到的像素坐标的最小和最大值
    return pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max


# 切割图片
def split_image(image_path, split_arr, pixel_arr):
    # 计算像素坐标的最小和最大值
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)

    # 创建一个空字典来存储分割后的图像
    split_images_dict = {}
    num = 0  # 用于分割图像的编号

    # 遍历分割参数列表
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]

        # 读取要分割的图片
        picture = cv2.imread(image_path)

        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 提取分割后的图像
                pic = picture[h0: h1, w0: w1, :]

                # 存储分割后图像的信息到字典中
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1

                # 情况2：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if w0 < pixel_x_max:
                    pic = picture[h0: h1, w0: w1, :]

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

                # 情况3：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

                # 情况4：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w + int(cut_width / 2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict


# 切割大图片
def split_image_large(image_path, split_arr, pixel_arr):
    # 计算像素坐标的最小和最大值
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)

    # 创建一个空字典来存储分割后的图像
    split_images_dict = {}
    num = 0  # 用于分割图像的编号

    # 遍历分割参数列表
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]

        # 读取要分割的图片，以及其尺寸等数据
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount

        # 如果不是三通道的tif图片，使用普通分割函数
        if num_bands != 3:
            split_images_dict = split_image(image_path, split_arr, pixel_arr)
            return split_images_dict

        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 读取分割图像的数据
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))

                # 创建分割后图像的numpy数组
                pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[2]  # Blue channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[0]  # Red channel

                # 存储分割后图像的信息到字典中
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1

                # 情况2：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if w0 < pixel_x_max:
                    # 读取分割图像的数据
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))

                    # 创建分割后图像的numpy数组
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

                # 情况3：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if h0 < pixel_y_max:
                    # 读取分割图像的数据
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))

                    # 创建分割后图像的numpy数组
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

                # 情况4：左上角坐标 (w0, h0)，右下角坐标 (w1, h1)
                w0 = w + int(cut_width / 2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height

                # 防止超出图片边界
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max

                # 仅当左上角坐标不越界时提取分割后的图像
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    # 读取分割图像的数据
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))

                    # 创建分割后图像的numpy数组
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel

                    # 存储分割后图像的信息到字典中
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    在图像上绘制一个边界框。

    Args:
        x (list): 边界框的坐标，通常为 [x1, y1, x2, y2] 形式的列表。
        img (numpy.ndarray): 输入图像，通常是一个NumPy数组。
        color (list, optional): 边界框的颜色，通常为包含三个整数值的列表，表示颜色的RGB值。如果未提供颜色，将随机生成一个颜色。
        label (str, optional): 边界框的标签文本。
        line_thickness (int, optional): 边界框线的粗细。

    Returns:
        None
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线条/字体的粗细
    color = color or [random.randint(0, 255) for _ in range(3)]  # 如果未提供颜色，随机生成一个颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 边界框的左上角和右下角坐标
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # 在图像上绘制矩形框

    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # 获取文本大小
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 确定文本框的位置
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 在边界框上方绘制填充矩形框
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)  # 绘制文本

# 模型预测
def model_predict(config_dict, mid_dict):
    """
    使用预训练模型进行目标检测和分割的预测。

    Args:
        config_dict (dict): 配置字典，包含模型路径、设备、置信度阈值等参数。
        mid_dict (dict): 中间结果字典，包含分割图像等数据。

    Returns:
        mid_dict (dict): 更新后的中间结果字典，包含目标检测和分割的预测结果。
    """
    logging.info('开始模型预测')

    # 从配置字典中获取必要的参数
    model_path = config_dict['model_path']  # 预训练模型路径
    device = config_dict['device']  # 设备（如'cuda'或'cpu'）
    conf_thres = config_dict['conf_thres']  # 置信度阈值
    polygon_threshold = config_dict['polygon_threshold']  # 多边形简化阈值

    split_images_dict = mid_dict['split_images_dict']  # 分割图像数据字典

    # 初始化用于存储预测结果的数据结构
    all_box_arr = []  # 边界框坐标列表
    weight_arr = []  # 权重列表
    label_arr = []  # 标签列表
    polygon_arr = []  # 多边形坐标列表
    mask_arr = []  # 掩码列表

    # 提取分割图像数据
    pics = [sub_dict['pic'] for sub_dict in split_images_dict.values()]

    # 初始化目标检测和分割模型
    model = YOLO(model_path)
    batch_size = 16  # 批量处理的图像数

    # 遍历分割图像数据进行预测
    for i in tqdm(range(0, len(pics), batch_size)):
        results = None
        sub_pics = pics[i:i + batch_size]

        # 使用模型进行预测，返回目标检测和分割结果
        results = model.predict(sub_pics, task='segment', save=False, conf=conf_thres,
                                device=device,
                                boxes=True,
                                save_crop=False)

        # 处理每个分割图像的预测结果
        for j in range(i, min(i + batch_size, len(pics))):
            x0 = split_images_dict[j]['location'][0]  # 图像左上角x坐标
            y0 = split_images_dict[j]['location'][1]  # 图像左上角y坐标
            x1 = split_images_dict[j]['location'][2]  # 图像右下角x坐标
            y1 = split_images_dict[j]['location'][3]  # 图像右下角y坐标

            if len(list(results[j - i].boxes.data)) == 0:
                continue

            masks = results[j - i].masks.xy  # 分割掩码
            boxes = results[j - i].boxes.data  # 边界框坐标
            dim0, dim1 = boxes.shape

            # 遍历每个检测到的目标
            for row in range(dim0):
                arr = boxes[row]
                location = []
                location.append([int(arr[0] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[3] + y0)])
                location.append([int(arr[0] + x0), int(arr[3] + y0)])

                # 去除边缘部分的目标
                if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
                        location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
                    continue

                weight = float(arr[-2])  # 目标权重
                label = int(arr[-1])  # 目标标签

                mask = np.array(masks[row]).astype(int)  # 分割掩码
                if mask.shape[0] <= 3:
                    continue

                # 对多边形进行简化，以减少点的数量
                points = Polygon(mask).buffer(0)
                if type(points) is MultiPolygon:
                    largest_polygon = max(points.geoms, key=lambda p: p.area)
                    points = largest_polygon

                # 将多边形坐标转换为相对于原图的坐标
                exterior_coords_tuples = list(
                    points.simplify(polygon_threshold, preserve_topology=True).exterior.coords)
                points = np.array([[int(t[0]), int(t[1])] for t in exterior_coords_tuples])
                points[:, 0] = points[:, 0] + x0
                points[:, 1] = points[:, 1] + y0

                mask[:, 0] = mask[:, 0] + x0
                mask[:, 1] = mask[:, 1] + y0

                # 将目标的相关信息添加到相应的列表中
                all_box_arr.append(location)
                weight_arr.append(weight)
                label_arr.append(label)
                polygon_arr.append(points.tolist())
                mask_arr.append(mask.tolist())

        # 释放内存
        del results
        torch.cuda.empty_cache()

    # 将预测结果存储在字典中
    predict_result = {'all_box_arr': all_box_arr, 'weight_arr': weight_arr, 'label_arr': label_arr,
                      'polygon_arr': polygon_arr, 'mask_arr': mask_arr}
    mid_dict['predict_result'] = predict_result
    logging.info('模型预测完成')

    return mid_dict


# 目标检测模型预测
def model_predict_bbox(config_dict, mid_dict):
    """
    使用预训练模型进行目标检测的预测。

    Args:
        config_dict (dict): 配置字典，包含模型路径、设备、置信度阈值等参数。
        mid_dict (dict): 中间结果字典，包含分割图像等数据。

    Returns:
        mid_dict (dict): 更新后的中间结果字典，包含目标检测的预测结果。
    """
    logging.info('开始模型预测')

    # 从配置字典中获取必要的参数
    model_path = config_dict['model_path']  # 预训练模型路径
    device = config_dict['device']  # 设备（如'cuda'或'cpu'）
    conf_thres = config_dict['conf_thres']  # 置信度阈值

    split_images_dict = mid_dict['split_images_dict']  # 分割图像数据字典

    # 初始化用于存储目标检测结果的数据结构
    all_box_arr = []  # 边界框坐标列表
    weight_arr = []  # 权重列表
    label_arr = []  # 标签列表

    # 提取分割图像数据
    pics = [sub_dict['pic'] for sub_dict in split_images_dict.values()]

    # 初始化目标检测模型
    model = YOLO(model_path)
    batch_size = 16  # 批量处理的图像数

    # 遍历分割图像数据进行目标检测预测
    for i in tqdm(range(0, len(pics), batch_size)):
        results = None
        sub_pics = pics[i:i + batch_size]

        # 使用模型进行目标检测预测，返回目标检测结果
        results = model.predict(sub_pics, task='detect', save=False, conf=conf_thres,
                                device=device,
                                boxes=True,
                                save_crop=False)

        # 处理每个分割图像的目标检测结果
        for j in range(i, min(i + batch_size, len(pics))):
            x0 = split_images_dict[j]['location'][0]  # 图像左上角x坐标
            y0 = split_images_dict[j]['location'][1]  # 图像左上角y坐标
            x1 = split_images_dict[j]['location'][2]  # 图像右下角x坐标
            y1 = split_images_dict[j]['location'][3]  # 图像右下角y坐标

            if len(list(results[j - i].boxes.data)) == 0:
                continue

            boxes = results[j - i].boxes.data  # 目标边界框坐标
            dim0, dim1 = boxes.shape

            # 遍历每个检测到的目标
            for row in range(dim0):
                arr = boxes[row]
                location = []
                location.append([int(arr[0] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[1] + y0)])
                location.append([int(arr[2] + x0), int(arr[3] + y0)])
                location.append([int(arr[0] + x0), int(arr[3] + y0)])

                # 去除边缘部分的目标
                if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
                        location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
                    continue

                weight = float(arr[-2])  # 目标权重
                label = int(arr[-1])  # 目标标签

                # 将目标的相关信息添加到相应的列表中
                all_box_arr.append(location)
                weight_arr.append(weight)
                label_arr.append(label)

        # 释放内存
        del results
        torch.cuda.empty_cache()

    # 将目标检测结果存储在字典中
    predict_result = {'all_box_arr': all_box_arr, 'weight_arr': weight_arr, 'label_arr': label_arr}
    mid_dict['predict_result'] = predict_result
    logging.info('模型预测完成')

    return mid_dict


# 数据增强
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
    """
    对输入的图像进行分割并应用CLAHE (Contrast Limited Adaptive Histogram Equalization) 增强。

    Args:
        image_path (str): 输入图像文件的路径。
        split_arr (list): 分割参数的列表，每个参数是一个包含切割宽度和高度的二元组。
        pixel_arr (list): 像素坐标的列表，表示允许的图像分割范围。

    Returns:
        split_images_dict (dict): 包含分割图像及其位置信息的字典。
    """
    # 计算图像的最小和最大像素坐标
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)

    # 初始化存储分割图像的字典
    split_images_dict = {}
    num = 0

    for split_data in split_arr:
        # 获取分割后的图像宽度和高度
        cut_width = split_data[0]
        cut_height = split_data[1]

        # 读取输入的图像文件
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

                # 应用CLAHE增强
                pic = apply_clahe_to_rgb_image(pic)

                # 存储分割图像及其位置信息
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
                w0 = w + int(cut_width / 2)
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
    """
    对输入的图像进行分割并应用CLAHE (Contrast Limited Adaptive Histogram Equalization) 增强。

    Args:
        image_path (str): 输入图像文件的路径。
        split_arr (list): 分割参数的列表，每个参数是一个包含切割宽度和高度的二元组。
        pixel_arr (list): 像素坐标的列表，表示允许的图像分割范围。

    Returns:
        split_images_dict (dict): 包含分割图像及其位置信息的字典。
    """
    # 计算图像的最小和最大像素坐标
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)

    # 初始化存储分割图像的字典
    split_images_dict = {}
    num = 0

    for split_data in split_arr:
        # 获取分割后的图像宽度和高度
        cut_width = split_data[0]
        cut_height = split_data[1]

        # 读取输入的图像文件
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
                # 应用CLAHE增强
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
                    # 应用CLAHE增强
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
                    # 应用CLAHE增强
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
                    # 应用CLAHE增强
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
    """
    计算两个多边形之间的交叠面积。

    Args:
        data1 (list): 第一个多边形的坐标点列表。
        data2 (list): 第二个多边形的坐标点列表。

    Returns:
        inter_area (float): 两个多边形之间的交叠面积。
    """
    # 创建多边形对象并获取其凸包
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两个多边形不相交，则交叠面积为0
    else:
        inter_area = poly1.intersection(poly2).area  # 计算相交部分的面积
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
    """
    生成塔吊边界框的多边形坐标。

    Args:
        bbox (list): 边界框的坐标，格式为 [x_min, y_min, x_max, y_max]。
        points (numpy.ndarray): 塔吊周围的点的坐标数组。
        width (float): 边界框的宽度。

    Returns:
        polygon (list): 塔吊边界框的多边形坐标列表。
    """
    # 计算两个最远点之间的距离
    distances = squareform(pdist(points))
    farthest_points_idx = np.unravel_index(np.argmax(distances), distances.shape)
    p1, p2 = points[farthest_points_idx[0]], points[farthest_points_idx[1]]

    # 计算边界框与两个最远点的交点
    p1, p2 = intersection_points(bbox, p1, p2)

    # 计算两个最远点之间的距离
    d = np.linalg.norm(p1 - p2)

    # 计算两个最远点的中点
    midpoint = (p1 + p2) / 2

    # 计算方向向量
    dir_vector = (p2 - p1) / d

    # 构建矩形的四个顶点
    v1 = p1 + (width / 2) * np.array([-dir_vector[1], dir_vector[0]])
    v2 = p1 + (width / 2) * np.array([dir_vector[1], -dir_vector[0]])
    v3 = v2 + d * dir_vector
    v4 = v1 + d * dir_vector

    polygon = [list(v1), list(v2), list(v3), list(v4)]
    polygon = [[int(j) for j in i] for i in polygon]
    return polygon

def intersection_points(bbox, p1, p2):
    """
    计算边界框和线段之间的交点。

    Args:
        bbox (list): 边界框的坐标，格式为 [x_min, y_min, x_max, y_max]。
        p1 (numpy.ndarray): 线段的起点坐标。
        p2 (numpy.ndarray): 线段的终点坐标。

    Returns:
        a (numpy.ndarray): 第一个交点的坐标。
        b (numpy.ndarray): 第二个交点的坐标。
    """
    # 计算线段的斜率和截距
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

    # 计算线段与边界框左边界的交点
    y = m * x1 + c
    if y1 <= y <= y2:
        intersections.append((x1, y))

    # 计算线段与边界框右边界的交点
    y = m * x2 + c
    if y1 <= y <= y2:
        intersections.append((x2, y))

    # 计算线段与边界框上边界的交点
    if m != 0:
        x = (y1 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y1))

    # 计算线段与边界框下边界的交点
    if m != 0:
        x = (y2 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y2))

    # 去除重复的交点
    intersections = list(set(intersections))
    # 将交点转换为NumPy数组格式
    a = np.array([int(intersections[0][0]), int(intersections[0][1])])
    b = np.array([int(intersections[1][0]), int(intersections[1][1])])
    return a, b



# 去重（融合Polygon的方法）
def after_handle_merge_polygon(config_dict, mid_dict):
    """
    执行目标检测结果的后处理，包括去除重叠的预测结果并合并。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了预测结果。

    Returns:
        mid_dict (dict): 包含经过后处理的中间结果的字典，包括去除重叠和合并后的结果。
    """
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']

    overlap_percent = config_dict['overlap_percent']

    # 标记需要移除的重叠预测结果
    box_len = len(polygon_arr)
    flags = [0 for x in range(box_len)]
    for row1 in range(box_len - 1):
        if flags[row1] == 1:
            continue
        poly1 = Polygon(polygon_arr[row1]).buffer(0)
        for row2 in range(row1 + 1, box_len):
            if flags[row2] == 1:
                continue
            poly2 = Polygon(polygon_arr[row2]).buffer(0)

            area1 = poly1.area
            area2 = poly2.area
            if not poly1.intersects(poly2):
                continue
            over_area = poly1.intersection(poly2).area
            if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
                flags[row1] = 1
                union_poly = poly1.union(poly2)
                if not union_poly.is_valid:
                    union_poly = union_poly.buffer(0)
                if type(union_poly) is MultiPolygon:
                    union_poly = max(union_poly.geoms, key=lambda p: p.area)
                union_poly = list(union_poly.exterior.coords)
                union_poly = np.array([[int(t[0]), int(t[1])] for t in union_poly])
                x_min = np.min(union_poly, axis=0)[0]
                x_max = np.max(union_poly, axis=0)[0]
                y_min = np.min(union_poly, axis=0)[1]
                y_max = np.max(union_poly, axis=0)[1]
                all_box_arr[row2] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                polygon_arr[row2] = union_poly.tolist()
                mask_arr[row2] = mask_arr[row2] + mask_arr[row1]
                mask_arr[row1] = []

    mid_dict['flags'] = flags

    # 生成最终的预测结果
    res_box = []
    res_weight = []
    res_label = []
    res_polygon = []
    res_mask = []
    for index in range(len(flags)):
        if flags[index] == 1:
            continue
        elif flags[index] == 0:
            res_box.append(all_box_arr[index])
            res_weight.append(weight_arr[index])
            res_label.append(label_arr[index])
            res_polygon.append(polygon_arr[index])
            res_mask.append(mask_arr[index])
    res_result = {'res_box': res_box, 'res_weight': res_weight, 'res_label': res_label, 'res_polygon': res_polygon,
                  'res_mask': res_mask}
    mid_dict['res_result'] = res_result
    logging.info('后处理完成')
    return mid_dict


# 去重（锚框去重方法）
def after_handle_bbox(config_dict, mid_dict, method='area'):
    """
    执行目标检测结果的后处理，包括去除重叠的预测结果并合并。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了预测结果。
        method (str): 用于确定哪个对象保留的方法。可选值为'area'和'weight'。

    Returns:
        mid_dict (dict): 包含经过后处理的中间结果的字典，包括去除重叠和合并后的结果。
    """
    import geopandas as gpd
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']

    overlap_percent = config_dict['overlap_percent']

    # 创建 GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'polygon': polygon_arr,
        'weight': weight_arr,
        'label': label_arr,
        'mask': mask_arr,
        'box': all_box_arr,
        'geometry': [Polygon(p) for p in all_box_arr]
    })

    # 空间自连接以查找重叠的多边形
    joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')

    # 初始化一个集合以跟踪已处理的索引
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
            # 选择要保留的多边形，根据指定的方法
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

    # 删除已处理（合并）的多边形
    gdf = gdf.drop(index=list(processed_indices))

    # 重新构建结果
    mid_dict['res_result'] = {
        'res_box': gdf['box'].tolist(),
        'res_weight': gdf['weight'].tolist(),
        'res_label': gdf['label'].tolist(),
        'res_mask': gdf['mask'].tolist(),
        'res_polygon': gdf['polygon'].tolist()
    }
    logging.info('后处理完成')

    return mid_dict



# 去重（多边形去重方法）
def after_handle_polygon(config_dict, mid_dict, method='area'):
    """
    执行目标检测结果的后处理，包括去除重叠的预测结果并合并。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了预测结果。
        method (str): 用于确定哪个对象保留的方法。可选值为'area'和'weight'。

    Returns:
        mid_dict (dict): 包含经过后处理的中间结果的字典，包括去除重叠和合并后的结果。
    """
    import geopandas as gpd
    logging.info('开始后处理')
    predict_result = mid_dict['predict_result']
    all_box_arr = predict_result['all_box_arr']
    weight_arr = predict_result['weight_arr']
    label_arr = predict_result['label_arr']
    polygon_arr = predict_result['polygon_arr']
    mask_arr = predict_result['mask_arr']

    overlap_percent = config_dict['overlap_percent']

    # 创建 GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'box': all_box_arr,
        'weight': weight_arr,
        'label': label_arr,
        'mask': mask_arr,
        'geometry': [Polygon(p) for p in polygon_arr]
    })

    # 空间自连接以查找重叠的多边形
    joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')

    # 初始化一个集合以跟踪已处理的索引
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
            # 选择要保留的多边形，根据指定的方法
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

    # 删除已处理（合并）的多边形
    gdf = gdf.drop(index=list(processed_indices))

    # 重新构建结果
    mid_dict['res_result'] = {
        'res_box': gdf['box'].tolist(),
        'res_weight': gdf['weight'].tolist(),
        'res_label': gdf['label'].tolist(),
        'res_mask': gdf['mask'].tolist(),
        'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']]
    }
    logging.info('后处理完成')

    return mid_dict


def sam_handle(config_dict, mid_dict, method='bbox'):
    """
    使用SAM（Segment Anything Model）对目标检测结果进行处理。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了预测结果。
        method (str): SAM 处理的方法，可选值为 'bbox' 或 'point'。

    Returns:
        mid_dict (dict): 包含经过 SAM 处理的中间结果的字典，包括 SAM 处理后的掩码。
    """
    logging.info('开始SAM处理')
    orig_img = config_dict['image_path']
    # SAM 参数
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

    # 创建 SAMPredictor
    overrides = dict(conf=conf, task='segment', save=False, mode='predict', imgsz=1024,
                     model="D:/Code/gitcode/yolov8/model_folder/SAM/sam_h.pt")
    predictor = SAMPredictor(overrides=overrides)

    predict_result = mid_dict['res_result']
    all_box_arr = predict_result['res_box']
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

    # 设置图像
    sam_masks = []
    predictor.set_image(cv2.imread(orig_img))  # 使用 np.ndarray 设置

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

        # 获取多边形的外部坐标作为元组的列表
        exterior_coords_tuples = list(sam_mask.exterior.coords)
        # 如有必要，转换为列表的列表
        sam_mask = [[int(t[0]), int(t[1])] for t in exterior_coords_tuples]
        sam_masks.append(sam_mask)

    # 重置图像
    predictor.reset_image()
    mid_dict['res_result']['res_sam_mask'] = sam_masks
    return mid_dict


def show_images(config_dict, mid_dict):
    """
    可视化目标检测结果并保存为图像文件。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了经过处理的检测结果。

    Returns:
        None
    """
    image_path = config_dict['image_path']
    model_path = config_dict['model_path']
    show_flag = config_dict['show_flag']
    class_dict = config_dict['class_dict']

    # 如果不需要显示，直接返回
    if not show_flag:
        return

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

    # 创建输出目录
    if not os.path.exists(show_path):
        os.makedirs(show_path)

    image_name = image_path.split('/')[-1]
    suf = image_name.split('.')[1]

    try:
        img = cv2.imread(image_path)
    except:
        return 0

    box_len = len(res_box)

    # 绘制边界框并保存图像
    img_bbox = img.copy()

    for row in range(box_len):
        arr = res_box[row]
        label = class_dict[res_label[row]]
        confidence = res_weight[row]
        xyxy = [arr[0][0], arr[0][1], arr[2][0], arr[2][1]]

        plot_one_box(xyxy, img_bbox, label=f'{label} {round(confidence, 2)}', color=(0, 0, 255), line_thickness=2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_bbox" + f'_{model_name}.' + suf)), img_bbox)

    # 绘制多边形轮廓并保存图像
    img_poly = img.copy()

    for row in range(box_len):
        polygon = res_polygon[row]
        polygon = np.array(polygon).reshape((-1, 1, 2))
        cv2.drawContours(img_poly, [polygon], -1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour" + f'_{model_name}.' + suf)), img_poly)

    # 绘制掩码点并保存图像
    img_mask = img.copy()

    for row in range(box_len):
        points = res_mask[row]
        for point in points:
            point = tuple(point)
            point = (int(point[0]), int(point[1]))
            cv2.circle(img_mask, point, radius=3, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_mask" + f'_{model_name}.' + suf)), img_mask)

    logger.info('生成图片完成')


def show_images_with_sam(config_dict, mid_dict):
    """
    可视化目标检测结果和SAM处理结果并保存为图像文件。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了经过处理的检测结果。

    Returns:
        None
    """
    image_path = config_dict['image_path']
    model_path = config_dict['model_path']
    show_flag = config_dict['show_flag']

    # 如果不需要显示，直接返回
    if not show_flag:
        return

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

    # 创建输出目录
    if not os.path.exists(show_path):
        os.makedirs(show_path)

    image_name = image_path.split('/')[-1]
    suf = image_name.split('.')[1]

    try:
        img = cv2.imread(image_path)
    except:
        return 0

    box_len = len(res_box)

    # 绘制边界框并保存图像
    img_bbox = img.copy()

    for row in range(box_len):
        arr = res_box[row]
        label = res_label[row]
        confidence = res_weight[row]
        xyxy = [arr[0][0], arr[0][1], arr[2][0], arr[2][1]]

        plot_one_box(xyxy, img_bbox, label=f'{label} {round(confidence, 2)}', color=(0, 0, 255), line_thickness=2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_bbox" + f'_{model_name}.' + suf)), img_bbox)

    # 绘制多边形轮廓并保存图像
    img_contour = img.copy()

    for row in range(box_len):
        polygon = res_polygon[row]
        polygon = np.array(polygon).reshape((-1, 1, 2))
        cv2.drawContours(img_contour, [polygon], -1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour" + f'_{model_name}.' + suf)),
                img_contour)

    # 绘制掩码点并保存图像
    img_mask = img.copy()

    for row in range(box_len):
        points = res_mask[row]
        for point in points:
            point = tuple(point)
            point = (int(point[0]), int(point[1]))
            cv2.circle(img_mask, point, radius=5, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_mask" + f'_{model_name}.' + suf)), img_mask)

    # 绘制SAM处理的掩码并保存图像
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
    """
    创建 GeoJSON 文件，将检测结果转换为地理信息数据。

    Args:
        config_dict (dict): 包含配置参数的字典。
        mid_dict (dict): 包含中间结果的字典，其中应该包含了经过处理的检测结果。

    Returns:
        mid_dict (dict): 包含了 GeoJSON 数据的中间结果字典。
    """
    start_time = config_dict['start_time']
    image_path = config_dict['image_path']
    out_flag = config_dict['out_flag']
    out_file_path = config_dict['out_file_path']
    class_names = config_dict['class_dict']

    res_result = mid_dict['res_result']
    res_box = res_result['res_box']
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

    for index in range(len(res_box)):
        polygon = res_polygon[index]
        label = res_label[index]
        weight = res_weight[index]
        name = class_names[label]

        feature = {
            "type": "Feature",
            "properties": {
                "Id": 0,
                "name": name,
                "date": date,
                "area": 0.0,
                "label": label,
                "result": 1,
                "XMMC": "",
                "HYMC": "",
                "weight": weight,
                "bz": 0
            },
            "geometry": {"type": "Polygon", "coordinates": []}
        }

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

