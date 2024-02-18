from osgeo import gdal
import geopandas as gpd
import torch
import os
import cv2
import json
import shutil
import numpy as np
import random
import re
from shutil import copyfile
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
import split_image_v1
import split_image_v2
from multiprocessing import Pool
from pypinyin import lazy_pinyin
from concurrent.futures import ThreadPoolExecutor

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')


def resize_image_aspect_ratio(img, new_size=640):
    # 获取原始图像的高度和宽度
    h, w = img.shape[:2]

    # 确定缩放因子
    if h > w:
        scale = new_size / h  # 计算高度的缩放因子
        new_h, new_w = new_size, int(w * scale)  # 根据缩放因子计算新的高度和宽度
    else:
        scale = new_size / w  # 计算宽度的缩放因子
        new_h, new_w = int(h * scale), new_size  # 根据缩放因子计算新的高度和宽度

    # 调整图像大小，使用线性插值方法
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized_img


def show_batch_image_segment(image_folder_path, label_folder_path):
    # 提取源文件夹的路径
    source_path = '/'.join(image_folder_path.split('/')[:-1])
    save_path = source_path + '/' + 'images_with_contour'

    # 如果保存路径不存在，创建它
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    # 获取图像文件夹中的所有文件名
    image_names_suf = os.listdir(image_folder_path)

    # 遍历图像文件夹中的每个图像文件
    for image_name_suf in image_names_suf:
        image_name, file_extension = os.path.splitext(image_name_suf)

        # 检查文件扩展名是否在支持的图像文件扩展名列表中
        if file_extension.lower() in IMAGE_EXTENSIONS:
            image_path = image_folder_path + '/' + image_name_suf
            label_path = label_folder_path + '/' + image_name + '.txt'

            try:
                img = cv2.imread(image_path)
            except:
                continue

            img_height, img_width, _ = img.shape

            res_boxes = []
            res_labels = []

            try:
                with open(label_path, 'r') as file:
                    for line in file:
                        # 解析标签文件中的数据
                        split_line = list(map(float, line.strip().split()[1:]))
                        label = int(line.strip().split()[0])
                        new_split_line = []

                        # 将标签中的坐标数据转换为图像坐标
                        for i in range(0, len(split_line), 2):
                            new_split_line.append([int(split_line[i] * img_width), int(split_line[i + 1] * img_height)])

                        res_boxes.append(new_split_line)
                        res_labels.append(label)

                # 处理每个标签，并在图像上绘制轮廓
                for i, res_box in enumerate(res_boxes):
                    res_box = np.array(res_box).reshape((-1, 1, 2))
                    res_label = res_labels[i] + 1
                    bin_label = bin(res_label)[2:].zfill(3)
                    color = [i * 255 for i in list(map(int, list(bin_label)))]

                    indices_of_negatives = np.where(res_box < 0)

                    if len(indices_of_negatives[0]) > 0:
                        print(label_path)
                        print('检测到负值坐标!!!')
                        return

                    cv2.drawContours(img, [res_box], -1, color=color, thickness=2)

                # 保存带有轮廓的图像
                output_path = os.path.join(save_path, image_name + "_contour" + file_extension)
                cv2.imwrite(output_path, img)
                print('已生成图片保存至{}'.format(output_path))

            except FileNotFoundError:
                pass


# 批量读取geojson文件并转为txt格式
def geojson2txt_segment(geojson_path, image_path):
    # 获取geojson文件列表
    json_file_names = os.listdir(geojson_path)

    # 提取源文件夹的路径
    source_dir = '/'.join(geojson_path.split('/')[:-1])

    # 创建存储txt文件的文件夹
    if not os.path.exists(source_dir + '/Label_txt'):
        os.makedirs(source_dir + '/Label_txt')

    # 遍历geojson文件
    for json_file_name in json_file_names:
        if not json_file_name.endswith('.geojson'):
            continue

        # 获取文件名和扩展名
        file_name_with_suf = json_file_name.split('/')[-1].split('.')
        file_name = file_name_with_suf[0]
        suf = file_name_with_suf[1]

        # 构建文件路径
        geojson_file_path = geojson_path + '/' + json_file_name
        image_file_path = image_path + '/' + file_name + '.tif'
        label_path = source_dir + '/Label_txt/' + file_name + '.txt'

        with open(label_path, 'w') as out:
            label = 0

            # 打开图像文件并获取相关信息
            in_ds = gdal.Open(image_file_path)
            width = in_ds.RasterXSize
            height = in_ds.RasterYSize
            outbandsize = in_ds.RasterCount
            adfGeoTransform = in_ds.GetGeoTransform()

            # 打开并解析geojson文件
            f = open(geojson_file_path, encoding='utf8')
            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']

            # 遍历geojson中的features
            for feature in features:
                try:
                    name = feature['properties']['name']
                    if name == '标注范围' or name == '项目范围':
                        continue
                except KeyError:
                    pass

                geometry = feature['geometry']
                coordinates = geometry['coordinates']
                if len(coordinates) == 0:
                    continue
                label_res = ''

                try:
                    for i in range(0, len(coordinates[0][0]) - 1):
                        lat_lon = coordinates[0][0][i]
                        lat = lat_lon[0]
                        lon = lat_lon[1]

                        # 将geojson坐标转换为像素坐标
                        x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                        y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                        label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                except:
                    for i in range(0, len(coordinates[0]) - 1):
                        lat_lon = coordinates[0][i]
                        lat = lat_lon[0]
                        lon = lat_lon[1]

                        # 将geojson坐标转换为像素坐标
                        x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                        y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                        label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                if label_res != '':
                    label_res = str(label) + label_res
                    out.write(label_res + '\n')

            f.close()


# 批量将shp文件转为geojson文件
def batch_read_shapfile(shp_path):
    # 获取源文件夹路径
    source_dir = shp_path
    file_names = os.listdir(shp_path)

    for file_name in file_names:
        if not file_name.endswith('.shp'):
            continue
        target_dir = shp_path
        target_filename = os.path.basename(file_name).split(".")[0]

        # 构建.geojson文件路径
        geojson_file = target_dir + '/' + target_filename + ".geojson"

        # 如果已存在.geojson文件，则跳过
        if os.path.exists(geojson_file):
            continue

        # 读取.shp文件为GeoDataFrame对象
        data = gpd.read_file(source_dir + '/' + file_name)

        # 设置坐标参考系统为EPSG:4326
        data.crs = 'EPSG:4326'

        # 将数据保存为.geojson文件
        data.to_file(geojson_file, driver="GeoJSON")

        # 将文件内容从gbk格式转换为utf-8格式
        with open(geojson_file, 'r', encoding='gbk') as f:
            content = f.read()
        with open(geojson_file, 'w', encoding='utf8') as f:
            f.write(content)


# 将json格式文件批量转为txt格式文件
def json2txt_segment(image_folder_path, json_path):
    # 获取图像文件夹的上级目录路径
    source_dir = '/'.join(image_folder_path.split('/')[:-1])

    # 创建存储.txt文件的文件夹
    txt_folder_path = source_dir + '/Label_txt'
    if not os.path.exists(txt_folder_path):
        os.mkdir(txt_folder_path)
    else:
        shutil.rmtree(txt_folder_path)
        os.makedirs(txt_folder_path)

    # 获取图像文件夹中的所有图像文件名，并按照数字顺序排序
    image_names = os.listdir(image_folder_path)
    image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))

    # 读取JSON文件中的标注信息
    with open(json_path, "r") as file:
        data = json.load(file)
    annotations = data["annotations"]
    annotations = sorted(annotations, key=lambda x: int(x['image_id']), reverse=True)

    # 遍历每个图像
    for image in image_names:
        image_name, file_extension = os.path.splitext(image)

        # 如果图像文件扩展名不在支持的列表中，跳过该图像
        if file_extension.lower() not in IMAGE_EXTENSIONS:
            continue

        image_name = str(int(image_name))
        new_image_path = image_folder_path + '/' + image_name + file_extension
        img_path = image_folder_path + '/' + image
        os.rename(img_path, new_image_path)
        print("开始处理ID为{}的图像".format(image_name))
        image = cv2.imread(new_image_path)
        txt_path = txt_folder_path + '/' + image_name + '.txt'

        img_height, img_width, _ = image.shape

        with open(txt_path, "w") as out_file:
            while True:
                try:
                    if str(annotations[-1]["image_id"]) == str(image_name):
                        seg = np.array(annotations[-1]["segmentation"]).reshape(-1, 2).astype(float)
                        seg[:, 0] = seg[:, 0] / img_width
                        seg[:, 1] = seg[:, 1] / img_height
                        seg[seg < 0] = 0
                        seg = seg.reshape(-1)
                        line = ' '.join(map(str, seg))
                        line = f"0 {line} \n"
                        out_file.write(line)
                        annotations.pop()
                    else:
                        break
                except IndexError:
                    print("处理完成！")
                    break


# 将mask格式文件批量转为yolo格式文件
def mask2segment(label_folder_path):
    # 获取标签文件夹的上级目录路径
    source_dir = '/'.join(label_folder_path.split('/')[:-1])

    # 创建存储.txt文件的文件夹
    txt_folder_path = source_dir + '/label_txt'
    if not os.path.exists(txt_folder_path):
        os.mkdir(txt_folder_path)

    # 获取标签文件夹中的所有标签文件名
    label_names = os.listdir(label_folder_path)

    # 遍历每个标签文件
    for label in label_names:
        label_name, suf = os.path.splitext(label)

        # 如果标签文件扩展名不在支持的列表中，跳过该标签文件
        if suf.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_folder_path + '/' + label

        # 读取标签TIFF图像
        dataset = gdal.Open(label_path)
        label_img = dataset.ReadAsArray().T  # 转置以匹配OpenCV的形状要求
        img_height, img_width = label_img.shape

        # 转换为8位图像（根据您的具体情况可能需要调整）
        # label_img_8bit = np.uint8(label_img)

        # 阈值化以转换为二值图像
        _, thresh = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 构建.txt文件路径
        txt_file_path = txt_folder_path + '/' + label_name + '.txt'

        with open(txt_file_path, 'w') as f:
            for contour in contours:
                contour = contour.astype(float)
                contour = np.reshape(contour, (-1, 2))
                first_col = contour[:, 0].copy()

                # 交换第一列和第三列
                contour[:, 0] = contour[:, 1]
                contour[:, 1] = first_col
                contour[:, 0] = contour[:, 0] / img_width
                contour[:, 1] = contour[:, 1] / img_height

                contour = np.reshape(contour, (-1))
                line = ' '.join(map(str, contour))
                line = f'0 {line}'
                print(line)
                f.write(line + '\n')


# 将geojson矩形框转化为txt矩形框
def geojson2txt_bbox(geojson_path):
    # 获取GeoJSON文件夹中的所有文件名
    json_file_names = os.listdir(geojson_path)

    # 获取GeoJSON文件夹的上级目录路径
    source_dir = '/'.join(geojson_path.split('/')[:-1])

    # 创建存储.txt文件的文件夹
    if not os.path.exists(source_dir + '/Label_txt'):
        os.makedirs(source_dir + '/Label_txt')

    # 遍历每个GeoJSON文件
    for json_file_name in json_file_names:
        if not json_file_name.endswith('.geojson'):
            continue

        # 提取文件名和扩展名
        file_name_with_suf = json_file_name.split('/')[-1].split('.')
        file_name = file_name_with_suf[0]
        suf = file_name_with_suf[1]

        # 构建GeoJSON文件路径、图像文件路径和.txt文件路径
        geojson_path = source_dir + '/Labels/' + file_name + '.geojson'
        image_path = source_dir + '/Images/' + file_name + '.tif'
        label_path = source_dir + '/Label_txt/' + file_name + '.txt'

        with open(label_path, 'w') as out:
            label = 0

            # 打开图像文件
            in_ds = gdal.Open(image_path)
            print("打开.tif文件成功")
            print(geojson_path)

            # 获取图像宽度、高度和波段数
            width = in_ds.RasterXSize
            height = in_ds.RasterYSize
            outbandsize = in_ds.RasterCount
            print(width, height, outbandsize)

            # 读取图像地理信息
            adfGeoTransform = in_ds.GetGeoTransform()
            print(adfGeoTransform)
            print('\n')

            # 读取GeoJSON信息
            f = open(geojson_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']

            for feature in features:
                geometry = feature['geometry']
                coordinates = geometry['coordinates']

                if len(coordinates) == 0:
                    continue
                label_res = ''

                for coordinate in coordinates[0]:
                    for i in range(1, 4):
                        lat_lon = coordinate[i]
                        lat = lat_lon[0]
                        lon = lat_lon[1]

                        # 将GeoJSON坐标转换为像素坐标
                        x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                        y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                        label_res = label_res + ' ' + str(x) + ' ' + str(y)

                    lat_lon = coordinate[0]
                    lat = lat_lon[0]
                    lon = lat_lon[1]

                    # 将GeoJSON坐标转换为像素坐标
                    x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                    y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                    label_res = label_res + ' ' + str(x) + ' ' + str(y)

                if label_res != '':
                    label_res = str(label) + label_res
                    out.write(label_res + '\n')

            f.close()


# 将xyxy格式的txt文件转为yolo格式
def txt2yolo_bbox(txt_path):
    # 获取.txt文件所在目录的上级目录路径
    source_dir = '/'.join(txt_path.split('/')[:-1])

    # 创建存储新.txt文件的文件夹
    new_folder = source_dir + '/txt_bbox'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    # 获取.txt文件夹中的所有.txt文件
    txt_files = os.listdir(txt_path)

    # 遍历每个.txt文件
    for txt_file in txt_files:
        if not txt_file.endswith('.txt'):
            continue

        # 提取文件名（不含扩展名）
        txt_file_name = txt_file.split('.')[0]
        txt_file_path = txt_path + '/' + txt_file

        # 从.txt文件加载坐标数据
        array = np.loadtxt(txt_file_path, dtype=float)

        # 检查坐标数组的形状
        if len(array.shape) == 1:
            array = array.reshape(1, -1)

        # 从原始数据中提取所需信息并计算新的坐标数据
        new_array = array[:, 0].astype(int)
        x_array = array[:, 1::2]
        y_array = array[:, 2::2]
        x_array_max = np.max(x_array, axis=1)
        x_array_min = np.min(x_array, axis=1)
        x_center = (x_array_max + x_array_min) / 2
        y_array_max = np.max(y_array, axis=1)
        y_array_min = np.min(y_array, axis=1)
        y_center = (y_array_max + y_array_min) / 2
        w_array = x_array_max - x_array_min
        h_array = y_array_max - y_array_min
        new_array = np.transpose(np.vstack((new_array, x_center, y_center, w_array, h_array)))

        # 构建新的.txt文件路径
        out_file_path = new_folder + '/' + txt_file

        # 将新坐标数据保存到新的.txt文件中
        np.savetxt(out_file_path, new_array, fmt=['%d', '%f', '%f', '%f', '%f'])


# 在图片上画出xyxy矩形框
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 在图像img上绘制一个边界框

    # 计算线条/字体的厚度（如果未指定厚度，则基于图像大小自动计算）
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # 如果未指定颜色，随机生成一个颜色
    color = color or [random.randint(0, 255) for _ in range(3)]

    # 获取边界框的两个角点坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # 在图像上绘制矩形边界框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # 如果有标签信息，将标签绘制在边界框上
    if label:
        # 计算字体的厚度
        tf = max(tl - 1, 1)

        # 计算标签文本的大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # 根据文本大小确定标签矩形的位置
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # 在图像上绘制填充的标签矩形
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)

        # 在标签矩形上绘制标签文本
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 批量画出图片加标注的效果
def show_batch_image_bbox(image_path, txt_file):
    # 获取源目录（图像和标签的根目录）
    source_dir = '/'.join(image_path.split('/')[:-1])

    # 创建保存可视化结果的文件夹
    show_path = source_dir + '/out'
    if not os.path.exists(show_path):
        os.makedirs(show_path)

    # 获取图像文件列表
    image_files = os.listdir(image_path)

    for image_file in image_files:
        # 检查是否为支持的图像文件类型
        if not image_file.endswith(IMAGE_EXTENSIONS):
            continue

        # 获取图像文件名（不带扩展名）
        image_name = os.path.splitext(image_file)[0]

        # 构建图像文件和对应标签文件的完整路径
        image_file_path = image_path + '/' + image_file
        txt_file_path = txt_file + '/' + image_name + '.txt'

        # 读取图像
        img = cv2.imread(image_file_path)
        img_height, img_width, _ = img.shape

        # 初始化标签和边界框列表
        res_label = [0]
        res_box = []

        # 打开对应的标签文件并解析标签数据
        with open(txt_file_path, 'r') as file:
            for line in file:
                split_line = list(map(float, line.strip().split()[1:]))

                # 计算边界框的四个坐标值
                split_line[0] = int(split_line[0] * img_width - split_line[2] * img_width / 2)
                split_line[1] = int(split_line[1] * img_height - split_line[3] * img_height / 2)
                split_line[2] = int(split_line[2] * img_width + split_line[0])
                split_line[3] = int(split_line[3] * img_height + split_line[1])
                res_box.append(split_line)

        box_len = len(res_box)

        # 为每个边界框绘制矩形和标签
        colors = [random.randint(0, 255) for _ in range(3)]
        for i in range(box_len):
            label = res_label[0]
            xyxy = res_box[i]

            # 使用plot_one_box函数绘制边界框和标签
            plot_one_box(xyxy, img, label=f'{label} ', color=colors, line_thickness=2)

        # 将带有边界框的图像保存到输出文件夹
        suf = os.path.splitext(image_file)[1]
        cv2.imwrite(os.path.join(show_path, image_name + '_contour.' + suf), img)
        print('已生成图片保存至{}'.format(os.path.join(show_path, image_name + '_contour.' + suf)))


# 归一化xyxy
def normalize_bbox_coords(xyxy, image_width, image_height):
    # 输入的xyxy是一个包含边界框坐标的列表，每个边界框由四个值表示：[x_min, y_min, x_max, y_max]
    # image_width和image_height是图像的宽度和高度

    # gn是用于归一化的增益，它的形状为[1, 0, 1, 0]
    gn = torch.tensor((image_height, image_width, 3))[[1, 0, 1, 0]]

    # 使用xyxy2xywh函数将xyxy坐标转换为xywh格式，然后除以gn进行归一化
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

    # 返回归一化后的坐标
    return xywh


# xyxy转xywh格式标注
def xyxy2xywh(x):
    # 将nx4的边界框坐标从[x1, y1, x2, y2]格式转换为[x, y, w, h]格式，其中xy1为左上角，xy2为右下角

    # 创建一个与输入x相同类型的新变量y，以便进行操作
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # 计算边界框的中心点坐标x和y
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x中心点
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y中心点

    # 计算边界框的宽度和高度
    y[:, 2] = x[:, 2] - x[:, 0]  # 宽度
    y[:, 3] = x[:, 3] - x[:, 1]  # 高度

    # 返回转换后的边界框坐标
    return y


# 目标检测图片分割
def split_images_bbox(IMAGE_FOLDER, split_size, LABEL_FOLDER):
    # 创建用于存放训练图像和标签的文件夹路径
    training_image_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_images'
    if not os.path.exists(training_image_folder_path):
        os.mkdir(training_image_folder_path)
    training_label_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_labels'
    if not os.path.exists(training_label_folder_path):
        os.mkdir(training_label_folder_path)

    # 遍历原始图像文件夹中的每个图像文件
    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.endswith(IMAGE_EXTENSIONS):  # 检查文件是否为 .tif 图像
            continue
        print(f"开始处理 {filename}，分割大小为 {split_size}")
        image_file_path = os.path.join(IMAGE_FOLDER, filename)
        image = gdal.Open(image_file_path)

        # 提取不带扩展名的基本名称以用于文件夹和命名
        basename = os.path.splitext(filename)[0]
        new_folder_path = os.path.join(IMAGE_FOLDER, basename)

        # 图像对应的标签路径
        label_path = LABEL_FOLDER + '/' + basename + '.txt'

        width, height, num_bands = image.RasterXSize, image.RasterYSize, image.RasterCount

        # 读取标签并将必要的数据存储在 labels 列表中
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                label_class = int(line.strip().split()[0])
                label_box = list(map(float, line.strip().split()[1:]))
                label_box[0] = int(label_box[0] * width)
                label_box[1] = int(label_box[1] * height)
                label_box[2] = int(label_box[2] * width + label_box[0])
                label_box[3] = int(label_box[3] * height + label_box[1])
                labels.append([label_class, label_box])

        # 将图像分割为大小为 [x, y] 的小块
        x = split_size[0]
        y = split_size[1]
        for i in range(0, width, x):
            for j in range(0, height, y):

                # cropped_image = image[j:j + y, i:i + x]
                local_image_width = min(x, width - i)
                local_image_height = min(y, height - j)

                # 保存分割后的图像，使用指定的格式
                cropped_filename = f"{basename}_{x}by{y}_{i // x + 1}_{j // y + 1}.jpg"

                # 遍历标签，检查是否在小块内
                flag = 0
                text_file = training_label_folder_path + '/' + cropped_filename.split('.')[0] + '.txt'
                try:
                    for label in labels:
                        label_class = label[0]
                        label_box = label[1]

                        if label_box[0] > i and label_box[2] < i + x and label_box[1] > j and label_box[3] < j + y:
                            flag = 1

                            local_xyxy = [label_box[0] - i, label_box[1] - j, label_box[2] - i, label_box[3] - j]
                            xywh = normalize_bbox_coords(local_xyxy, local_image_width, local_image_height)
                            with open(text_file, 'a') as f:
                                f.write(f"{label_class} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

                    # 如果小块包含标签信息，保存图像为训练图像
                    if flag == 1:
                        # 读取每个波段并存储在列表中
                        data = []
                        for band in range(num_bands):
                            b = image.GetRasterBand(band + 1)
                            data.append(b.ReadAsArray(i, j, local_image_width, local_image_height))

                        training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                        # 创建新的数据集以保存输出
                        driver = gdal.GetDriverByName("GTiff")
                        dst_ds = driver.Create(training_image_file_name, local_image_width, local_image_height,
                                               num_bands, gdal.GDT_Byte)

                        # 将数据写入输出数据集
                        for k, arr in enumerate(data):
                            dst_ds.GetRasterBand(k + 1).WriteArray(arr)
                        # 设置输出数据集的地理转换和投影信息
                        geotransform = list(image.GetGeoTransform())
                        geotransform[0] = geotransform[0] + i * geotransform[1]
                        geotransform[3] = geotransform[3] + j * geotransform[5]
                        dst_ds.SetGeoTransform(geotransform)
                        dst_ds.SetProjection(image.GetProjection())

                        # 保存并关闭输出数据集
                        dst_ds.FlushCache()
                        dst_ds = None

                    # 如果小块不包含标签信息，跳过保存
                    else:
                        pass
                except FileNotFoundError:
                    print('文件不存在！')
                    pass


# 图片翻转脚本
def imgFlipping(img_path, label_path):
    # 定义图像和标签文件夹路径
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)
    source_dir = '/'.join(img_path.split('/')[:-1])

    # 打印图像和标签文件数量
    print(len(image_names))
    print(len(image_labels))

    # 创建用于存放翻转后图像和标签的文件夹路径
    out_image_path = source_dir + '/images_flip'
    out_label_path = source_dir + '/labels_flip'

    # 创建用于存放翻转后图像和标签的文件夹（如果不存在）
    folder = os.path.exists(out_image_path)
    if not folder:
        os.makedirs(out_image_path)
    folder = os.path.exists(out_label_path)
    if not folder:
        os.makedirs(out_label_path)

    # 遍历每个图像文件
    for image_name in image_names:
        prefix = os.path.splitext(image_name)[0]
        suffix = image_name.split('.')[-1]
        s = os.path.join(img_path, image_name)
        tmp = cv2.imread(s)

        # 翻转图像（上下翻转、左右翻转、上下左右翻转）
        imgFlip0 = cv2.flip(tmp, 0)  # 上下翻转
        imgFlip1 = cv2.flip(tmp, 1)  # 左右翻转
        imgFlip2 = cv2.flip(tmp, -1)  # 上下左右翻转

        # 保存翻转后的图像
        cv2.imwrite(os.path.join(out_image_path, prefix + "_0." + suffix), imgFlip0)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_1." + suffix), imgFlip1)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_2." + suffix), imgFlip2)

        # 如果存在与图像对应的标签文件
        if prefix + '.txt' in image_labels:
            # 读取标签文件内容
            f = open(os.path.join(label_path, prefix + '.txt'), "r")
            lines = f.readlines()
            f.close()

            # YOLO目标检测格式转换：
            # 上下翻转，Y坐标变为1-Y
            tmp0 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(2, len(tmpK), 2):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                tmp0 += tmpK

            # 保存上下翻转后的标签
            f = open(os.path.join(out_label_path, prefix + "_0.txt"), "w")
            f.writelines(tmp0)
            f.close()

            # 左右翻转，X坐标变为1-X
            tmp1 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(1, len(tmpK), 2):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                tmp1 += tmpK

            # 保存左右翻转后的标签
            f = open(os.path.join(out_label_path, prefix + "_1.txt"), "w")
            f.writelines(tmp1)
            f.close()

            # 上下左右翻转，X和Y坐标变为1-X和1-Y
            tmp2 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(1, len(tmpK)):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                tmp2 += tmpK

            # 保存上下左右翻转后的标签
            f = open(os.path.join(out_label_path, prefix + "_2.txt"), "w")
            f.writelines(tmp2)
            f.close()


# 合并数据
def merge_data(img_path, label_path):
    # 定义源图像和标签文件夹路径
    source_dir = '/'.join(img_path.split('/')[:-1])
    # 创建用于存放合并后图像和标签的文件夹路径
    target1 = source_dir + '\\images_all'
    target2 = source_dir + '\\labels_all'

    # 如果目标文件夹不存在，则创建它们
    if not os.path.exists(target1):
        os.makedirs(target1)
    if not os.path.exists(target2):
        os.makedirs(target2)

    # 源1：将原始图像和标签复制到合并后的文件夹
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)

    for name in image_names:
        source = img_path + '/' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = label_path + '/' + label
        copyfile(source, target2 + '/' + label)

    # 源2：将翻转后的图像和标签复制到合并后的文件夹
    image_names = os.listdir(source_dir + '\\images_flip')
    image_labels = os.listdir(source_dir + '\\labels_flip')

    for name in image_names:
        source = source_dir + '\\images_flip\\' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = source_dir + '\\labels_flip\\' + label
        copyfile(source, target2 + '/' + label)


# 分割训练验证测试集
def split_train_val_test(img_path, label_path, val_percentage, test_percentage):
    # 获取根目录
    root_path = os.path.dirname(img_path)

    # 定义目标文件夹的路径
    paths = {
        "train_img": os.path.join(root_path, 'train', 'training_img'),
        "train_label": os.path.join(root_path, 'train', 'training_label'),
        "val_img": os.path.join(root_path, 'val', 'val_img'),
        "val_label": os.path.join(root_path, 'val', 'val_label'),
        "test_img": os.path.join(root_path, 'test', 'test_img'),
        "test_label": os.path.join(root_path, 'test', 'test_label')
    }

    # 创建目标文件夹（如果不存在）
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # 获取图像文件名列表，仅包括指定扩展名的文件
    img_names = [f for f in os.listdir(img_path) if f.endswith(tuple(IMAGE_EXTENSIONS))]

    # 根据验证集和测试集的百分比分割数据集
    val_img_names = set(random.sample(img_names, int(len(img_names) * val_percentage)))
    remaining_imgs = list(set(img_names) - val_img_names)
    test_img_names = set(random.sample(remaining_imgs, int(len(remaining_imgs) * test_percentage)))

    # 定义复制文件的函数
    def copy_files(img_name):
        img_name_no_ext = os.path.splitext(img_name)[0]
        old_img_path = os.path.join(img_path, img_name)
        old_label_path = os.path.join(label_path, img_name_no_ext + '.txt')

        if img_name in test_img_names:
            img_folder, label_folder = "test_img", "test_label"
        elif img_name in val_img_names:
            img_folder, label_folder = "val_img", "val_label"
        else:
            img_folder, label_folder = "train_img", "train_label"

        new_img_path = os.path.join(paths[img_folder], img_name)
        new_label_path = os.path.join(paths[label_folder], img_name_no_ext + '.txt')

        # 复制图像和标签文件到目标文件夹
        shutil.copy2(old_img_path, new_img_path)
        shutil.copy2(old_label_path, new_label_path)

    # 使用线程池并行复制文件
    with ThreadPoolExecutor() as executor:
        executor.map(copy_files, img_names)


# 生成yoyov7所需的train、test、val。path代表存images和labels的文件夹地址
def get_train_test_val(img_path, label_path, percentage):
    # 获取根目录
    source_dir = '/'.join(img_path.split('/')[:-1])

    # 获取图像和标签文件列表
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)

    print(len(image_names))
    print(len(image_labels))

    # 创建训练、测试和验证文件夹
    for name in ['train', 'test', 'val']:
        if os.path.exists(source_dir + '\\' + name):
            shutil.rmtree(source_dir + '\\' + name)  # 递归删除文件夹，即：删除非空文件夹
        os.makedirs(source_dir + '\\' + name)
        os.makedirs(source_dir + '\\' + name + '\\images')
        os.makedirs(source_dir + '\\' + name + '\\labels')

    sums = len(image_names)
    num = 0

    # 遍历图像文件并分配到训练、测试和验证集
    for name in image_names:
        suf = f".{name.split('.')[-1]}"
        source = img_path + '/' + name
        num += 1
        if num % 1000 == 0:
            print(num)

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= percentage:
            target = source_dir + '\\train\\images\\' + name
        else:
            target = source_dir + '\\val\\images\\' + name
        copyfile(source, target)

        # 如果有相应的标签文件，也复制到相应的文件夹中
        if name.replace(suf, '.txt') in image_labels:
            source = label_path + '/' + name.replace(suf, '.txt')
            if rand_rate <= percentage:
                target = source_dir + '\\train\\labels\\' + name.replace(suf, '.txt')
            else:
                target = source_dir + '\\val\\labels\\' + name.replace(suf, '.txt')

            copyfile(source, target)


# 生成文件夹
def create_folder(root_path, folder_name):
    """
    创建文件夹及其子文件夹。

    Args:
        root_path (str): 根目录路径，文件夹将在此目录下创建。
        folder_name (str): 要创建的文件夹名称。

    Returns:
        None
    """
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'images'))
        os.mkdir(os.path.join(path, 'label_txt'))


# 房地产项目训练数据生成脚本
def realestate_data_ge(root_path):
    """
    处理房地产数据，生成标签文件和剪裁图片。

    Args:
        root_path (str): 数据根目录路径。

    Returns:
        None
    """
    name_dic = {'土方开挖': "DiggingStarted", '开挖中': "DiggingStarted", '场地硬化': "FieldHardening",
                '基础': "Foundation", '垫层': "Padding",
                '主体结构施工': "MainBodyConstruct", '场地平整': "LandEven", '未拆迁房屋': "UndemolishedHouses",
                '塔吊': "TowerCrane", '主体结构封顶': "Topping", '场地清理': "SiteClearance",
                '临建及加工厂': "TempBuildings",
                '临建及加厂': "TempBuildings"}
    folder_name = 'RealEstate'
    create_folder(root_path, folder_name)
    data_path = os.path.join(root_path, '房地产数据')
    data_cities = os.listdir(data_path)
    kaiwazhong_count = 0
    jichu_count = 0
    dianceng_count = 0
    yifengding_count = 0
    weifengding_count = 0
    linjian_count = 0
    for data_city in data_cities:
        data_city_pinyin = ''.join(lazy_pinyin(data_city))
        data_city_path = os.path.join(data_path, data_city)  # 郑州房地产4目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        else:
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录
        geojson_files = os.listdir(geojson_root_path)
        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                print(f'{image_file_path} does not exist')
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']
            for feature in features:
                properties = feature['properties']
                if not (properties["name"] == '主体结构施工' or properties["name"] == '主体结构封顶' or
                        properties["name"] == '临建及加工厂' or properties["name"] == '临建及加厂' or
                        properties["name"] == '基础' or properties["name"] == '垫层' or properties[
                            "name"] == '开挖中' or
                        properties["name"] == '土方开挖'):
                    continue

                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    if properties["name"] == '土方开挖' or properties["name"] == '开挖中':
                        kaiwazhong_count += 1
                        label = 0
                    elif properties["name"] == '基础':
                        jichu_count += 1
                        label = 1
                    elif properties["name"] == '垫层':
                        dianceng_count += 1
                        label = 2
                    elif properties["name"] == '临建及加工厂' or properties["name"] == '临建及加厂':
                        linjian_count += 1
                        label = 3
                    elif properties["name"] == '主体结构封顶':
                        yifengding_count += 1
                        label = 4
                    else:
                        weifengding_count += 1
                        label = 5

                    in_ds = gdal.Open(image_file_path)  # 读取要切的原图
                    print("open tif file succeeded")
                    print(geojson_file_path)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    outbandsize = in_ds.RasterCount  # 获取数据波段数
                    print(width, height, outbandsize)
                    adfGeoTransform = in_ds.GetGeoTransform()  # 读取地理信息
                    print(adfGeoTransform)

                    geometry = feature['geometry']
                    try:
                        coordinates = geometry['coordinates']
                    except TypeError:
                        continue
                    if len(coordinates) == 0:
                        continue
                    label_res = ''
                    try:
                        for i in range(0, len(coordinates[0]) - 1):
                            lat_lon = coordinates[0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                    except:
                        for i in range(0, len(coordinates[0][0]) - 1):
                            lat_lon = coordinates[0][0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                    if label_res != '':
                        label_res = str(label) + label_res
                        out.write(label_res + '\n')
            f.close()

    print(
        f"已封顶数量：{yifengding_count}; 未封顶数量：{weifengding_count}; 临建数量：{linjian_count}; 开挖中数量：{kaiwazhong_count};"
        f"基础数量：{jichu_count}; 垫层数量：{dianceng_count}")


# 房地产项目训练数据生成脚本
def building_data_ge(root_path):
    """
    处理房地产数据中的建筑信息，生成标签文件和剪裁图片。

    Args:
        root_path (str): 数据根目录路径。

    Returns:
        None
    """
    name_dic = {'土方开挖': "DiggingStarted", '开挖中': "DiggingStarted", '场地硬化': "FieldHardening",
                '基础': "Foundation", '垫层': "Padding",
                '主体结构施工': "MainBodyConstruct", '场地平整': "LandEven", '未拆迁房屋': "UndemolishedHouses",
                '塔吊': "TowerCrane", '主体结构封顶': "Topping", '场地清理': "SiteClearance",
                '临建及加工厂': "TempBuildings",
                '临建及加厂': "TempBuildings"}
    folder_name = 'Temp_and_Large_Buildings'
    create_folder(root_path, folder_name)
    data_path = os.path.join(root_path, '房地产数据')
    data_cities = os.listdir(data_path)
    yifengding_count = 0
    weifengding_count = 0
    linjian_count = 0
    for data_city in data_cities:
        from pypinyin import lazy_pinyin
        data_city_pinyin = ''.join(lazy_pinyin(data_city))
        data_city_path = os.path.join(data_path, data_city)  # 郑州房地产4目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        else:
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录
        geojson_files = os.listdir(geojson_root_path)
        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                print(f'{image_file_path} does not exist')
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']
            for feature in features:
                properties = feature['properties']
                if not (properties["name"] == '主体结构施工' or properties["name"] == '主体结构封顶' or properties[
                    "name"] == '临建及加工厂' or properties["name"] == '临建及加厂'):
                    continue

                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    if properties["name"] == '临建及加工厂' or properties["name"] == '临建及加厂':
                        linjian_count += 1
                        label = 0
                    elif properties["name"] == '主体结构封顶':
                        yifengding_count += 1
                        label = 1
                    else:
                        weifengding_count += 1
                        label = 2

                    in_ds = gdal.Open(image_file_path)  # 读取要切的原图
                    print("open tif file succeeded")
                    print(geojson_file_path)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    outbandsize = in_ds.RasterCount  # 获取数据波段数
                    print(width, height, outbandsize)
                    adfGeoTransform = in_ds.GetGeoTransform()  # 读取地理信息
                    print(adfGeoTransform)

                    geometry = feature['geometry']
                    try:
                        coordinates = geometry['coordinates']
                    except TypeError:
                        continue
                    if len(coordinates) == 0:
                        continue
                    label_res = ''
                    try:
                        for i in range(0, len(coordinates[0]) - 1):
                            lat_lon = coordinates[0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                    except:
                        for i in range(0, len(coordinates[0][0]) - 1):
                            lat_lon = coordinates[0][0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                    if label_res != '':
                        label_res = str(label) + label_res
                        out.write(label_res + '\n')
            f.close()

    print(f"已封顶数量：{yifengding_count}; 未封顶数量：{weifengding_count}; 临建数量：{linjian_count}")


# 房地产项目训练数据生成脚本
def foundation_data_ge(root_path):
    """
    处理房地产数据中的基础和垫层信息，生成标签文件和剪裁图片。

    Args:
        root_path (str): 数据根目录路径。

    Returns:
        None
    """
    name_dic = {'土方开挖': "DiggingStarted", '开挖中': "DiggingStarted", '场地硬化': "FieldHardening",
                '基础': "Foundation", '垫层': "Padding",
                '主体结构施工': "MainBodyConstruct", '场地平整': "LandEven", '未拆迁房屋': "UndemolishedHouses",
                '塔吊': "TowerCrane", '主体结构封顶': "Topping", '场地清理': "SiteClearance",
                '临建及加工厂': "TempBuildings",
                '临建及加厂': "TempBuildings"}
    folder_name = 'Foundation_and_Padding'
    create_folder(root_path, folder_name)
    data_path = os.path.join(root_path, '房地产数据')
    data_cities = os.listdir(data_path)
    for data_city in data_cities:
        from pypinyin import lazy_pinyin
        data_city_pinyin = ''.join(lazy_pinyin(data_city))
        data_city_path = os.path.join(data_path, data_city)  # 郑州房地产4目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        else:
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录
        geojson_files = os.listdir(geojson_root_path)
        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                print(f'{image_file_path} does not exist')
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']
            for feature in features:
                properties = feature['properties']
                if not (properties["name"] == '基础' or properties["name"] == '垫层'):
                    continue

                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    if properties["name"] == '基础':
                        label = 0
                    else:
                        label = 1

                    in_ds = gdal.Open(image_file_path)  # 读取要切的原图
                    print("open tif file succeeded")
                    print(geojson_file_path)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    outbandsize = in_ds.RasterCount  # 获取数据波段数
                    print(width, height, outbandsize)
                    adfGeoTransform = in_ds.GetGeoTransform()  # 读取地理信息
                    print(adfGeoTransform)

                    geometry = feature['geometry']
                    coordinates = geometry['coordinates']
                    if len(coordinates) == 0:
                        continue
                    label_res = ''
                    try:
                        for i in range(0, len(coordinates[0]) - 1):
                            lat_lon = coordinates[0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                    except:
                        for i in range(0, len(coordinates[0][0]) - 1):
                            lat_lon = coordinates[0][0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                    if label_res != '':
                        label_res = str(label) + label_res
                        out.write(label_res + '\n')
            f.close()


# 批量将shape文件转为geojson文件
def batch_convert_shapely(root_path):
    """
    批量处理风机数据中的地理信息文件 (GeoJSON)。

    Args:
        root_path (str): 数据根目录路径。

    Returns:
        None
    """
    data_path = os.path.join(root_path, '风机数据')
    data_cities = os.listdir(data_path)
    for data_city in data_cities:
        data_city_path = os.path.join(data_path, data_city)  # 郑州房地产4目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        elif os.path.exists(os.path.join(data_city_path, '矢量')):
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录
        elif os.path.exists(os.path.join(data_city_path, '数据')):
            geojson_root_path = os.path.join(data_city_path, '数据')  # geojson目录
        else:
            print('未找到标注文件夹！')
            break
        batch_read_shapfile(geojson_root_path)


# 房地产项目训练集生成脚本
def geojson_classify(root_path):
    """
    根据地理信息文件 (GeoJSON) 中的属性信息将数据进行分类和标注。

    Args:
        root_path (str): 数据根目录路径。

    Returns:
        None
    """
    name_dic = {'土方开挖': "DiggingStarted", '开挖中': "DiggingStarted", '场地硬化': "FieldHardening",
                '基础': "Foundation", '垫层': "Padding",
                '主体结构施工': "MainBodyConstruct", '场地平整': "LandEven", '未拆迁房屋': "UndemolishedHouses",
                '塔吊': "TowerCrane", '主体结构封顶': "Topping", '场地清理': "SiteClearance",
                '临建及加工厂': "TempBuildings",
                '临建及加厂': "TempBuildings"}

    # 创建各个分类的文件夹
    for name in name_dic.values():
        create_folder(root_path, name)

    data_path = os.path.join(root_path, '房地产数据')
    data_cities = os.listdir(data_path)

    for data_city in data_cities:
        from pypinyin import lazy_pinyin
        data_city_pinyin = ''.join(lazy_pinyin(data_city))
        data_city_path = os.path.join(data_path, data_city)

        # 查找包含标注内容的目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        else:
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录

        geojson_files = os.listdir(geojson_root_path)

        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]

            if geojson_file_name == '标注范围':
                continue
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'

            if not os.path.exists(image_file_path):
                print(f'{image_file_path} does not exist')
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'

            f = open(geojson_file_path, encoding='utf8')
            res = ''

            for line in f:
                res = res + line

            res_json = json.loads(res)
            features = res_json['features']

            for feature in features:
                properties = feature['properties']
                if properties["name"] == '标注范围' or properties["name"] == '项目范围':
                    continue
                label_path = root_path + '/' + name_dic[properties['name']] + '/label_txt/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                new_image_file_path = root_path + '/' + name_dic[properties['name']] + '/images/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)

                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)

                with open(label_path, 'a') as out:
                    label = 0
                    in_ds = gdal.Open(image_file_path)
                    print("open tif file succeeded")
                    print(geojson_file_path)
                    width = in_ds.RasterXSize
                    height = in_ds.RasterYSize
                    outbandsize = in_ds.RasterCount
                    print(width, height, outbandsize)
                    adfGeoTransform = in_ds.GetGeoTransform()
                    print(adfGeoTransform)
                    print('\n')
                    geometry = feature['geometry']
                    coordinates = geometry['coordinates']

                    if len(coordinates) == 0:
                        continue
                    label_res = ''

                    try:
                        for i in range(0, len(coordinates[0]) - 1):
                            lat_lon = coordinates[0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                    except:
                        for i in range(0, len(coordinates[0][0]) - 1):
                            lat_lon = coordinates[0][0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                    if label_res != '':
                        label_res = str(label) + label_res
                        out.write(label_res + '\n')
            f.close()


# 修改数据集类别
def change_dataset_class(txt_path, class_num):
    """
    更改文本文件中标签的类别编号。

    Args:
        txt_path (str): 包含原始标签文件的目录路径。
        class_num (int): 新的类别编号。

    Returns:
        None
    """
    source_dir = '/'.join(txt_path.split('/')[:-1])
    new_folder_name = f'label_txt_{class_num}'
    new_dir = os.path.join(source_dir, new_folder_name)

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    txt_file_names = os.listdir(txt_path)

    for txt_file_name in txt_file_names:
        if not txt_file_name.endswith('.txt'):
            continue
        txt_file_path = os.path.join(txt_path, txt_file_name)
        new_txt_file_path = os.path.join(new_dir, txt_file_name)

        # 打开文件进行读取
        with open(txt_file_path, "r") as file:
            lines = file.readlines()

        # 修改每行中的第一个数字
        modified_lines = []

        for line in lines:
            parts = line.split(" ", 1)  # 在第一个空格处分割行
            if parts[0] == "0":  # 检查第一个部分是否为 "0"
                modified_lines.append(f"{class_num} " + parts[1])
            else:
                modified_lines.append(line)

        # 将修改后的行写回文件
        with open(new_txt_file_path, "w") as file:
            file.writelines(modified_lines)


# 图像数据增强脚本
def apply_clahe_to_rgb_image(bgr_image, clip_limit=2, tile_grid_size=(8, 8)):
    """
    对输入的BGR格式的RGB图像应用CLAHE（对比度受限的自适应直方图均衡化）处理。

    Args:
        bgr_image (numpy.ndarray): 输入的BGR格式的RGB图像，通常是一个三维NumPy数组。
        clip_limit (float, optional): CLAHE中的剪切限制参数。默认为2。
        tile_grid_size (tuple, optional): CLAHE中的瓦片网格大小，形式为 (rows, cols)。默认为 (8, 8)。

    Returns:
        numpy.ndarray: 经CLAHE处理后的BGR图像。
    """
    # 将RGB图像转换为Lab色彩空间
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)

    # 分离Lab图像的通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 对L通道应用CLAHE
    l_channel_clahe = clahe.apply(l_channel)

    # 将CLAHE增强的L通道与a和b通道合并
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # 转换回BGR色彩空间
    bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)

    return bgr_image_clahe


def process_image(args):
    """
    处理图像，将CLAHE（对比度受限的自适应直方图均衡化）应用于输入图像。

    Args:
        args (tuple): 一个包含以下参数的元组:
            - image_name_suf (str): 图像文件名（包含后缀）。
            - image_folder_path (str): 输入图像所在的文件夹路径。
            - enhanced_image_folder_path (str): 处理后图像的保存文件夹路径。
            - clip_limit (float): CLAHE中的剪切限制参数。
            - tile_grid_size (tuple): CLAHE中的瓦片网格大小，形式为 (rows, cols)。
    """
    image_name_suf, image_folder_path, enhanced_image_folder_path, clip_limit, tile_grid_size = args

    # 构建输入图像和处理后图像的文件路径
    image_path = os.path.join(image_folder_path, image_name_suf)
    new_image_path = os.path.join(enhanced_image_folder_path, image_name_suf)

    # 检查处理后的图像是否已经存在，如果存在则跳过处理
    if os.path.exists(new_image_path):
        return  # 如果图像已存在，则跳过处理

    # 读取输入图像为BGR格式
    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 应用CLAHE处理到BGR图像
    clahe_image = apply_clahe_to_rgb_image(bgr_image, clip_limit, tile_grid_size)

    # 保存CLAHE处理后的图像
    cv2.imwrite(new_image_path, clahe_image)


# 数据预处理脚本
def dataset_preprocessing(image_folder_path, clip_limit, tile_grid_size, num_processes=4):
    """
    对数据集中的多张图像进行预处理，包括应用CLAHE（对比度受限的自适应直方图均衡化）来增强图像的对比度。

    Args:
        image_folder_path (str): 包含输入图像的文件夹路径。
        clip_limit (float): CLAHE中的剪切限制参数。
        tile_grid_size (tuple): CLAHE中的瓦片网格大小，形式为 (rows, cols)。
        num_processes (int): 同时处理图像的进程数，默认为4。
    """
    # 创建保存增强图像的文件夹
    enhanced_image_folder_path = os.path.join(os.path.dirname(image_folder_path), 'enhanced_images')
    os.makedirs(enhanced_image_folder_path, exist_ok=True)

    # 获取文件夹中所有支持的图像文件名
    image_names_suf = [f for f in os.listdir(image_folder_path) if f.endswith(IMAGE_EXTENSIONS)]

    # 准备处理图像所需的参数
    process_args = [(name, image_folder_path, enhanced_image_folder_path, clip_limit, tile_grid_size) for name in
                    image_names_suf]

    # 使用多进程并行处理图像
    with Pool(num_processes) as p:
        p.map(process_image, process_args)

    print('图片增强完成')


# txt文件转geojson脚本
def txt_2_geojson(image_folder_path, label_folder_path, class_dict):
    """
    将文本标注文件（.txt格式）转换为GeoJSON文件格式，并将地理坐标信息添加到生成的GeoJSON文件中。

    Args:
        image_folder_path (str): 包含图像文件的文件夹路径。
        label_folder_path (str): 包含文本标注文件的文件夹路径。
        class_dict (dict): 类别字典，将类别标签映射到类别名称。

    Returns:
        None
    """
    # 创建保存GeoJSON文件的文件夹
    source_path = '/'.join(image_folder_path.split('/')[:-1])
    save_path = source_path + '/' + 'label_geojson'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    # 获取图像文件夹中的图像文件名列表
    image_names_suf = os.listdir(image_folder_path)

    # 遍历每个图像文件
    for image_name_suf in image_names_suf:
        image_name, file_extension = os.path.splitext(image_name_suf)
        if file_extension.lower() in IMAGE_EXTENSIONS:
            image_path = image_folder_path + '/' + image_name_suf
            label_path = label_folder_path + '/' + image_name + '.txt'

            # 打开图像文件以获取地理坐标信息
            gdal.AllRegister()
            dataset = gdal.Open(image_path)
            img_width, img_height = dataset.RasterXSize, dataset.RasterYSize
            adfGeoTransform = dataset.GetGeoTransform()

            # 准备生成的GeoJSON数据结构
            res_dict = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }

            # 解析文本标注文件中的标注信息
            date = image_name.split('.')[0][-8:]
            res_boxes = []
            res_labels = []
            with open(label_path, 'r') as file:
                for line in file:
                    label = int(line.strip().split()[0])
                    res_labels.append(label)
                    split_line = list(map(float, line.strip().split()[1:]))
                    new_split_line = []
                    for i in range(0, len(split_line), 2):
                        new_split_line.append([int(split_line[i] * img_width), int(split_line[i + 1] * img_height)])
                    res_boxes.append(new_split_line)

            # 根据标注信息生成GeoJSON数据
            for index in range(len(res_boxes)):
                polygon = res_boxes[index]
                label = res_labels[index]
                weight = 0
                name = class_dict[label]
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
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": []
                    }
                }
                coordinate = []
                for xy in polygon:
                    location = [xy[0] * adfGeoTransform[1] + adfGeoTransform[0],
                                xy[1] * adfGeoTransform[5] + adfGeoTransform[3]]
                    coordinate.append(location)
                coordinate.append(coordinate[0])
                feature['geometry']['coordinates'].append(coordinate)
                res_dict['features'].append(feature)

            # 将生成的GeoJSON数据写入文件
            out_dir = save_path + '/' + image_name + '.geojson'
            with open(out_dir, 'w', encoding='utf-8') as out_file:
                json.dump(res_dict, out_file, ensure_ascii=False, indent=4)


# 光伏数据生成脚本
def solar_panel_data_ge(root_path):
    """
    整理太阳能光伏建设的相关数据，并将数据按类别复制到相应的目录，同时生成包含地理信息的GeoJSON文件。

    Args:
        root_path (str): 数据根目录，包含各类别的数据子目录。

    Returns:
        None
    """
    # 定义类别名称到目录名称的映射
    name_dic = {'进场道路': "AccessRoad", '临时建筑': "TempBuilding", '施工机械': "ConstructionMachinery",
                '小型汽车': "SmallCar", '大型汽车': "LargeCar",
                '场地平整': "LandEvening", '光伏基础': 'PVFoundation', '材料堆放': 'MaterialStacking',
                '光伏支架': 'PVMount',
                '光伏面板': 'PVPanel', '箱变基础': 'PVBoxFoundation', '光伏箱变': 'PVBox', '升压站': "BoosterStation"}

    # 创建类别目录及子目录（'images'和'label_geojson'）
    for name in name_dic.values():
        path = os.path.join(root_path, name)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'images'))
            os.mkdir(os.path.join(path, 'label_geojson'))

    # 遍历数据根目录下的城市子目录
    data_path = os.path.join(root_path, '4875_光伏建设')
    data_cities = os.listdir(data_path)
    for data_city1 in data_cities:
        data_city_path1 = os.path.join(data_path, data_city1)
        data_city_path = os.path.join(data_city_path1, os.listdir(data_city_path1)[0])
        data_city_name = ''.join(lazy_pinyin(os.listdir(data_city_path1)[0]))

        # 遍历各子目录下的数据
        for item in os.listdir(data_city_path):
            full_path = os.path.join(data_city_path, item)
            if not os.path.isdir(full_path):
                continue
            image_file_path = os.path.join(full_path, '影像', os.listdir(os.path.join(full_path, '影像'))[0])
            geojson_path = os.path.join(full_path, 'geojson')

            # 遍历每个子目录下的GeoJSON文件
            for geojson_name_suf in os.listdir(geojson_path):
                if not geojson_name_suf.endswith('.geojson'):
                    continue
                geojson_file_path = os.path.join(geojson_path, geojson_name_suf)
                geojson_name = os.path.splitext(geojson_name_suf)[0]
                geojson_class = geojson_name.split('_')[-1]

                # 生成新的图片文件路径和GeoJSON文件路径
                new_image_file_path = os.path.join(root_path, name_dic[geojson_class], 'images',
                                                   name_dic[geojson_class] + '_' + data_city_name + '_' +
                                                   geojson_name.split('_')[0] + '.tif')
                new_label_path = os.path.join(root_path, name_dic[geojson_class], 'label_geojson',
                                              name_dic[geojson_class] + '_' + data_city_name + '_' +
                                              geojson_name.split('_')[0] + '.geojson')

                # 复制图像文件和GeoJSON文件
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                if not os.path.exists(new_label_path):
                    shutil.copy2(geojson_file_path, new_label_path)


class PrecisionRecallParameters:
    def __init__(self):
        self.num_of_labels = 0  # 初始化真正的正例数量为0
        self.num_of_predicts = 0  # 初始化模型预测的正例数量为0
        self.num_of_correct_predicts = 0  # 初始化模型正确预测的正例数量为0

    def calculate_pr(self):
        # 计算精确度和召回率
        precision = self.num_of_correct_predicts / self.num_of_predicts if self.num_of_predicts else 0
        recall = self.num_of_correct_predicts / self.num_of_labels if self.num_of_labels else 0
        return ("precision: %d/%d=%.2f%%, \nRecall: %d/%d=%.2f%%" % (
            self.num_of_correct_predicts, self.num_of_predicts, precision * 100, self.num_of_correct_predicts,
            self.num_of_labels, recall * 100))

    # Methods to update counts

    def add_labels(self, count):
        self.num_of_labels += count  # 增加真正的正例数量

    def add_predicts(self, count):
        self.num_of_predicts += count  # 增加模型预测的正例数量

    def add_correct_predicts(self, count):
        self.num_of_correct_predicts += count  # 增加模型正确预测的正例数量


def excel_writer(arr, file_dir):
    """
    将二维数组写入Excel文件

    参数:
    arr (list): 二维数组，要写入Excel文件的数据。
    file_dir (str): Excel文件的保存路径。

    返回:
    无
    """
    import pandas as pd
    df = pd.DataFrame(arr)  # 创建一个DataFrame对象，将二维数组转换为数据框
    df.to_excel(file_dir, index=False, header=False)  # 将数据框写入Excel文件，不包括行索引和列标题


# 统计准召
def precision_recall_calculate(label_geojson_folder_path, predict_geojson_folder_path, class_dict):
    """
    计算预测结果的精确度和召回率，并生成一些统计信息

    参数:
    label_geojson_folder_path (str): 包含标签GeoJSON文件的文件夹路径。
    predict_geojson_folder_path (str): 包含预测结果GeoJSON文件的文件夹路径。
    class_dict (dict): 包含类别名称映射的字典。

    返回:
    无
    """
    label_geojson_file_names = os.listdir(label_geojson_folder_path)
    predict_geojson_file_names = os.listdir(predict_geojson_folder_path)
    source_dir = '/'.join(label_geojson_folder_path.split('/')[:-1])
    if not os.path.exists(source_dir + '/out'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(source_dir + '/out')
    images_arr = [['']]
    for value in class_dict.values():
        images_arr[0].append(f'{value}: No. of correct predicts')
        images_arr[0].append(f'{value}: No. of labels')
        images_arr[0].append(f'{value}: No. of predicts')

    precision_recall_parameter_dict = {}
    for item in class_dict.values():
        precision_recall_parameter_dict[item] = PrecisionRecallParameters()
    for label_geojson_file_name in label_geojson_file_names:
        if not label_geojson_file_name.endswith('.geojson'):
            continue
        label_file_name_with_suf = label_geojson_file_name.split('/')[-1].split('.')
        label_file_name = label_file_name_with_suf[0]
        suf = label_file_name_with_suf[1]
        image_arr = [label_file_name] + [0] * (len(images_arr[0]) - 1)
        label_geojson_file_path = label_geojson_folder_path + '/' + label_geojson_file_name
        label_region = None
        for predict_geojson_file_name in predict_geojson_file_names:
            if label_file_name not in predict_geojson_file_name:
                continue
            predict_geojson_file_path = predict_geojson_folder_path + '/' + predict_geojson_file_name
            geojson_labels = open(label_geojson_file_path, encoding='utf8')
            res = ''
            for line in geojson_labels:
                res = res + line
            label_json = json.loads(res)
            features = label_json['features']
            label_polygons = []
            correct_labels = []
            for feature in features:
                if feature['properties']['name'] == '标注范围' or feature['properties']['name'] == 'Label Region' or \
                        feature['properties']['name'] == '项目范围':
                    label_region = Polygon(feature['geometry']['coordinates'][0]).buffer(0)
                    continue
                geometry = feature['geometry']
                label = feature['properties']['name']
                # if label == 'Padding':
                #     label = 'Foundation'
                corresponding_key = [key for key, value in class_dict.items() if value == label][0]
                correct_labels.append(label)
                precision_recall_parameter_dict[label].add_labels(1)
                arr_index = 1 + 3 * corresponding_key
                image_arr[arr_index + 1] += 1

                coordinates = geometry['coordinates']
                if len(coordinates) == 0:
                    continue
                label_polygon = []
                for i in range(0, len(coordinates[0]) - 1):
                    lat_lon = coordinates[0][i]
                    lat = lat_lon[0]
                    lon = lat_lon[1]
                    label_polygon.append([lat, lon])
                label_polygons.append(label_polygon)
            geojson_labels.close()
            geojson_predicts = open(predict_geojson_file_path, encoding='utf8')
            res = ''
            for line in geojson_predicts:
                res = res + line
            predict_json = json.loads(res)
            features = predict_json['features']
            predict_polygons = []
            predict_labels = []
            for feature in features:
                label = feature['properties']['name']
                if label == 'Padding':
                    label = 'Foundation'
                corresponding_key = [key for key, value in class_dict.items() if value == label][0]
                geometry = feature['geometry']
                coordinates = geometry['coordinates']
                if len(coordinates[0]) < 4:
                    continue
                predict_polygon = Polygon(coordinates[0]).buffer(0)
                if not predict_polygon.intersects(label_region):
                    continue

                predict_polygon = []
                for i in range(0, len(coordinates[0]) - 1):
                    lat_lon = coordinates[0][i]
                    lat = lat_lon[0]
                    lon = lat_lon[1]
                    predict_polygon.append([lat, lon])
                predict_polygons.append(predict_polygon)
                predict_labels.append(label)
                precision_recall_parameter_dict[label].add_predicts(1)
                arr_index = 1 + 3 * corresponding_key
                image_arr[arr_index + 2] += 1

            geojson_predicts.close()
            gdf2 = gpd.GeoSeries([Polygon(p).buffer(0) for p in label_polygons])
            gdf1 = gpd.GeoSeries([Polygon(p).buffer(0) for p in predict_polygons])
            spatial_index = gdf2.sindex

            flags = [0] * len(gdf1)
            for i, poly1 in enumerate(gdf1):
                predict_label = predict_labels[i]
                possible_matches_index = list(spatial_index.intersection(poly1.bounds))
                possible_matches = gdf2.iloc[possible_matches_index]
                for j, poly2 in enumerate(possible_matches):
                    correct_label = correct_labels[possible_matches_index[j]]
                    if poly1.intersects(poly2):
                        intersection_area = poly1.intersection(poly2).area
                        area1 = poly1.area
                        area2 = poly2.area
                        if (intersection_area / area1 >= 0.65 or intersection_area / area2 >= 0.65):
                            if predict_label == correct_label:
                                corresponding_key = \
                                    [key for key, value in class_dict.items() if value == predict_label][0]
                                arr_index = 1 + 3 * corresponding_key
                                image_arr[arr_index] += 1
                                precision_recall_parameter_dict[predict_label].add_correct_predicts(1)
                                flags[i] = 1
                                break  # No need to check other polygons if one is found overlapping
            out_file_path = source_dir + '/out/' + os.path.splitext(predict_geojson_file_name)[0] + '_out' + \
                            os.path.splitext(predict_geojson_file_name)[1]
            res_dict = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            for index in range(len(predict_polygons)):
                date = 0
                polygon = predict_polygons[index]
                label = 0
                weight = 0
                name = '1' if flags[index] else f'{predict_labels[index]} Incorrect'
                feature = {"type": "Feature",
                           "properties": {"Id": 0, "name": name, "date": date, "area": 0.0, "label": label, "result": 1,
                                          "XMMC": "", "HYMC": "", "weight": weight, "bz": 0},
                           "geometry": {"type": "Polygon", "coordinates": []}}
                coordinate = []
                for xy in polygon:
                    coordinate.append(xy)
                coordinate.append(coordinate[0])
                feature['geometry']['coordinates'].append(coordinate)
                res_dict['features'].append(feature)
                with open(out_file_path, 'w', encoding='utf8') as out_file:
                    json.dump(res_dict, out_file)
            images_arr.append(image_arr)
    for key in class_dict.values():
        print(f'{key}:')
        print(precision_recall_parameter_dict[key].calculate_pr())

    def natural_sort_key(s):
        """
        生成一个排序键，按自然排序方式排序，其中字符串中的数字会被当作数值来处理。

        参数:
        s (str): 待排序的字符串。

        返回:
        list: 排序键的列表。
        """
        # 使用正则表达式将字符串拆分为数字和非数字部分，并将数字部分转换为整数
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    # 使用自然排序键对2D数组进行排序
    images_arr.sort(key=lambda row: natural_sort_key(row[0]))

    excel_out_path = source_dir + '/out/' + "result.xlsx"
    excel_writer(images_arr, excel_out_path)


# 计算类别数
def class_num_calculate(txt_dir, class_dict):
    """
    统计每个类别的样本数量并打印结果。

    参数:
    txt_dir (str): 包含文本文件的文件夹路径，每个文本文件包含一行，格式为 "<类别编号> <其他信息>"。
    class_dict (dict): 包含类别名称映射的字典，其中键是类别编号，值是类别名称。

    返回:
    无
    """
    # 创建一个长度与类别字典中类别数量相等的列表，用于统计每个类别的样本数量
    num_count = [0] * len(class_dict)

    # 遍历文本文件夹中的每个文本文件
    for txt_name_and_suf in os.listdir(txt_dir):
        if not txt_name_and_suf.endswith('.txt'):
            continue
        txt_path = txt_dir + '/' + txt_name_and_suf

        # 打开文本文件并逐行读取
        with open(txt_path, 'r') as file:
            for line in file:
                # 从每行中提取类别编号，并将其转换为整数
                label = int(line.strip().split()[0])

                # 根据类别编号更新对应类别的样本数量统计
                num_count[label] += 1

    # 打印每个类别的样本数量
    for i in range(len(num_count)):
        print(f'{class_dict[i]}: {num_count[i]}')


# 风机数据生成脚本
def wind_turbine_geojson_classify(root_path):
    name_dic = {'升压站在建': "unfinished_boost_station", '升压站（在建）': "unfinished_boost_station",
                '风机基础': "wind_turbine_foundation", '进场道路': "access_road", '施工机械': "machine",
                '风机塔筒': "wind_turbine_tube", '风机叶片': "wind_turbine_blade", '风机塔架': "wind_turbine_tower",
                '风电机组': "wind_turbine", '风电施工船': "wind_turbine_vessel", '升压站建成': "finished_boost_station",
                "升压站（建成）": "finished_boost_station", "升压站": "finished_boost_station",
                "升压站(建成)": "finished_boost_station", "升压站(在建)": "unfinished_boost_station"}
    for name in name_dic.values():
        create_folder(root_path, name)
    data_path = os.path.join(root_path, '风机数据')
    data_cities = os.listdir(data_path)
    for data_city in data_cities:
        from pypinyin import lazy_pinyin
        data_city_pinyin = ''.join(lazy_pinyin(data_city))
        data_city_path = os.path.join(data_path, data_city)  # 郑州房地产4目录
        if os.path.exists(os.path.join(data_city_path, '标注内容')):
            geojson_root_path = os.path.join(data_city_path, '标注内容')
        elif os.path.exists(os.path.join(data_city_path, '矢量')):
            geojson_root_path = os.path.join(data_city_path, '矢量')  # geojson目录
        elif os.path.exists(os.path.join(data_city_path, '数据')):
            geojson_root_path = os.path.join(data_city_path, '数据')
        else:
            print("找不到标注路径！")
            break

        # batch_read_shapfile(geojson_root_path)
        geojson_files = os.listdir(geojson_root_path)
        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_name_pinyin = ''.join(lazy_pinyin(geojson_file_name))
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                print(f'{image_file_path} does not exist')
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']
            for feature in features:
                properties = feature['properties']
                if properties["name"] == '标注范围' or properties["name"] == '项目范围' or properties["name"] is None:
                    continue
                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + name_dic[properties['name']] + '/label_txt/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name_pinyin + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + name_dic[properties['name']] + '/images/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name_pinyin + '.tif'
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    # out = open(label_path, 'w', encoding='utf8')
                    label = 0

                    in_ds = gdal.Open(image_file_path)  # 读取要切的原图
                    print("open tif file succeeded")
                    print(geojson_file_path)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    outbandsize = in_ds.RasterCount  # 获取数据波段数
                    print(width, height, outbandsize)
                    # 读取图片地理信息
                    adfGeoTransform = in_ds.GetGeoTransform()
                    print(adfGeoTransform)
                    print('\n')

                    geometry = feature['geometry']
                    coordinates = geometry['coordinates']
                    if len(coordinates) == 0:
                        continue
                    label_res = ''
                    try:
                        for i in range(0, len(coordinates[0]) - 1):
                            lat_lon = coordinates[0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            # 将geojson转换为像素坐标
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                    except:
                        for i in range(0, len(coordinates[0][0]) - 1):
                            lat_lon = coordinates[0][0][i]
                            lat = lat_lon[0]
                            lon = lat_lon[1]
                            # 将geojson转换为像素坐标
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                            label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)

                    if label_res != '':
                        label_res = str(label) + label_res
                        out.write(label_res + '\n')
            f.close()


def main():
    root_path = "D:/Code/Datasets/wind_turbine/dataset20240103"
    # 批量转shp脚本
    # batch_convert_shapely(root_path)
    # 风机数据生成脚本
    # wind_turbine_geojson_classify(root_path)
    # 房地产数据生成脚本
    # geojson_classify(root_path)
    # 房地产数据生成脚本
    # foundation_data_ge(root_path)
    # 房地产数据生成脚本
    # building_data_ge(root_path)
    # 房地产数据生成脚本
    # realestate_data_ge(root_path)
    # 光伏数据生成脚本
    # solar_panel_data_ge(root)

    images_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/training_images"
    # 数据增强脚本
    # dataset_preprocessing(images_folder_path, clip_limit=2, tile_grid_size=[8, 8])

    # 模型准召自动计算
    label_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/RealEstate/label_folder"
    predict_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/RealEstate/predict_folder"
    class_dict = {0: 'Digging', 1: 'Foundation', 2: 'Temp Building', 3: 'Finished Building', 4: 'Unfinished Building'}
    # precision_recall_calculate(label_folder_path, predict_folder_path, class_dict)

    # 批量预览图片标注
    image_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/RealEstate/train/training_images"
    txt_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/RealEstate/train/training_labels"
    # show_batch_image_segment(image_path, txt_path)
    # show_batch_image_bbox(image_path, txt_path)

    # shape文件转geojson
    shp_path = u"D:/Code/gitcode/yolov8/data/济南房地产1/矢量"
    # batch_read_shapfile(shp_path)

    # json文件转yolo
    # json2txt_segment(images_folder_path, json_path)

    # mask文件转yolo
    # mask2segment(label_folder_path)

    # 将xyxy标注转yolo格式
    # txt2yolo_bbox(txt_path)

    # geojson转yolo格式脚本
    image_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/image_tif"
    geojson_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/label_geojson"
    # geojson2txt_segment(geojson_path, image_path)

    # 无重叠切割
    # split_image_v1.main()

    # 有重叠切割
    # split_image_v2.main()

    # 划分训练集验证集脚本
    img_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/Combined/images"
    label_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/Combined/label_txt_1"
    # split_train_val_test(img_path, label_path, 0.1, 0.1)

    img_path = "D:/Code/Datasets/wind_turbine/dataset20231225/wind_turbine_tube/images"
    label_path = "D:/Code/Datasets/wind_turbine/dataset20240103/wind_turbine_tube/training_labels"
    class_dict = {0: 'Digging', 1: 'Foundation', 2: 'Padding', 3: 'Temp Building', 4: 'Finished Building',
                  5: 'Unfinished Building'}
    # txt文件转geojson脚本
    # txt_2_geojson(img_path, label_path, class_dict)

    # 计算标注个数
    # class_num_calculate(label_path, class_dict)

    # 分割标注转锚框脚本
    label_path = "D:/Code/Datasets/wind_turbine/dataset20231225/wind_turbine_foundation/val/training_labels"
    # txt2yolo_bbox(label_path)

    # 更改图片分辨率
    # resize_image_aspect_ratio(img_path)

    # 修改数据集分类
    # change_dataset_class(txt_path, class_num=1)


if __name__ == '__main__':
    main()
