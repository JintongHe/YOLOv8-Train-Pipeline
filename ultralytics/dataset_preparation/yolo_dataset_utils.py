from osgeo import gdal
import geopandas as gpd
import pyproj
import math
import torch
import os
import cv2
import json
import shutil
import numpy as np
import random
from shutil import copyfile
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity
from multiprocessing import Pool
from pypinyin import lazy_pinyin
import copy
from concurrent.futures import ThreadPoolExecutor

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')


def dataset_rename(image_folder_path, label_folder_path):
    source_path = '/'.join(image_folder_path.split('/')[:-1])
    save_path = source_path + '/' + 'training_folder'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    new_image_path = save_path + '/' + 'images'
    new_label_path = save_path + '/' + 'labels'
    if not os.path.exists(new_image_path):
        os.mkdir(new_image_path)
    if not os.path.exists(new_label_path):
        os.mkdir(new_label_path)
    image_names_suf = os.listdir(image_folder_path)
    for image_name_suf in image_names_suf:
        old_image_name = os.path.splitext(image_name_suf)[0]
        new_image_name_suf = os.path.splitext(image_name_suf)[0]
        label_path = label_folder_path + '/' + old_image_name + '.txt'
        image_path = image_folder_path + '/' + image_name_suf
        img = cv2.imread(image_path)
        img_out_path = new_image_path + '/' + new_image_name_suf + '.png'

        label_out_path = new_label_path + '/' + new_image_name_suf.split('.')[0] + '.txt'
        cv2.imwrite(img_out_path, img)
        shutil.copy2(label_path, label_out_path)


# 切割图片并保留边缘部分
def split_images_segment_v3(IMAGE_FOLDER, split_sizes, LABEL_FOLDER, threshold=0.6):
    # Create training images and labels folder
    training_image_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_images'
    if not os.path.exists(training_image_folder_path):
        os.mkdir(training_image_folder_path)

    training_label_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_labels'
    if not os.path.exists(training_label_folder_path):
        os.mkdir(training_label_folder_path)
    # loop through each split size
    for split_size in split_sizes:
        # Loop through each file in the Images folder
        for filename in os.listdir(IMAGE_FOLDER):
            if not filename.endswith(IMAGE_EXTENSIONS):  # check if the file is a .tif image
                continue
            print(f"starting to process {filename} with split size {split_size}")
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image = gdal.Open(image_path)

            # Extract the base name (without extension) for the folder and naming
            basename = os.path.splitext(filename)[0]

            new_folder_path = os.path.join(IMAGE_FOLDER, basename)

            # The path of the images' corresponding labels
            label_path = LABEL_FOLDER + '/' + basename + '.txt'

            width, height, num_bands = image.RasterXSize, image.RasterYSize, image.RasterCount

            # read labels and save necessary data in labels list
            labels = []
            with open(label_path, 'r') as file:
                for line in file:
                    label_class = line.split()[0]
                    polygon = np.array(line.split()[1:]).reshape((-1, 2)).astype(float)
                    if polygon.shape[0] < 2:
                        print('invalid data, continue')
                        continue

                    polygon[:, 0] = polygon[:, 0] * width
                    polygon[:, 1] = polygon[:, 1] * height
                    # max_values = np.amax(polygon, axis=0)
                    # min_values = np.amin(polygon, axis=0)
                    # max_x, min_x, max_y, min_y = max_values[0], min_values[0], max_values[1], min_values[1]
                    labels.append([label_class, polygon])

            # Split the image into pieces of size [x, y]
            cut_width = split_size[0]
            cut_height = split_size[1]
            for w in range(0, width, cut_width):
                for h in range(0, height, cut_height):
                    # 情况1
                    w0 = w
                    h0 = h
                    w1 = min(w0 + cut_width, width)
                    h1 = min(h0 + cut_height, height)

                    # cropped_image = image[j:j + y, i:i + x]
                    local_image_width = w1 - w0
                    local_image_height = h1 - h0
                    # Save the split image with the specified format
                    cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}.png"

                    # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练集里
                    flag = 0
                    text_file = training_label_folder_path + '/' + os.path.splitext(cropped_filename)[0] + '.txt'

                    for label in labels:
                        # label = list(map(float, label.split()))
                        label_class = int(label[0])
                        polygon = Polygon(label[1])
                        if not polygon.is_valid:
                            print("Polygon is invalid:", explain_validity(polygon))
                            polygon = polygon.buffer(0)
                        bounding_box = box(*(w0, h0, w1, h1))  # (min_x, min_y, max_x, max_y)

                        if polygon.intersects(bounding_box):
                            inter_polygon = polygon.intersection(bounding_box).buffer(0)
                            if type(inter_polygon) is MultiPolygon:
                                inter_polygon = max(inter_polygon.geoms, key=lambda p: p.area)
                                if type(inter_polygon) is MultiPolygon:
                                    inter_polygon = inter_polygon.geoms[0]
                            if inter_polygon.area / polygon.area < threshold:
                                continue
                            if isinstance(inter_polygon, GeometryCollection):
                                for geom in inter_polygon.geoms:
                                    if isinstance(geom, Polygon):
                                        inter_polygon = np.array(
                                            [list(point) for point in geom.exterior.coords])
                                        break
                            else:
                                inter_polygon = np.array([list(point) for point in inter_polygon.exterior.coords])

                            # print(inter_polygon)
                            flag = 1

                            inter_polygon[:, 0] = inter_polygon[:, 0] - w0
                            inter_polygon[:, 0] = inter_polygon[:, 0] / local_image_width
                            inter_polygon[:, 1] = inter_polygon[:, 1] - h0
                            inter_polygon[:, 1] = inter_polygon[:, 1] / local_image_height
                            flattened_label = ' '.join(inter_polygon.reshape(-1).astype(str).tolist())

                            # flattened_label = ' '.join([str(item) for sublist in label_box for item in sublist])

                            with open(text_file, 'a') as f:
                                f.write(f"{label_class} {flattened_label}\n")

                    # 检测到图像包含锚框时，保存图片为训练图片
                    if flag == 1:
                        # Read each band and store in a list
                        data = []
                        for band in range(num_bands):
                            b = image.GetRasterBand(band + 1)
                            data.append(b.ReadAsArray(w0, h0, local_image_width, local_image_height))

                        training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                        # cv2.imwrite(training_image_file_name, cropped_image)

                        pic = np.zeros((local_image_height, local_image_width, 3), dtype=np.uint8)
                        for b in range(3):
                            pic[:, :, 0] = data[2]  # Blue channel
                            pic[:, :, 1] = data[1]  # Green channel
                            pic[:, :, 2] = data[0]  # Red channel
                        pic = resize_image_aspect_ratio(pic)
                        cv2.imwrite(training_image_file_name, pic)

                    # 如果图像未包含标签，跳过储存
                    else:
                        pass

                    # 情况2
                    w0 = w + int(cut_width / 2)
                    h0 = h
                    w1 = min(w0 + cut_width, width)
                    h1 = min(h0 + cut_height, height)
                    if w0 < width:
                        # cropped_image = image[j:j + y, i:i + x]
                        local_image_width = w1 - w0
                        local_image_height = h1 - h0

                        # Save the split image with the specified format
                        cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_1.png"

                        # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练集里
                        flag = 0
                        text_file = training_label_folder_path + '/' + os.path.splitext(cropped_filename)[
                            0] + '.txt'

                        for label in labels:
                            # label = list(map(float, label.split()))
                            label_class = int(label[0])
                            polygon = Polygon(label[1])
                            if not polygon.is_valid:
                                print("Polygon is invalid:", explain_validity(polygon))
                                polygon = polygon.buffer(0)
                            bounding_box = box(*(w0, h0, w1, h1))  # (min_x, min_y, max_x, max_y)

                            if polygon.intersects(bounding_box):
                                inter_polygon = polygon.intersection(bounding_box).buffer(0)
                                if type(inter_polygon) is MultiPolygon:
                                    inter_polygon = max(inter_polygon.geoms, key=lambda p: p.area)
                                    if type(inter_polygon) is MultiPolygon:
                                        inter_polygon = inter_polygon.geoms[0]
                                if inter_polygon.area / polygon.area < threshold:
                                    continue
                                if isinstance(inter_polygon, GeometryCollection):
                                    for geom in inter_polygon.geoms:
                                        if isinstance(geom, Polygon):
                                            inter_polygon = np.array(
                                                [list(point) for point in geom.exterior.coords])
                                            break
                                else:
                                    inter_polygon = np.array(
                                        [list(point) for point in inter_polygon.exterior.coords])

                                flag = 1

                                inter_polygon[:, 0] = inter_polygon[:, 0] - w0
                                inter_polygon[:, 0] = inter_polygon[:, 0] / local_image_width
                                inter_polygon[:, 1] = inter_polygon[:, 1] - h0
                                inter_polygon[:, 1] = inter_polygon[:, 1] / local_image_height
                                flattened_label = ' '.join(inter_polygon.reshape(-1).astype(str).tolist())

                                # flattened_label = ' '.join([str(item) for sublist in label_box for item in sublist])

                                with open(text_file, 'a') as f:
                                    f.write(f"{label_class} {flattened_label}\n")

                        # 检测到图像包含锚框时，保存图片为训练图片
                        if flag == 1:
                            # Read each band and store in a list
                            data = []
                            for band in range(num_bands):
                                b = image.GetRasterBand(band + 1)
                                data.append(b.ReadAsArray(w0, h0, local_image_width, local_image_height))

                            training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                            # cv2.imwrite(training_image_file_name, cropped_image)

                            pic = np.zeros((local_image_height, local_image_width, 3), dtype=np.uint8)
                            for b in range(3):
                                pic[:, :, 0] = data[2]  # Blue channel
                                pic[:, :, 1] = data[1]  # Green channel
                                pic[:, :, 2] = data[0]  # Red channel
                            pic = resize_image_aspect_ratio(pic)
                            cv2.imwrite(training_image_file_name, pic)

                        # 如果图像未包含标签，跳过储存
                        else:
                            pass

                    # 情况3
                    w0 = w
                    h0 = h + int(cut_height / 2)
                    w1 = min(w0 + cut_width, width)
                    h1 = min(h0 + cut_height, height)
                    if h0 < height:
                        # cropped_image = image[j:j + y, i:i + x]
                        local_image_width = w1 - w0
                        local_image_height = h1 - h0

                        # Save the split image with the specified format
                        cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_2.png"

                        # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练集里
                        flag = 0
                        text_file = training_label_folder_path + '/' + os.path.splitext(cropped_filename)[
                            0] + '.txt'

                        for label in labels:
                            # label = list(map(float, label.split()))
                            label_class = int(label[0])
                            polygon = Polygon(label[1])
                            if not polygon.is_valid:
                                print("Polygon is invalid:", explain_validity(polygon))

                                polygon = polygon.buffer(0)
                            # print(polygon)
                            bounding_box = box(*(w0, h0, w1, h1))  # (min_x, min_y, max_x, max_y)

                            if polygon.intersects(bounding_box):
                                inter_polygon = polygon.intersection(bounding_box)

                                if type(inter_polygon) is MultiPolygon:
                                    inter_polygon = max(inter_polygon.geoms, key=lambda p: p.area)
                                    if type(inter_polygon) is MultiPolygon:
                                        inter_polygon = inter_polygon.geoms[0]
                                if inter_polygon.area / polygon.area < threshold:
                                    continue
                                if isinstance(inter_polygon, GeometryCollection):
                                    for geom in inter_polygon.geoms:
                                        if isinstance(geom, Polygon):
                                            inter_polygon = np.array(
                                                [list(point) for point in geom.exterior.coords])
                                            break
                                else:
                                    inter_polygon = np.array([list(point) for point in inter_polygon.exterior.coords])

                                flag = 1

                                inter_polygon[:, 0] = inter_polygon[:, 0] - w0
                                inter_polygon[:, 0] = inter_polygon[:, 0] / local_image_width
                                inter_polygon[:, 1] = inter_polygon[:, 1] - h0
                                inter_polygon[:, 1] = inter_polygon[:, 1] / local_image_height
                                flattened_label = ' '.join(inter_polygon.reshape(-1).astype(str).tolist())

                                # flattened_label = ' '.join([str(item) for sublist in label_box for item in sublist])

                                with open(text_file, 'a') as f:
                                    f.write(f"{label_class} {flattened_label}\n")

                        # 检测到图像包含锚框时，保存图片为训练图片
                        if flag == 1:
                            # Read each band and store in a list
                            data = []
                            for band in range(num_bands):
                                b = image.GetRasterBand(band + 1)
                                data.append(b.ReadAsArray(w0, h0, local_image_width, local_image_height))

                            training_image_file_name = os.path.join(training_image_folder_path,
                                                                    cropped_filename)
                            # cv2.imwrite(training_image_file_name, cropped_image)

                            pic = np.zeros((local_image_height, local_image_width, 3), dtype=np.uint8)
                            for b in range(3):
                                pic[:, :, 0] = data[2]  # Blue channel
                                pic[:, :, 1] = data[1]  # Green channel
                                pic[:, :, 2] = data[0]  # Red channel
                            pic = resize_image_aspect_ratio(pic)
                            cv2.imwrite(training_image_file_name, pic)

                        # 如果图像未包含标签，跳过储存
                        else:
                            pass

                    # 情况4
                    w0 = w + int(cut_width / 2)
                    h0 = h + int(cut_height / 2)
                    w1 = min(w0 + cut_width, width)
                    h1 = min(h0 + cut_height, height)
                    if h0 < height and w0 < width:
                        # cropped_image = image[j:j + y, i:i + x]
                        local_image_width = w1 - w0
                        local_image_height = h1 - h0

                        # Save the split image with the specified format
                        cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_3.png"

                        # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练集里
                        flag = 0
                        text_file = training_label_folder_path + '/' + os.path.splitext(cropped_filename)[
                            0] + '.txt'

                        for label in labels:
                            # label = list(map(float, label.split()))
                            label_class = int(label[0])
                            polygon = Polygon(label[1])
                            if not polygon.is_valid:
                                print("Polygon is invalid:", explain_validity(polygon))
                                polygon = polygon.buffer(0)
                            bounding_box = box(*(w0, h0, w1, h1))  # (min_x, min_y, max_x, max_y)

                            if polygon.intersects(bounding_box):
                                inter_polygon = polygon.intersection(bounding_box).buffer(0)
                                if type(inter_polygon) is MultiPolygon:
                                    inter_polygon = max(inter_polygon.geoms, key=lambda p: p.area)
                                    if type(inter_polygon) is MultiPolygon:
                                        inter_polygon = inter_polygon.geoms[0]
                                if inter_polygon.area / polygon.area < threshold:
                                    continue
                                if isinstance(inter_polygon, GeometryCollection):
                                    for geom in inter_polygon.geoms:
                                        if isinstance(geom, Polygon):
                                            inter_polygon = np.array(
                                                [list(point) for point in geom.exterior.coords])
                                            break
                                else:
                                    inter_polygon = np.array(
                                        [list(point) for point in inter_polygon.exterior.coords])

                                flag = 1

                                inter_polygon[:, 0] = inter_polygon[:, 0] - w0
                                inter_polygon[:, 0] = inter_polygon[:, 0] / local_image_width
                                inter_polygon[:, 1] = inter_polygon[:, 1] - h0
                                inter_polygon[:, 1] = inter_polygon[:, 1] / local_image_height
                                flattened_label = ' '.join(inter_polygon.reshape(-1).astype(str).tolist())

                                # flattened_label = ' '.join([str(item) for sublist in label_box for item in sublist])

                                with open(text_file, 'a') as f:
                                    f.write(f"{label_class} {flattened_label}\n")

                        # 检测到图像包含锚框时，保存图片为训练图片
                        if flag == 1:
                            # Read each band and store in a list
                            data = []
                            for band in range(num_bands):
                                b = image.GetRasterBand(band + 1)
                                data.append(b.ReadAsArray(w0, h0, local_image_width, local_image_height))

                            training_image_file_name = os.path.join(training_image_folder_path,
                                                                    cropped_filename)
                            # cv2.imwrite(training_image_file_name, cropped_image)

                            pic = np.zeros((local_image_height, local_image_width, 3), dtype=np.uint8)
                            for b in range(3):
                                pic[:, :, 0] = data[2]  # Blue channel
                                pic[:, :, 1] = data[1]  # Green channel
                                pic[:, :, 2] = data[0]  # Red channel
                            pic = resize_image_aspect_ratio(pic)
                            cv2.imwrite(training_image_file_name, pic)

                        # 如果图像未包含标签，跳过储存
                        else:
                            pass


def resize_image_aspect_ratio(img, new_size=640):
    # img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Determine the scaling factor
    if h > w:
        scale = new_size / h
        new_h, new_w = new_size, int(w * scale)
    else:
        scale = new_size / w
        new_h, new_w = int(h * scale), new_size

    # Resize the image
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
    json_file_names = os.listdir(geojson_path)
    source_dir = '/'.join(geojson_path.split('/')[:-1])
    if not os.path.exists(source_dir + '/Label_txt'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(source_dir + '/Label_txt')
    for json_file_name in json_file_names:
        if not json_file_name.endswith('.geojson'):
            continue
        file_name_with_suf = json_file_name.split('/')[-1].split('.')
        file_name = file_name_with_suf[0]
        suf = file_name_with_suf[1]
        print(json_file_name)
        # print(file_name)
        # print(suf)
        geojson_file_path = geojson_path + '/' + json_file_name
        image_file_path = image_path + '/' + file_name + '.tif'
        label_path = source_dir + '/Label_txt/' + file_name + '.txt'
        # img = cv2.imread(image_path)
        # img_height, img_width, _ = img.shape
        with open(label_path, 'w') as out:
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
            # 读取geojson信息
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']

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
                        # 将geojson转换为像素坐标
                        x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                        y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                        label_res = label_res + ' ' + str(x / width) + ' ' + str(y / height)
                except:
                    for i in range(0, len(coordinates[0]) - 1):
                        # for lat_lon in coordinate[-2:]:
                        lat_lon = coordinates[0][i]
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


# 批量将shp文件转为geojson文件
def batch_read_shapfile(shp_path):
    # source_dir = r'E:/work/data/car20230705/mark'
    source_dir = shp_path  # '/'.join(json_path.split('/')[:-1])
    file_names = os.listdir(shp_path)

    for file_name in file_names:
        if not file_name.endswith('.shp'):
            continue
        target_dir = shp_path
        target_filename = os.path.basename(file_name).split(".")[0]
        print(source_dir + '/' + file_name)
        geojson_file = target_dir + '/' + target_filename + ".geojson"
        if os.path.exists(geojson_file):
            continue
        print(geojson_file)
        data = gpd.read_file(source_dir + '/' + file_name)
        print(data)

        data.crs = 'EPSG:4326'
        data.to_file(geojson_file, driver="GeoJSON")
        # gbk格式转换为utf-8格式
        with open(geojson_file, 'r', encoding='gbk') as f:
            content = f.read()
        with open(geojson_file, 'w', encoding='utf8') as f:
            f.write(content)


# 将json格式文件批量转为txt格式文件
def json2txt_segment(image_folder_path, json_path):
    source_dir = '/'.join(image_folder_path.split('/')[:-1])
    txt_folder_path = source_dir + '/Label_txt'
    # Create a new folder with the image name
    if not os.path.exists(txt_folder_path):
        os.mkdir(txt_folder_path)
    else:
        shutil.rmtree(txt_folder_path)
        os.makedirs(txt_folder_path)

    image_names = os.listdir(image_folder_path)
    image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))
    with open(json_path, "r") as file:
        data = json.load(file)
    annotations = data["annotations"]
    annotations = sorted(annotations, key=lambda x: int(x['image_id']), reverse=True)
    # for annotation in annotations:
    #     print(annotation)
    for image in image_names:
        image_name, file_extension = os.path.splitext(image)
        if file_extension.lower() not in IMAGE_EXTENSIONS:
            continue

        image_name = str(int(image_name))
        new_image_path = image_folder_path + '/' + image_name + file_extension
        img_path = image_folder_path + '/' + image
        os.rename(img_path, new_image_path)
        print("started processing image with id {}".format(image_name))
        image = cv2.imread(new_image_path)
        # json_path = json_folder_path + '/' + image_name + '.json'
        txt_path = txt_folder_path + '/' + image_name + '.txt'
        # Load the JSON file

        img_height, img_width, _ = image.shape  # Replace with your image's dimensions

        with open(txt_path, "w") as out_file:
            while True:
                # print(image_name, annotations[-1]["image_id"])
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
                        # print("Segmentation for image_id {}: {}".format(image_name, annotations[-1]["segmentation"]))
                        annotations.pop()
                    else:
                        break
                except IndexError:
                    print("Finished processing!")
                    break


# 将mask格式文件批量转为txt格式文件
def mask2contour(label_folder_path):
    source_dir = '/'.join(label_folder_path.split('/')[:-1])
    txt_folder_path = source_dir + '/label_txt'
    if not os.path.exists(txt_folder_path):
        os.mkdir(txt_folder_path)
    label_names = os.listdir(label_folder_path)
    for label in label_names:
        label_name, suf = os.path.splitext(label)
        if suf.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_folder_path + '/' + label

        # Read the label TIFF image
        dataset = gdal.Open(label_path)
        label_img = dataset.ReadAsArray().T  # Transpose to match OpenCV shape expectations
        # print(label_img.shape)
        img_height, img_width = label_img.shape
        # label_img = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
        # Convert to an 8-bit image (you may need to adjust this based on your specific case)
        # label_img_8bit = np.uint8(label_img)
        # Threshold to convert to a binary image
        _, thresh = cv2.threshold(label_img, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        txt_file_path = txt_folder_path + '/' + label_name + '.txt'

        with open(txt_file_path, 'w') as f:
            for contour in contours:
                contour = contour.astype(float)
                contour = np.reshape(contour, (-1, 2))
                first_col = contour[:, 0].copy()

                # Swap the first and the third columns
                contour[:, 0] = contour[:, 1]
                contour[:, 1] = first_col
                contour[:, 0] = contour[:, 0] / img_width
                contour[:, 1] = contour[:, 1] / img_height
                # print(contour)
                contour = np.reshape(contour, (-1))
                # print(contour)
                line = ' '.join(map(str, contour))
                line = f'0 {line}'
                print(line)
                f.write(line + '\n')

                # Create an output image
                # output_img = np.zeros_like(label_img_8bit)
                # #
                # # # Draw contours on the output image
                # cv2.drawContours(output_img, contours, -1, 255, 1)  # 255 is the color, 1 is the thickness
                #
                # # Display the image (optional)
                # cv2.imshow('Contours', output_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    # Save the output image (optional)
    # cv2.imwrite('contours.tif', output_img)


def geojson2txt_bbox(geojson_path):
    json_file_names = os.listdir(geojson_path)
    source_dir = '/'.join(geojson_path.split('/')[:-1])
    if not os.path.exists(source_dir + '/Label_txt'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(source_dir + '/Label_txt')
    for json_file_name in json_file_names:
        if not json_file_name.endswith('.geojson'):
            continue
        file_name_with_suf = json_file_name.split('/')[-1].split('.')
        file_name = file_name_with_suf[0]
        suf = file_name_with_suf[1]
        print(json_file_name)
        # print(file_name)
        # print(suf)
        geojson_path = source_dir + '/Labels/' + file_name + '.geojson'
        image_path: str = source_dir + '/Images/' + file_name + '.tif'
        label_path = source_dir + '/Label_txt/' + file_name + '.txt'
        with open(label_path, 'w') as out:
            # out = open(label_path, 'w', encoding='utf8')
            label = 0

            in_ds = gdal.Open(image_path)  # 读取要切的原图
            print("open tif file succeeded")
            print(geojson_path)
            width = in_ds.RasterXSize  # 获取数据宽度
            height = in_ds.RasterYSize  # 获取数据高度
            outbandsize = in_ds.RasterCount  # 获取数据波段数
            print(width, height, outbandsize)
            # 读取图片地理信息
            adfGeoTransform = in_ds.GetGeoTransform()
            print(adfGeoTransform)
            print('\n')
            # 读取geojson信息
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
                        # for lat_lon in coordinate[-2:]:
                        lat_lon = coordinate[i]
                        lat = lat_lon[0]
                        lon = lat_lon[1]
                        # 将geojson转换为像素坐标
                        x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                        y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                        label_res = label_res + ' ' + str(x) + ' ' + str(y)

                    lat_lon = coordinate[0]
                    lat = lat_lon[0]
                    lon = lat_lon[1]
                    # 将geojson转换为像素坐标
                    x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                    y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])

                    label_res = label_res + ' ' + str(x) + ' ' + str(y)

                if label_res != '':
                    label_res = str(label) + label_res
                    out.write(label_res + '\n')

            f.close()


# 将segment格式的txt文件转为yolo格式
def txt2yolo_bbox(txt_path):
    source_dir = os.path.dirname(txt_path)
    new_folder = source_dir + '/txt_bbox'
    os.makedirs(new_folder, exist_ok=True)
    txt_files = os.listdir(txt_path)
    for txt_file in txt_files:
        if not txt_file.endswith('.txt'):
            continue
        txt_file_name = txt_file.split('.')[0]
        txt_file_path = txt_path + '/' + txt_file

        array = np.loadtxt(txt_file_path, dtype=float)

        if len(array.shape) == 1:
            array = array.reshape(1, -1)
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
        #
        #
        # array = np.delete(array, [3, 4, 7, 8], axis=1)
        # array[:, 3] = array[:, 3] - array[:, 1]
        # array[:, 4] = array[:, 4] - array[:, 2]
        # array = array.astype(float)
        # array[:, 1] = array[:, 1] / img_width
        # array[:, 2] = array[:, 2] / img_height
        # array[:, 3] = array[:, 3] / img_width
        # array[:, 4] = array[:, 4] / img_height
        # array[:, 0] = array[:, 0].astype(int)
        # print(array)
        out_file_path = new_folder + '/' + txt_file
        np.savetxt(out_file_path, new_array, fmt=['%d', '%f', '%f', '%f', '%f'])


# 将segment格式的txt文件转为obb格式
def segment2obb(txt_path):
    source_dir = os.path.dirname(txt_path)
    new_folder = source_dir + '/txt_bbox'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    txt_files = os.listdir(txt_path)
    for txt_file in txt_files:
        if not txt_file.endswith('.txt'):
            continue
        txt_file_name = txt_file.split('.')[0]
        txt_file_path = txt_path + '/' + txt_file

        array = np.loadtxt(txt_file_path, dtype=float)
        array = array.reshape(-1, 9).tolist()
        new_array = []
        for line in array:
            new_line = [int(line[0])]
            min_y = float('inf')
            min_y_idx = -1
            for i in range(2, len(line), 2):
                y_val = line[i]
                if y_val < min_y:
                    min_y = y_val
                    min_y_idx = i
            new_line = new_line + line[min_y_idx - 1:] + line[1:min_y_idx - 1]
            new_array.append(new_line)
        new_array = np.array(new_array)

        # if len(array.shape) == 1:
        #     array = array.reshape((1, -1))
        # new_array = array[:, 0].astype(int).reshape((-1, 1))
        # x_array = array[:, 1::2]
        # y_array = array[:, 2::2]
        # x_array_max = np.max(x_array, axis=1).reshape((-1, 1))
        # x_array_min = np.min(x_array, axis=1).reshape((-1, 1))
        # y_array_max = np.max(y_array, axis=1).reshape((-1, 1))
        # y_array_min = np.min(y_array, axis=1).reshape((-1, 1))
        # new_array = np.hstack((new_array, x_array_min, y_array_min, x_array_max, y_array_min, x_array_max, y_array_max, x_array_min, y_array_max))
        out_file_path = new_folder + '/' + txt_file

        np.savetxt(out_file_path, new_array, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])


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


def show_batch_image_bbox(image_path, txt_file):
    source_dir = '/'.join(image_path.split('/')[:-1])
    show_path = source_dir + '/out'
    if not os.path.exists(show_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(show_path)
    image_files = os.listdir(image_path)
    for image_file in image_files:
        if not image_file.endswith(IMAGE_EXTENSIONS):
            continue
        image_name = os.path.splitext(image_file)[0]
        image_file_path = image_path + '/' + image_file
        txt_file_path = txt_file + '/' + image_name + '.txt'
        # path = image_path.replace('/' + image_name, '')
        #
        suf = os.path.splitext(image_file)[1]
        # show_path = path + image_name.replace('.' + suf, '_show')

        # image_name = image_path.split('/')[-1]
        # suf = image_name.split('.')[1]
        img = cv2.imread(image_file_path)
        img_height, img_width, _ = img.shape
        res_label = [0]
        res_box = []
        with open(txt_file_path, 'r') as file:
            for line in file:
                # print(line.strip().split())
                split_line = list(map(float, line.strip().split()[1:]))
                split_line[0] = int(split_line[0] * img_width - split_line[2] * img_width / 2)
                split_line[1] = int(split_line[1] * img_height - split_line[3] * img_height / 2)
                split_line[2] = int(split_line[2] * img_width + split_line[0])
                split_line[3] = int(split_line[3] * img_height + split_line[1])
                res_box.append(split_line)
                # res_box.append(
                #     [[split_line[0], split_line[1]], [split_line[2], split_line[3]], [split_line[4], split_line[5]],
                #      [split_line[6], split_line[7]]])

        box_len = len(res_box)
        # print(split_line)
        # print(res_box)
        # img_contour = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        # img_contour = cv2.drawContours(img_contour, contours1, -1, (255, 0, 0), 1)
        colors = [random.randint(0, 255) for _ in range(3)]
        for i in range(box_len):
            label = res_label[0]
            xyxy = res_box[i]  # [res_box[i][0][0], res_box[i][0][1], res_box[i][2][0], res_box[i][2][1]]
            plot_one_box(xyxy, img, label=f'{label} ', color=colors, line_thickness=2)
        cv2.imwrite(os.path.join(show_path, image_name + '_contour.' + suf), img)
        print('已生成图片保存至{}'.format(os.path.join(show_path, image_name + '_contour.' + suf)))


def normalize_bbox_coords(xyxy, image_width, image_height):
    # xyxy = [[40,50,60,70]]
    gn = torch.tensor((image_height, image_width, 3))[[1, 0, 1, 0]]  # normalization gain whwh
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
    return xywh


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def split_images_bbox(IMAGE_FOLDER, split_size, LABEL_FOLDER):
    # Create training images and labels folder
    training_image_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_images'
    if not os.path.exists(training_image_folder_path):
        os.mkdir(training_image_folder_path)
    training_label_folder_path = '/'.join(IMAGE_FOLDER.split('/')[:-1]) + '/training_labels'
    if not os.path.exists(training_label_folder_path):
        os.mkdir(training_label_folder_path)

    # Loop through each file in the Images folder
    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.endswith(IMAGE_EXTENSIONS):  # check if the file is a .tif image
            continue
        print(f"starting to process {filename} with split size {split_size}")
        image_file_path = os.path.join(IMAGE_FOLDER, filename)
        image = gdal.Open(image_file_path)

        # Extract the base name (without extension) for the folder and naming
        basename = os.path.splitext(filename)[0]
        new_folder_path = os.path.join(IMAGE_FOLDER, basename)

        # The path of the images' corresponding labels
        label_path = LABEL_FOLDER + '/' + basename + '.txt'

        width, height, num_bands = image.RasterXSize, image.RasterYSize, image.RasterCount

        # read labels and save necessary data in labels list
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                label_class = int(line.strip().split()[0])
                label_box = list(map(float, line.strip().split()[1:]))
                label_box[0] = int(label_box[0] * width)
                label_box[1] = int(label_box[1] * height)
                label_box[2] = int(label_box[2] * width + label_box[0])
                label_box[3] = int(label_box[3] * height + label_box[1])
                # label_box[:, 0] = label_box[:, 0] * width
                # label_box[:, 1] = label_box[:, 1] * height
                # max_values = np.amax(label_box, axis=0)
                # min_values = np.amin(label_box, axis=0)
                # max_x, min_x, max_y, min_y = max_values[0], min_values[0], max_values[1], min_values[1]
                labels.append([label_class, label_box])

        # Split the image into pieces of size [x, y]
        x = split_size[0]
        y = split_size[1]
        for i in range(0, width, x):
            for j in range(0, height, y):

                # cropped_image = image[j:j + y, i:i + x]
                local_image_width = min(x, width - i)
                local_image_height = min(y, height - j)

                # Save the split image with the specified format
                cropped_filename = f"{basename}_{x}by{y}_{i // x + 1}_{j // y + 1}.jpg"

                # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练集里
                flag = 0
                text_file = training_label_folder_path + '/' + cropped_filename.split('.')[0] + '.txt'
                try:
                    for label in labels:
                        # label = list(map(float, label.split()))
                        label_class = label[0]
                        label_box = label[1]

                        if label_box[0] > i and label_box[2] < i + x and label_box[1] > j and label_box[3] < j + y:
                            flag = 1

                            local_xyxy = [label_box[0] - i, label_box[1] - j, label_box[2] - i, label_box[3] - j]
                            # print(local_xyxy)
                            xywh = normalize_bbox_coords(local_xyxy, local_image_width, local_image_height)
                            # print(xywh)
                            with open(text_file, 'a') as f:
                                f.write(f"{label_class} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

                    # 检测到图像包含锚框时，保存图片为训练图片
                    if flag == 1:
                        # Read each band and store in a list
                        data = []
                        for band in range(num_bands):
                            b = image.GetRasterBand(band + 1)
                            data.append(b.ReadAsArray(i, j, local_image_width, local_image_height))

                        training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                        # cv2.imwrite(training_image_file_name, cropped_image)
                        # Create a new dataset for output
                        driver = gdal.GetDriverByName("GTiff")
                        dst_ds = driver.Create(training_image_file_name, local_image_width, local_image_height,
                                               num_bands, gdal.GDT_Byte)

                        # Write data to the output dataset
                        for k, arr in enumerate(data):
                            dst_ds.GetRasterBand(k + 1).WriteArray(arr)
                        # Set the geotransform and projection of the output dataset
                        geotransform = list(image.GetGeoTransform())
                        geotransform[0] = geotransform[0] + i * geotransform[1]
                        geotransform[3] = geotransform[3] + j * geotransform[5]
                        dst_ds.SetGeoTransform(geotransform)
                        dst_ds.SetProjection(image.GetProjection())

                        # Save and close the output dataset
                        dst_ds.FlushCache()
                        dst_ds = None

                    # 如果图像未包含标签，跳过储存
                    else:
                        pass
                        # training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                        # cv2.imwrite(training_image_file_name, cropped_image)
                except FileNotFoundError:
                    print('file not found!')
                    pass


def imgFlipping(img_path, label_path):
    # path = 'E:/work/python/ai/fan/datasets'
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)
    source_dir = '/'.join(img_path.split('/')[:-1])
    print(len(image_names))
    print(len(image_labels))

    out_image_path = source_dir + '/images_flip'
    out_label_path = source_dir + '/labels_flip'
    folder = os.path.exists(out_image_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_image_path)
    folder = os.path.exists(out_label_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_label_path)

    for image_name in image_names:
        prefix = os.path.splitext(image_name)[0]
        suffix = image_name.split('.')[-1]
        s = os.path.join(img_path, image_name)
        tmp = cv2.imread(s)
        # 翻转图像
        imgFlip0 = cv2.flip(tmp, 0)  # 上下
        imgFlip1 = cv2.flip(tmp, 1)  # 左右
        imgFlip2 = cv2.flip(tmp, -1)  # 上下左右
        cv2.imwrite(os.path.join(out_image_path, prefix + "_0." + suffix), imgFlip0)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_1." + suffix), imgFlip1)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_2." + suffix), imgFlip2)

        if prefix + '.txt' in image_labels:
            # f = open(os.path.join(files, filesDir[i].replace(".bmp",".txt")),"r")
            f = open(os.path.join(label_path, prefix + '.txt'), "r")
            lines = f.readlines()
            f.close()
            # 以下为YOLO目标检测格式转换
            # 上下翻转，Y坐标变化为1-Y
            tmp0 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(2, len(tmpK), 2):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp0 += tmpK

            f = open(os.path.join(out_label_path, prefix + "_0.txt"), "w")
            f.writelines(tmp0, )
            f.close()

            # 左右翻转，X坐标变化为1-X
            tmp1 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(1, len(tmpK), 2):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp1 += tmpK
            f = open(os.path.join(out_label_path, prefix + "_1.txt"), "w")
            f.writelines(tmp1, )
            f.close()

            # 上下左右翻转，X坐标变化为1-X，Y坐标变化为1-Y
            tmp2 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                for num in range(1, len(tmpK)):
                    tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp2 += tmpK
            f = open(os.path.join(out_label_path, prefix + "_2.txt"), "w")
            f.writelines(tmp2, )
            f.close()


# 合并数据
def merge_data(img_path, label_path):
    # path = 'E:/work/python/ai/fan/datasets'
    source_dir = '/'.join(img_path.split('/')[:-1])
    target1 = source_dir + '//images_all'
    target2 = source_dir + '//labels_all'

    if not os.path.exists(target1):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(target1)
    if not os.path.exists(target2):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(target2)
    # 源1
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)

    for name in image_names:
        source = img_path + '/' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = label_path + '/' + label
        copyfile(source, target2 + '/' + label)

    # 源2
    image_names = os.listdir(source_dir + '//images_flip')
    image_labels = os.listdir(source_dir + '//labels_flip')
    for name in image_names:
        source = source_dir + '//images_flip//' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = source_dir + '//labels_flip//' + label
        copyfile(source, target2 + '/' + label)


def split_train_val_test(img_path, label_path, val_percentage, test_percentage):
    root_path = os.path.dirname(img_path)
    paths = {
        "train_img": os.path.join(root_path, 'train', 'training_img'),
        "train_label": os.path.join(root_path, 'train', 'training_label'),
        "val_img": os.path.join(root_path, 'val', 'val_img'),
        "val_label": os.path.join(root_path, 'val', 'val_label'),
        "test_img": os.path.join(root_path, 'test', 'test_img'),
        "test_label": os.path.join(root_path, 'test', 'test_label')
    }

    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    img_names = [f for f in os.listdir(img_path) if f.endswith(tuple(IMAGE_EXTENSIONS))]

    # Splitting for test set
    val_img_names = set(random.sample(img_names, int(len(img_names) * val_percentage)))
    remaining_imgs = list(set(img_names) - val_img_names)

    # Splitting for validation set
    test_img_names = set(random.sample(remaining_imgs, int(len(remaining_imgs) * test_percentage)))

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

        shutil.copy2(old_img_path, new_img_path)
        shutil.copy2(old_label_path, new_label_path)

    with ThreadPoolExecutor() as executor:
        executor.map(copy_files, img_names)


# 生成yoyov7所需的train、test、val。path代表存images和labels的地方
def get_train_test_val(img_path, label_path, percentage):
    # path = 'E://work//python//ai//fan//datasets'
    source_dir = '/'.join(img_path.split('/')[:-1])
    image_names = os.listdir(img_path)
    image_labels = os.listdir(label_path)

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    for name in ['train', 'test', 'val']:
        if os.path.exists(source_dir + '//' + name):
            shutil.rmtree(source_dir + '//' + name)  # 递归删除文件夹，即：删除非空文件夹
        os.makedirs(source_dir + '//' + name)
        os.makedirs(source_dir + '//' + name + '//images')
        os.makedirs(source_dir + '//' + name + '//labels')

    sums = len(image_names)
    num = 0
    for name in image_names:
        suf = f".{name.split('.')[-1]}"
        source = img_path + '/' + name
        num += 1
        if num % 1000 == 0:
            print(num)

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= percentage:
            target = source_dir + '//train//images//' + name
        # elif rand_rate <= 0.9:
        #     target = path + '//test//images//' + name
        else:
            target = source_dir + '//val//images//' + name
        copyfile(source, target)

        if name.replace(suf, '.txt') in image_labels:
            source = label_path + '/' + name.replace(suf, '.txt')
            if rand_rate <= percentage:
                target = source_dir + '//train//labels//' + name.replace(suf, '.txt')
            # elif rand_rate <= 0.9:
            #     target = path + '//test//labels//' + name.replace('.jpg', '.txt')
            else:
                target = source_dir + '//val//labels//' + name.replace(suf, '.txt')

            copyfile(source, target)


# root_path = u"D:/Code/gitcode/yolov8/data/Real_Estate/focus"

def create_folder(root_path, folder_name):
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'images'))
        os.mkdir(os.path.join(path, 'label_txt'))


def realestate_data_ge(root_path):
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
        # batch_read_shapfile(geojson_root_path)
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
                # if properties["name"] == '标注范围' or properties["name"] == '项目范围':
                #     continue
                # 主体结构施工
                # 主体结构封顶
                # 临建及加工厂
                # 临建及加厂
                if not (properties["name"] == '主体结构施工' or properties["name"] == '主体结构封顶' or
                        properties["name"] == '临建及加工厂' or properties["name"] == '临建及加厂' or
                        properties["name"] == '基础' or properties["name"] == '垫层' or properties[
                            "name"] == '开挖中' or
                        properties["name"] == '土方开挖'):
                    continue

                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    # label = 0
                    # out = open(label_path, 'w', encoding='utf8')
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
                    # 读取图片地理信息
                    adfGeoTransform = in_ds.GetGeoTransform()
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

    print(
        f"已封顶数量：{yifengding_count}; 未封顶数量：{weifengding_count}; 临建数量：{linjian_count}; 开挖中数量：{kaiwazhong_count};"
        f"基础数量：{jichu_count}; 垫层数量：{dianceng_count}")


def building_data_ge(root_path):
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
        # batch_read_shapfile(geojson_root_path)
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
                # if properties["name"] == '标注范围' or properties["name"] == '项目范围':
                #     continue
                # 主体结构施工
                # 主体结构封顶
                # 临建及加工厂
                # 临建及加厂
                if not (properties["name"] == '主体结构施工' or properties["name"] == '主体结构封顶' or properties[
                    "name"] == '临建及加工厂' or properties["name"] == '临建及加厂'):
                    continue

                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    # label = 0
                    # out = open(label_path, 'w', encoding='utf8')
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
                    # 读取图片地理信息
                    adfGeoTransform = in_ds.GetGeoTransform()
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

    print(f"已封顶数量：{yifengding_count}; 未封顶数量：{weifengding_count}; 临建数量：{linjian_count}")


def foundation_data_ge(root_path):
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
        # batch_read_shapfile(geojson_root_path)
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
                # if properties["name"] == '标注范围' or properties["name"] == '项目范围':
                #     continue

                if not (properties["name"] == '基础' or properties["name"] == '垫层'):
                    continue

                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + folder_name + '/label_txt/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + folder_name + '/images/' + folder_name + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    # out = open(label_path, 'w', encoding='utf8')
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
                    # 读取图片地理信息
                    adfGeoTransform = in_ds.GetGeoTransform()
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


def batch_convert_shapely(root_path):
    data_path = os.path.join(root_path, '水泥厂数据')
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


def geojson_classify(root_path):
    name_dic = {'土方开挖': "DiggingStarted", '开挖中': "DiggingStarted", '场地硬化': "FieldHardening",
                '基础': "Foundation", '垫层': "Padding",
                '主体结构施工': "MainBodyConstruct", '场地平整': "LandEven", '未拆迁房屋': "UndemolishedHouses",
                '塔吊': "TowerCrane", '主体结构封顶': "Topping", '场地清理': "SiteClearance",
                '临建及加工厂': "TempBuildings",
                '临建及加厂': "TempBuildings"}
    for name in name_dic.values():
        create_folder(root_path, name)
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
        # batch_read_shapfile(geojson_root_path)
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
                if properties["name"] == '标注范围' or properties["name"] == '项目范围':
                    continue
                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + name_dic[properties['name']] + '/label_txt/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + name_dic[properties['name']] + '/images/' + name_dic[
                    properties['name']] + '_' + data_city_pinyin + '_' + geojson_file_name + '.tif'
                print(image_file_path)
                print(new_image_file_path)
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


root_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231110"


# geojson_classify(root_path)

def change_dataset_class(txt_path, class_num):
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
        # Open the file for reading
        with open(txt_file_path, "r") as file:
            lines = file.readlines()

        # Modify the first number in each line
        modified_lines = []
        for line in lines:
            parts = line.split(" ", 1)  # Split the line at the first space
            if parts[0] == "0":  # Check if the first part is "0"
                modified_lines.append(f"{class_num} " + parts[1])
            else:
                modified_lines.append(line)

        # Write the modified lines back to the file
        with open(new_txt_file_path, "w") as file:
            file.writelines(modified_lines)


def split_image_large(image_path, split_size):
    # Create training images and labels folder
    training_image_folder_path = '/'.join(image_path.split('/')[:-1]) + '/training_images'
    if not os.path.exists(training_image_folder_path):
        os.mkdir(training_image_folder_path)

    # Loop through each file in the Images folder

    print(f"starting to process {image_path} with split size {split_size}")
    image = gdal.Open(image_path)

    # Extract the base name (without extension) for the folder and naming
    basename = os.path.splitext(image_path.split('/')[-1])[0]

    width, height, num_bands = image.RasterXSize, image.RasterYSize, image.RasterCount

    # read labels and save necessary data in labels list

    # Split the image into pieces of size [x, y]
    x = split_size[0]
    y = split_size[1]
    for i in range(0, width, x):
        for j in range(0, height, y):

            # cropped_image = image[j:j + y, i:i + x]
            local_image_width = min(x, width - i)
            local_image_height = min(y, height - j)

            # Save the split image with the specified format
            cropped_filename = f"{basename}_{x}by{y}_{i // x + 1}_{j // y + 1}.jpg"

            # 遍历图片对应的标签。如果标签在图片内即将图片和标签保存在训练
            # Read each band and store in a list
            data = []
            for band in range(num_bands):
                b = image.GetRasterBand(band + 1)
                data.append(b.ReadAsArray(i, j, local_image_width, local_image_height))

            training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)

            # cv2.imwrite(training_image_file_name, cropped_image)
            # Create a new dataset for output
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(training_image_file_name, local_image_width, local_image_height, num_bands,
                                   gdal.GDT_Byte)

            # Write data to the output dataset
            for k, arr in enumerate(data):
                dst_ds.GetRasterBand(k + 1).WriteArray(arr)
            # Set the geotransform and projection of the output dataset
            geotransform = list(image.GetGeoTransform())
            geotransform[0] = geotransform[0] + i * geotransform[1]
            geotransform[3] = geotransform[3] + j * geotransform[5]
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(image.GetProjection())

            # Save and close the output dataset
            dst_ds.FlushCache()
            dst_ds = None


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


# def dataset_preprocessing(image_folder_path, clip_limit, tile_grid_size):
#     # Create training images and labels folder
#     enhanced_image_folder_path = '/'.join(image_folder_path.split('/')[:-1]) + '/enhanced_images'
#     if not os.path.exists(enhanced_image_folder_path):
#         os.mkdir(enhanced_image_folder_path)
#     image_names_suf = os.listdir(image_folder_path)
#     for image_name_suf in image_names_suf:
#         if not image_name_suf.endswith(IMAGE_EXTENSIONS):
#             continue
#         image_name = os.path.splitext(image_name_suf)[0]
#         suf = os.path.splitext(image_name_suf)[1]
#         image_path = os.path.join(image_folder_path, image_name_suf)
#         new_image_path = os.path.join(enhanced_image_folder_path, image_name_suf)
#         bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#         # Apply CLAHE
#         clahe_image = apply_clahe_to_rgb_image(bgr_image, clip_limit, tile_grid_size)
#         cv2.imwrite(new_image_path, clahe_image)
#     print('图片增强完成')
def process_image(args):
    image_name_suf, image_folder_path, enhanced_image_folder_path, clip_limit, tile_grid_size = args
    image_path = os.path.join(image_folder_path, image_name_suf)
    new_image_path = os.path.join(enhanced_image_folder_path, image_name_suf)

    # Check if the image has already been processed
    if os.path.exists(new_image_path):
        return  # Skip processing if the image already exists

    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    clahe_image = apply_clahe_to_rgb_image(bgr_image, clip_limit, tile_grid_size)
    cv2.imwrite(new_image_path, clahe_image)


def dataset_preprocessing(image_folder_path, clip_limit, tile_grid_size, num_processes=4):
    enhanced_image_folder_path = os.path.join(os.path.dirname(image_folder_path), 'enhanced_images')
    os.makedirs(enhanced_image_folder_path, exist_ok=True)

    image_names_suf = [f for f in os.listdir(image_folder_path) if f.endswith(IMAGE_EXTENSIONS)]
    process_args = [(name, image_folder_path, enhanced_image_folder_path, clip_limit, tile_grid_size) for name in
                    image_names_suf]

    with Pool(num_processes) as p:
        p.map(process_image, process_args)

    print('图片增强完成')


def txt_2_geojson(image_folder_path, label_folder_path, class_dict):
    source_path = '/'.join(image_folder_path.split('/')[:-1])
    save_path = source_path + '/' + 'label_geojson'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    image_names_suf = os.listdir(image_folder_path)
    for image_name_suf in image_names_suf:
        image_name, file_extension = os.path.splitext(image_name_suf)
        if file_extension.lower() in IMAGE_EXTENSIONS:
            image_path = image_folder_path + '/' + image_name_suf
            label_path = label_folder_path + '/' + image_name + '.txt'
            gdal.AllRegister()
            dataset = gdal.Open(image_path)
            img_width, img_height = dataset.RasterXSize, dataset.RasterYSize
            adfGeoTransform = dataset.GetGeoTransform()
            out_dir = save_path + '/' + image_name + '.geojson'
            res_dict = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
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
            for index in range(len(res_boxes)):
                polygon = res_boxes[index]
                label = res_labels[index]
                weight = 0
                name = class_dict[label]
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

            with open(out_dir, 'w', encoding='utf-8') as out_file:
                json.dump(res_dict, out_file, ensure_ascii=False, indent=4)


def wind_turbine_data_ge(root_path):
    name_dic = {'进场道路': "AccessRoad", '临时建筑': "TempBuilding", '施工机械': "ConstructionMachinery",
                '小型汽车': "SmallCar", '大型汽车': "LargeCar",
                '风机基础': "WTFoundation", '风机塔筒': "WTTower", '风机叶片': "WTBlade",
                '风机塔架': "WTFrame", '风电机组': "WindTurbine", '升压站': "BoosterStation",
                '施工运维船': 'ConstructionVessel'}
    for name in name_dic.values():
        path = os.path.join(root_path, name)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'images'))
            os.mkdir(os.path.join(path, 'label_geojson'))
    data_path = os.path.join(root_path, '风机数据')
    data_cities = os.listdir(data_path)
    for data_city1 in data_cities:
        from pypinyin import lazy_pinyin
        data_city_path1 = data_path + '/' + data_city1
        data_city_path = data_city_path1 + '/' + os.listdir(data_city_path1)[0]
        data_city_name = ''.join(lazy_pinyin(os.listdir(data_city_path1)[0]))
        for item in os.listdir(data_city_path):
            full_path = os.path.join(data_city_path, item)
            if not os.path.isdir(full_path):
                continue
            image_file_path = full_path + '/影像/' + os.listdir(os.path.join(full_path, '影像'))[0]
            geojson_path = os.path.join(full_path, 'geojson')
            for geojson_name_suf in os.listdir(geojson_path):
                if not geojson_name_suf.endswith('.geojson'):
                    continue
                geojson_file_path = geojson_path + '/' + geojson_name_suf
                geojson_name = os.path.splitext(geojson_name_suf)[0]
                geojson_class = geojson_name.split('_')[-1]
                new_image_file_path = root_path + '/' + name_dic[geojson_class] + '/images/' + name_dic[
                    geojson_class] + '_' + data_city_name + '_' + geojson_name.split('_')[0] + '.tif'
                new_label_path = root_path + '/' + name_dic[geojson_class] + '/label_geojson/' + name_dic[
                    geojson_class] + '_' + data_city_name + '_' + geojson_name.split('_')[0] + '.geojson'
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                if not os.path.exists(new_label_path):
                    print(geojson_file_path)
                    print(new_label_path)
                    shutil.copy2(geojson_file_path, new_label_path)



class PrecisionRecallParameters:
    def __init__(self):
        self.num_of_labels = 0
        self.num_of_predicts = 0
        self.num_of_correct_predicts = 0

    def calculate_pr(self):
        precision = self.num_of_correct_predicts / self.num_of_predicts if self.num_of_predicts else 0
        recall = self.num_of_correct_predicts / self.num_of_labels if self.num_of_labels else 0
        return ("precision: %d/%d=%.2f%%, \nRecall: %d/%d=%.2f%%" % (
            self.num_of_correct_predicts, self.num_of_predicts, precision * 100, self.num_of_correct_predicts,
            self.num_of_labels, recall * 100))

    # Methods to update counts
    def add_labels(self, count):
        self.num_of_labels += count

    def add_predicts(self, count):
        self.num_of_predicts += count

    def add_correct_predicts(self, count):
        self.num_of_correct_predicts += count


def excel_writer(arr, file_dir):
    import pandas as pd
    df = pd.DataFrame(arr)
    df.to_excel(file_dir, index=False, header=False)


# 统计准召
def precision_recall_calculate(label_geojson_folder_path, predict_geojson_folder_path, class_dict):
    label_geojson_file_names = os.listdir(label_geojson_folder_path)
    predict_geojson_file_names = os.listdir(predict_geojson_folder_path)
    source_dir = '/'.join(label_geojson_folder_path.split('/')[:-1])
    out_dir = source_dir + '/PR_Calculate'
    if not os.path.exists(out_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_dir)
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
        # print(file_name)
        # print(suf)
        label_geojson_file_path = label_geojson_folder_path + '/' + label_geojson_file_name
        label_region = None
        for predict_geojson_file_name in predict_geojson_file_names:
            if not predict_geojson_file_name.endswith('.geojson'):
                continue
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
                if label == 'Padding':
                    label = 'Foundation'
                corresponding_key = [key for key, value in class_dict.items() if value == label][0]
                correct_labels.append(label)
                precision_recall_parameter_dict[label].add_labels(1)
                arr_index = 1 + 3 * corresponding_key
                image_arr[arr_index + 1] += 1
                # geometry = feature['geometry']
                # if geometry is None:
                #     continue
                coordinates = geometry['coordinates']
                if len(coordinates) == 0:
                    continue
                label_polygon = []
                for i in range(0, len(coordinates[0]) - 1):
                    # for lat_lon in coordinate[-2:]:
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
                if label_region and not predict_polygon.intersects(label_region):
                    continue

                predict_polygon = []
                for i in range(0, len(coordinates[0]) - 1):
                    # for lat_lon in coordinate[-2:]:
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
            # Convert polygon lists to GeoSeries
            gdf2 = gpd.GeoSeries([Polygon(p).buffer(0) for p in label_polygons])
            gdf1 = gpd.GeoSeries([Polygon(p).buffer(0) for p in predict_polygons])
            # Create spatial index for gdf2
            spatial_index = gdf2.sindex

            predict_flags = [0] * len(gdf1)
            label_flags = [0] * len(gdf2)
            # Loop through polygons in gdf1 and find possible matches in gdf2 using spatial index
            for i, poly1 in enumerate(gdf1):
                predict_label = predict_labels[i]
                possible_matches_index = list(spatial_index.intersection(poly1.bounds))
                possible_matches = gdf2.iloc[possible_matches_index]
                for j, poly2 in enumerate(possible_matches):
                    # print(poly2)
                    correct_label = correct_labels[possible_matches_index[j]]
                    if poly1.intersects(poly2):
                        if label_flags[possible_matches_index[j]]:
                            continue
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
                                predict_flags[i] = 1
                                label_flags[possible_matches_index[j]] = 1
                                # break  # No need to check other polygons if one is found overlapping
            out_file_path = out_dir + '/' + os.path.splitext(predict_geojson_file_name)[0] + '_out' + \
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
                name = '1' if predict_flags[index] else f'{predict_labels[index]} Incorrect'
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
                # res = str(res_dict).replace('/'', '"').replace('None', '"None"')
                with open(out_file_path, 'w', encoding='utf8') as out_file:
                    json.dump(res_dict, out_file)
            images_arr.append(image_arr)
    last_row = []
    for key in class_dict.values():
        print(f'{key}:')
        pr = precision_recall_parameter_dict[key].calculate_pr()
        print(pr)
        last_row.append([key, pr])
    import re

    def natural_sort_key(s):
        """
        Generate a sort key that follows 'natural' sorting, where numbers in strings
        are treated numerically for sorting purposes.
        """
        # Use regular expression to split the string into number and non-number parts
        # Convert number parts into integers
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Sort the 2D array using the windows_sort_key_2d
    images_arr.sort(key=lambda row: natural_sort_key(row[0]))
    images_arr = images_arr + last_row
    excel_out_path = os.path.join(out_dir, "result.xlsx")
    excel_writer(images_arr, excel_out_path)
    # print(precision_recall_parameter_dict['未拆迁房屋'].calculate_pr())


def class_num_calculate(txt_dir, class_dict):
    num_count = [0] * len(class_dict)
    for txt_name_and_suf in os.listdir(txt_dir):
        if not txt_name_and_suf.endswith('.txt'):
            continue
        txt_path = txt_dir + '/' + txt_name_and_suf
        with open(txt_path, 'r') as file:
            for line in file:
                label = int(line.strip().split()[0])
                num_count[label] += 1
    for i in range(len(num_count)):
        print(f'{class_dict[i]}: {num_count[i]}')


def wind_turbine_geojson_classify(root_path):
    name_dic = {'升压站在建': "unfinished_boost_station", '升压站（在建）': "unfinished_boost_station",
                '风机基础': "wt_tube_blade_foundation", '进场道路': "access_road", '施工机械': "machine",
                '风机塔筒': "wt_tube_blade_foundation", '风机叶片': "wt_tube_blade_foundation",
                '风机塔架': "wind_turbine_tower",
                '风电机组': "wt_tube_blade_foundation", '风电施工船': "wind_turbine_vessel",
                '升压站建成': "finished_boost_station",
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
            if not geojson_file.endswith('.geojson') or geojson_file.startswith('linestrip'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_name_pinyin = ''.join(lazy_pinyin(geojson_file_name))
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
                if not os.path.exists(image_file_path):
                    if geojson_file_name[:2] == '风机':
                        image_file_path = data_city_path + '/影像/' + '风电' + geojson_file_name[2:] + '.tif'  # 对应图片地址
                    else:
                        image_file_path = data_city_path + '/影像/' + '风机' + geojson_file_name[2:] + '.tif'  # 对应图片地址
                    if not os.path.exists(image_file_path):
                        print(f'{image_file_path} does not exist')
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
                if not properties["name"] in ['风电机组', '风机塔筒', '风机叶片', '风机基础']:
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
                    # label = 0
                    if properties['name'] == '风电机组':
                        label = 0
                    elif properties['name'] == '风机塔筒':
                        label = 1
                    elif properties['name'] == '风机叶片':
                        label = 2
                    elif properties['name'] == '风机基础':
                        label = 3
                    else:
                        raise Exception('No corresponding label founded!')

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

def cement_plant_geojson_classify(root_path):
    name_dic = {'采石设备': "quarry_equipment"}
    for name in name_dic.values():
        create_folder(root_path, name)
    data_path = os.path.join(root_path, '水泥厂数据')
    data_cities = os.listdir(data_path)
    for data_city in data_cities:
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
            if not geojson_file.endswith('.geojson') or geojson_file.startswith('linestrip'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name == '标注范围':
                continue
            geojson_file_name_pinyin = ''.join(lazy_pinyin(geojson_file_name))
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = data_city_path + '/' + geojson_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                image_file_path = data_city_path + '/影像/' + geojson_file_name + '.tif'  # 对应图片地址
                if not os.path.exists(image_file_path):
                    raise Exception(f'{image_file_path} does not exist')
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

def solar_panel_geojson_classify(root_path):
    name_dic = {'材料堆放': "material_stacking_transformer",
                '施工机械': "machine", '光伏基础': "pv_foundation",
                '光伏支架': "pv_bracket", '光伏面板': "pv_panel", '光伏箱变': "material_stacking_transformer",
                "升压站(建成)": "finished_boost_station", "升压站(在建)": "unfinished_boost_station",
                '升压站建成': "finished_boost_station", "升压站在建": "unfinished_boost_station"}
    for name in name_dic.values():
        create_folder(root_path, name)
    data_path = os.path.join(root_path, '光伏数据')
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
                if not os.path.exists(image_file_path):
                    print(f'{image_file_path} does not exist')
                    return
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
                if not properties["name"] in ['光伏支架']:
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
                    # if properties['name'] == '材料堆放':
                    #     label = 1
                    # else:
                    #     label = 0

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
                    if not geometry:
                        continue
                    coordinates = geometry['coordinates']
                    if not coordinates:
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

def car_geojson_classify(root_path):
    name_dic = {'小轿车': "small_big_vehicle",
                '运输车': "small_big_vehicle", '挖掘机': "excavator"}
    for name in name_dic.values():
        create_folder(root_path, name)
    data_path = os.path.join(root_path, '车辆')
    projects = os.listdir(data_path)
    for project in projects:
        project_pinyin = ''.join(lazy_pinyin(project))
        project_path = os.path.join(data_path, project)  # 郑州房地产4目录
        if os.path.exists(os.path.join(project_path, '标注内容')):
            geojson_root_path = os.path.join(project_path, '标注内容')
        elif os.path.exists(os.path.join(project_path, '矢量')):
            geojson_root_path = os.path.join(project_path, '矢量')  # geojson目录
        elif os.path.exists(os.path.join(project_path, '数据')):
            geojson_root_path = os.path.join(project_path, '数据')
        else:
            raise Exception("找不到标注路径！")

        # batch_read_shapfile(geojson_root_path)
        geojson_files = os.listdir(geojson_root_path)
        for geojson_file in geojson_files:
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            if geojson_file_name.endswith('挖掘机'):
                continue
            geojson_file_name_stripped = geojson_file_name[:8]
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            image_file_path = project_path + '/' + geojson_file_name[:8] + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                raise Exception(f"{image_file_path} not found!")
            f = open(geojson_file_path, encoding='utf8')

            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            features = res_json['features']
            for feature in features:
                properties = feature['properties']
                if properties["name"] is None:
                    continue
                # yolo标注位置
                # print(data_path + '/' + name_dic[properties['name']] + '/label_txt/' + geojson_file_name + '.txt')
                label_path = root_path + '/' + name_dic[properties['name']] + '/label_txt/' + name_dic[
                    properties['name']] + '_' + project_pinyin + '_' + geojson_file_name_stripped + '.txt'
                # 图片存储位置
                new_image_file_path = root_path + '/' + name_dic[properties['name']] + '/images/' + name_dic[
                    properties['name']] + '_' + project_pinyin + '_' + geojson_file_name_stripped + '.tif'
                if not os.path.exists(new_image_file_path):
                    shutil.copy2(image_file_path, new_image_file_path)
                with open(label_path, 'a') as out:
                    # out = open(label_path, 'w', encoding='utf8')
                    # label = 0
                    if properties['name'] == '小轿车':
                        label = 0
                    elif properties['name'] == '运输车':
                        label = 1
                    else:
                        raise Exception(f"{properties['name']} not in dict!")

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

                    geometry = feature['geometry']
                    if not geometry:
                        continue
                    coordinates = geometry['coordinates']
                    if not coordinates:
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
def road_network_data_ge(root_path):
    name_dic = {'进场道路': "access_road"}
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
            if not geojson_file.startswith('linestrip_'):
                continue
            if not geojson_file.endswith('.geojson'):
                continue
            geojson_file_name = os.path.splitext(geojson_file)[0]
            geojson_file_name_pinyin = ''.join(lazy_pinyin(geojson_file_name))
            geojson_file_path = os.path.join(geojson_root_path, geojson_file)  # geojson文件地址
            img_file_name = geojson_file_name_pinyin.split('_')[1]
            image_file_path = data_city_path + '/' + img_file_name + '.tif'  # 对应图片地址
            if not os.path.exists(image_file_path):
                image_file_path = data_city_path + '/影像/' + img_file_name + '.tif'  # 对应图片地址
                if not os.path.exists(image_file_path):
                    image_file_path = data_city_path + '/影像/' + '风电' + data_city[
                                                                           3:] + '-' + img_file_name + '.tif'  # 对应图片地址
                    if not os.path.exists(image_file_path):
                        image_file_path = data_city_path + '/影像/' + '风机' + data_city[
                                                                               3:] + '-' + img_file_name + '.tif'  # 对应图片地址

            # 图片存储位置
            new_image_file_path = (root_path + '/' + name_dic['进场道路'] + '/images/' + name_dic['进场道路'] + '_' +
                                   data_city_pinyin + '_' + img_file_name + '.tif')
            if not os.path.exists(new_image_file_path):
                shutil.copy2(image_file_path, new_image_file_path)

            # geojson存储位置
            new_geojson_file_path = (
                    root_path + '/' + name_dic['进场道路'] + '/label_txt/' + name_dic['进场道路'] + '_' +
                    data_city_pinyin + '_' + img_file_name + '.geojson')
            if not os.path.exists(new_geojson_file_path):
                shutil.copy2(geojson_file_path, new_geojson_file_path)


def geojson_to_shp(shp_dir):
    file_names = os.listdir(shp_dir)

    for file_name in file_names:
        if not file_name.endswith('.geojson'):
            continue
        target_dir = shp_dir
        target_filename = os.path.basename(file_name).split(".")[0]
        # Path to your GeoJSON file
        geojson_path = os.path.join(shp_dir, file_name)

        # Load the GeoJSON file into a GeoDataFrame
        gdf = gpd.read_file(geojson_path)

        # Path where you want to save the Shapefile
        shp_path = target_dir + '/' + target_filename + '.shp'

        # Save the GeoDataFrame as a Shapefile
        gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')


def main():
    # split_arr = [[10000, 10000]]
    # image_path = "data/BuildingSegmentation/Shijiazhuang/20221230.tif"
    # split_image_large(image_path, split_arr[0])
    root_path = "D:/Code/Datasets/cement_plant/data20240425"
    # cement_plant_geojson_classify(root_path)
    # car_geojson_classify(root_path)
    # road_network_data_ge(root_path)
    # wind_turbine_geojson_classify(root_path)
    # solar_panel_geojson_classify(root_path)
    # batch_convert_shapely(root_path)
    # geojson_classify(root_path)
    # foundation_data_ge(root_path)
    # building_data_ge(root_path)
    # realestate_data_ge(root_path)
    # solar_panel_data_ge(root)

    images_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/training_images"
    # dataset_preprocessing(images_folder_path, clip_limit=2, tile_grid_size=[8, 8])
    label_folder_path = "D:/Code/Datasets/cement_plant/data20240425/quarry_equipment/test/label_geojson"
    predict_folder_path = "D:/Code/Datasets/cement_plant/data20240425/quarry_equipment/test/test_img/out"
    # class_dict = {0: 'Digging', 1: 'Foundation', 2: 'Temp Building', 3: 'Finished Building', 4: 'Unfinished Building'}
    class_dict = {0: 'quarry_equipment'}
    # precision_recall_calculate(label_folder_path, predict_folder_path, class_dict)

    # 批量预览图片标注
    image_path = "D:/Code/Datasets/solar_panel/data20240123/pv_panel_and_bracket/test/label_geojson"
    txt_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231211/RealEstate/train/training_labels"
    # show_batch_image_segment(image_path, txt_path)

    # shape文件转geojson
    shp_path = "D:/Code/Datasets/car/长安汽车20210914/20210914"
    # batch_read_shapfile(shp_path)

    # geojson转yolo格式脚本
    image_path = "D:/Code/Datasets/solar_panel/data20240123/images"
    geojson_path = "D:/Code/Datasets/solar_panel/data20240123/labels"
    # geojson2txt_segment(geojson_path, image_path)

    # 大图分割脚本
    split_sizes = [[300, 300], [500, 500], [700, 700], [900, 900], [1100, 1100]]
    images_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231128(Modified)/Temp_and_Large_Buildings/train/training_img_tif"
    labels_folder_path = "D:/Code/gitcode/yolov8/data/Real_Estate/Data_20231128(Modified)/Temp_and_Large_Buildings/train/training_label_tif"

    # 无重叠切割
    # split_images_segment_v1(images_folder_path, split_sizes, labels_folder_path)
    # 有重叠切割
    # split_images_segment_v2(images_folder_path, split_sizes, labels_folder_path)
    # 有重叠切割并保留边缘标注框
    # split_images_segment_v3(images_folder_path, split_sizes, labels_folder_path, threshold=0.65)

    # 划分训练集验证集脚本
    img_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/Combined/images"
    label_path = "D:/Code/gitcode/yolov8/data/Solar_Panel/SolarPanelData_New/Centralized/Combined/label_txt_1"
    # split_train_val(img_path, label_path, percentage=0.1)

    img_path = "D:/Code/Datasets/cement_plant/data20240425/quarry_equipment/test/test_img"
    label_path = "D:/Code/Datasets/cement_plant/data20240425/quarry_equipment/test/test_label"
    # class_dict = {0: 'Digging', 1: 'Foundation', 2: 'Padding', 3: 'Temp Building', 4: 'Finished Building', 5: 'Unfinished Building'}

    class_dict = {0: 'quarry_equipment'}
    # txt_2_geojson(img_path, label_path, class_dict)
    # class_num_calculate(label_path, class_dict)

    label_path = "D:/Code/Datasets/car/small_big_vehicle/train/labels"
    # txt2yolo_bbox(label_path)
    # segment2obb(label_path)

    shp_dir = "D:/Code/Datasets/car/力帆20141006/20141006"
    # geojson_to_shp(shp_dir)


if __name__ == '__main__':
    main()
