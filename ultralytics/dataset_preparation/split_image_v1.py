import os
import numpy as np
from osgeo import gdal
import cv2
from shapely.geometry import box, Polygon, MultiPolygon
from multiprocessing import Pool
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')
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

def process_image_segment_noedge(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path, LABEL_FOLDER, num_bands):

    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = gdal.Open(image_path)
    basename = os.path.splitext(filename)[0]
    label_path = os.path.join(LABEL_FOLDER, basename + '.txt')

    width, height = image.RasterXSize, image.RasterYSize
    labels = []

    # Process labels once for each image
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 9:
                    continue  # Skip invalid lines
                label_class, label_box = parts[0], np.array(parts[1:]).reshape((-1, 2)).astype(float)
                label_box[:, 0] = label_box[:, 0] * width
                label_box[:, 1] = label_box[:, 1] * height
                max_values = np.amax(label_box, axis=0)
                min_values = np.amin(label_box, axis=0)
                labels.append([label_class, max_values[0], min_values[0], max_values[1], min_values[1], label_box.tolist()])

    # Image segmentation
    cut_width, cut_height = split_size
    for w in range(0, width, cut_width):
        for h in range(0, height, cut_height):
            w0, h0 = w, h
            w1 = min(w0 + cut_width, width)
            h1 = min(h0 + cut_height, height)
            local_image_width, local_image_height = w1 - w0, h1 - h0
            cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}.png"
            text_file = os.path.join(training_label_folder_path, os.path.splitext(cropped_filename)[0] + '.txt')

            # Process labels for each segment
            for label in labels:
                label_class, max_x, min_x, max_y, min_y, label_box = label
                if min_x > w0 and max_x < w1 and min_y > h0 and max_y < h1:
                    label_box = np.array(label_box)
                    label_box[:, 0] = (label_box[:, 0] - w0) / local_image_width
                    label_box[:, 1] = (label_box[:, 1] - h0) / local_image_height
                    flattened_label = ' '.join(label_box.reshape(-1).astype(str).tolist())
                    with open(text_file, 'a') as f:
                        f.write(f"{label_class} {flattened_label}\n")
                    training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                    if not os.path.exists(training_image_file_name):
                        data = [image.GetRasterBand(band + 1).ReadAsArray(w0, h0, local_image_width,
                                                                          local_image_height) for band in
                                range(num_bands)]
                        pic = np.dstack(data[::-1])  # Assuming RGB order
                        pic = resize_image_aspect_ratio(pic)
                        cv2.imwrite(training_image_file_name, pic)

def process_image_segment_withedge(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path, LABEL_FOLDER, num_bands, threshold):

    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = gdal.Open(image_path)
    basename = os.path.splitext(filename)[0]
    label_path = os.path.join(LABEL_FOLDER, basename + '.txt')

    width, height = image.RasterXSize, image.RasterYSize
    labels = []

    # Process labels once for each image
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 9:
                    continue  # Skip invalid lines
                label_class, label_box = parts[0], np.array(parts[1:]).reshape((-1, 2)).astype(float)
                label_box[:, 0] = label_box[:, 0] * width
                label_box[:, 1] = label_box[:, 1] * height
                labels.append([label_class, label_box])

    # Image segmentation
    cut_width, cut_height = split_size
    for w in range(0, width, cut_width):
        for h in range(0, height, cut_height):
            w0, h0 = w, h
            w1 = min(w0 + cut_width, width)
            h1 = min(h0 + cut_height, height)
            local_image_width, local_image_height = w1 - w0, h1 - h0
            cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}.png"
            text_file = os.path.join(training_label_folder_path, os.path.splitext(cropped_filename)[0] + '.txt')

            segment_box = box(w0, h0, w1, h1)

            # Process labels for each segment

            for label in labels:
                label_class, label_box = label
                label_polygon = Polygon(label_box).buffer(0)
                intersect_polygon = label_polygon.intersection(segment_box)

                if label_polygon.area == 0 or intersect_polygon.area == 0:
                    continue

                if intersect_polygon.area / label_polygon.area >= threshold:

                    # Handle different types of intersect_polygon geometries
                    if isinstance(intersect_polygon, Polygon):
                        intersect_coords = intersect_polygon.exterior.coords
                    elif isinstance(intersect_polygon, MultiPolygon):
                        # If there are multiple polygons, pick the largest one
                        largest_polygon = max(intersect_polygon.geoms, key=lambda p: p.area)
                        intersect_coords = largest_polygon.exterior.coords
                    else:
                        # Skip non-polygon geometries
                        continue

                    # Normalize the intersecting polygon coordinates
                    normalized_coords = []
                    for coord in intersect_coords:
                        normalized_x = (coord[0] - w0) / local_image_width
                        normalized_y = (coord[1] - h0) / local_image_height
                        normalized_coords.append((normalized_x, normalized_y))

                    # Flatten the list of coordinates for writing to file
                    flattened_label = ' '.join(map(str, np.array(normalized_coords).reshape(-1)))

                    with open(text_file, 'a') as f:
                        f.write(f"{label_class} {flattened_label}\n")
                    training_image_file_name = os.path.join(training_image_folder_path, cropped_filename)
                    if not os.path.exists(training_image_file_name):
                        data = [image.GetRasterBand(band + 1).ReadAsArray(w0, h0, local_image_width,
                                                                          local_image_height) for band in
                                range(num_bands)]
                        pic = np.dstack(data[::-1])  # Assuming RGB order
                        pic = resize_image_aspect_ratio(pic)
                        cv2.imwrite(training_image_file_name, pic)

def split_images_segment_v1(IMAGE_FOLDER, split_sizes, LABEL_FOLDER, with_edge=True):
    training_image_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'images')
    training_label_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'labels')
    os.makedirs(training_image_folder_path, exist_ok=True)
    os.makedirs(training_label_folder_path, exist_ok=True)

    filenames = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(IMAGE_EXTENSIONS)]
    num_bands = gdal.Open(os.path.join(IMAGE_FOLDER, filenames[0])).RasterCount if filenames else 3  # Default to 3 if no files

    # Prepare arguments for parallel processing
    if with_edge:
        threshold = 0.65
        args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path,
                 LABEL_FOLDER, num_bands, threshold)
                for filename in filenames for split_size in split_sizes]
        with Pool() as pool:
            pool.starmap(process_image_segment_withedge, args)
    else:
        args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path, LABEL_FOLDER, num_bands)
            for filename in filenames for split_size in split_sizes]

        with Pool() as pool:
            pool.starmap(process_image_segment_noedge, args)

    print('Image segmentation completed!')

def main():
    # 指定切割大小（像素点）
    split_sizes = [[300, 300], [500, 500], [700, 700], [900, 900]]
    # 图片文件夹路径
    images_folder_path = "D:/Code/Datasets/wind_turbine/dataset20240103/wind_turbine_tube/images"
    # 标注文件夹路径
    labels_folder_path = "D:/Code/Datasets/wind_turbine/dataset20240103/wind_turbine_tube/label_txt"
    # 切割图片（平切）（with_edge: 图片边缘部分标注是否保留）
    split_images_segment_v1(images_folder_path, split_sizes, labels_folder_path, with_edge=False)


if __name__ == '__main__':
    main()
