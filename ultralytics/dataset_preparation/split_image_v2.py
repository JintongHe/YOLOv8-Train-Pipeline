import os
import numpy as np
import gdal
import cv2
from multiprocessing import Pool
from shapely.geometry import box, Polygon, MultiPolygon
import argparse
import ast
import math

def pixel_2_meter(img_path):
    # Open the raster file using GDAL
    ds = gdal.Open(img_path)

    # Get raster size (width and height)
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Get georeferencing information
    geoTransform = ds.GetGeoTransform()
    pixel_size_x = geoTransform[1]  # Pixel width
    pixel_size_y = abs(geoTransform[5])  # Pixel height (absolute value)

    # Get the top latitude from the geotransform and the height
    # geoTransform[3] is the top left y, which gives the latitude
    latitude = geoTransform[3] - pixel_size_y * height
    # Close the dataset
    ds = None

    # Convert road width from meters to pixels
    # road_width_meters = line_width
    meters_per_degree = 111139 * math.cos(math.radians(latitude))
    thickness_pixels_ratio = 1 / (pixel_size_x * meters_per_degree)
    return thickness_pixels_ratio

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


def process_image_segment_noedge(filename, IMAGE_FOLDER, split_size, training_image_folder_path,
                                 training_label_folder_path, LABEL_FOLDER, num_bands):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    ratio = pixel_2_meter(image_path)
    split_size = [int(i * ratio) for i in split_size]  # meters to pixels
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
                labels.append(
                    [label_class, max_values[0], min_values[0], max_values[1], min_values[1], label_box.tolist()])

    # Image segmentation
    cut_width, cut_height = split_size
    for w in range(0, width, cut_width):
        for h in range(0, height, cut_height):
            for dx in [0, int(cut_width / 2)]:
                for dy in [0, int(cut_height / 2)]:
                    w0, h0 = w + dx, h + dy
                    if w0 >= width or h0 >= height:
                        continue

                    w1 = min(w0 + cut_width, width)
                    h1 = min(h0 + cut_height, height)
                    local_image_width, local_image_height = w1 - w0, h1 - h0
                    cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_{dx // int(cut_width / 2)}_{dy // int(cut_height / 2)}.png"
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


def process_image_segment_withedge(filename, IMAGE_FOLDER, split_size, training_image_folder_path,
                                   training_label_folder_path, LABEL_FOLDER, num_bands, threshold):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    ratio = pixel_2_meter(image_path)
    split_size = [int(i * ratio) for i in split_size]  # meters to pixels
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
                    continue
                label_class, label_box = parts[0], np.array(parts[1:]).reshape((-1, 2)).astype(float)
                label_box[:, 0] *= width
                label_box[:, 1] *= height
                labels.append([label_class, label_box])

    # Image segmentation
    cut_width, cut_height = split_size
    for w in range(0, width, cut_width):
        for h in range(0, height, cut_height):
            for dx in [0, int(cut_width / 2)]:
                for dy in [0, int(cut_height / 2)]:
                    w0, h0 = w + dx, h + dy
                    if w0 >= width or h0 >= height:
                        continue
                    w1, h1 = min(w0 + cut_width, width), min(h0 + cut_height, height)
                    local_image_width, local_image_height = w1 - w0, h1 - h0
                    cropped_filename = f"{basename}_{local_image_width}_{local_image_height}_{w // cut_width + 1}_{h // cut_height + 1}_{dx // int(cut_width / 2)}_{dy // int(cut_height / 2)}.png"
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


def split_images_segment_v2(IMAGE_FOLDER, split_sizes, LABEL_FOLDER, with_edge=False):
    training_image_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'images')
    training_label_folder_path = os.path.join(os.path.dirname(IMAGE_FOLDER), 'labels')
    os.makedirs(training_image_folder_path, exist_ok=True)
    os.makedirs(training_label_folder_path, exist_ok=True)

    filenames = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.tif')]
    num_bands = gdal.Open(
        os.path.join(IMAGE_FOLDER, filenames[0])).RasterCount if filenames else 3  # Default to 3 if no files

    # Prepare arguments for parallel processing
    # args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path, LABEL_FOLDER, num_bands)
    #         for filename in filenames for split_size in split_sizes]

    if with_edge:
        threshold = 0.65
        args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path,
                 LABEL_FOLDER, num_bands, threshold)
                for filename in filenames for split_size in split_sizes]
        with Pool() as pool:
            pool.starmap(process_image_segment_withedge, args)
    else:
        args = [(filename, IMAGE_FOLDER, split_size, training_image_folder_path, training_label_folder_path,
                 LABEL_FOLDER, num_bands)
                for filename in filenames for split_size in split_sizes]
        with Pool() as pool:
            pool.starmap(process_image_segment_noedge, args)
    print('Image segmentation completed!')


def main(images_folder_path, labels_folder_path, split_sizes):
    split_images_segment_v2(images_folder_path, split_sizes, labels_folder_path, with_edge=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--split_sizes', type=str, help='List of split sizes')
    parser.add_argument('--images_folder_path', type=str, help='Path to images folder')
    parser.add_argument('--labels_folder_path', type=str, help='Path to labels folder')

    args = parser.parse_args()

    # Assuming split_sizes is a list of lists like [[300, 300], [500, 500]]
    split_sizes = ast.literal_eval(args.split_sizes)

    main(args.images_folder_path, args.labels_folder_path, split_sizes)



