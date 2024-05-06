import cv2
import numpy as np


def get_center_point(coordinate_dict):
    di = dict()

    for key in coordinate_dict.keys():
        ymin, xmin, ymax, xmax = coordinate_dict[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        di[key] = (x_center, y_center)

    return di


def find_miss_corner(coordinate_dict):
    position_name = ["TopLeft", "TopRight", "BottomLeft", "BottomRight"]
    position_index = np.array([0, 0, 0, 0])

    for name in coordinate_dict.keys():
        if name in position_name:
            position_index[position_name.index(name)] = 1

    index = np.argmin(position_index)

    return index


def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    index = find_miss_corner(coordinate_dict)

    # calculate missed corner coordinate
    # case 1: missed corner is TopLeft
    if index == 0:
        midpoint = (
            np.add(coordinate_dict["TopRight"], coordinate_dict["BottomLeft"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["BottomRight"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["BottomRight"][0] - thresh
        coordinate_dict["TopLeft"] = (x, y)
    elif index == 1:  # "TopRight"
        midpoint = (
            np.add(coordinate_dict["TopLeft"], coordinate_dict["BottomRight"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["BottomLeft"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["BottomLeft"][0] - thresh
        coordinate_dict["TopRight"] = (x, y)
    elif index == 2:  # "BottomLeft"
        midpoint = (
            np.add(coordinate_dict["TopLeft"], coordinate_dict["BottomRight"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["TopRight"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["TopRight"][0] - thresh
        coordinate_dict["BottomLeft"] = (x, y)
    elif index == 3:  # "BottomRight"
        midpoint = (
            np.add(coordinate_dict["BottomLeft"], coordinate_dict["TopRight"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["TopLeft"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["TopLeft"][0] - thresh
        coordinate_dict["BottomRight"] = (x, y)

    return coordinate_dict


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


def align_image(image, coordinate_dict):
    if len(coordinate_dict) < 3:
        raise ValueError("Please try again")

    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)

    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)

    top_left_point = coordinate_dict["TopLeft"]
    top_right_point = coordinate_dict["TopRight"]
    bottom_right_point = coordinate_dict["BottomRight"]
    bottom_left_point = coordinate_dict["BottomLeft"]

    source_points = np.float32(
        [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
    )

    # transform image and crop
    crop = perspective_transform(image, source_points)

    return crop
