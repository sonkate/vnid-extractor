import numpy as np

def get_y1(x):
    return x[0]


def get_x1(x):
    return x[1]


def sort_each_category(object_dict: dict):
    output_dict = {}
    for k in object_dict.keys():
        min_y1 = min(object_dict[k], key=get_y1)[0]

        mask = np.where(object_dict[k][:, 0] < min_y1 + 10, True, False)
        line1_text_boxes = object_dict[k][mask]
        line2_text_boxes = object_dict[k][np.invert(mask)]

        line1_text_boxes = sorted(line1_text_boxes, key=get_x1)
        line2_text_boxes = sorted(line2_text_boxes, key=get_x1)

        if len(line2_text_boxes) != 0:
            output_dict[k] = [*line1_text_boxes, *line2_text_boxes]
        else:
            output_dict[k] = line1_text_boxes

    return output_dict


def sort_text(boxes, classes, labels):
    result_dict = {}
    classes = np.array(classes)
    for i, category in enumerate(labels):
        result_dict[category] = boxes[classes == i]

    result_dict = sort_each_category(result_dict)

    return result_dict