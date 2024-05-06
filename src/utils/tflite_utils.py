import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


def non_max_suppression_fast(boxes, labels, scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_scores = [scores[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")
    return final_boxes, final_labels, final_scores


def load_model(model_path):
    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_label(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def detect_objects(cvImage, interpreter, nms_threshold=0.2, min_conf=0.5):
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    float_input = input_details[0]["dtype"] == np.float32

    input_mean = 127.5
    input_std = 127.5

    # Load image and resize to expected shape [1xHxWx3]
    image = cvImage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    # Bounding box coordinates of detected objects
    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    # Class index of detected objects
    classes = interpreter.get_tensor(output_details[3]["index"])[0]
    # Confidence of detected objects
    scores = interpreter.get_tensor(output_details[0]["index"])[0]

    mask = np.array(scores > min_conf)
    boxes = np.array(boxes)[mask]
    classes = np.array(classes)[mask]
    scores = np.array(scores)[mask]

    # Convert coordinate to original coordinate
    h, w, _ = cvImage.shape
    boxes[:, 0] *= h
    boxes[:, 1] *= w
    boxes[:, 2] *= h
    boxes[:, 3] *= w

    # Apply non-max suppression
    boxes, classes, scores = non_max_suppression_fast(
        boxes, classes, scores, overlapThresh=nms_threshold
    )

    return boxes, np.array(classes).astype("int"), scores
