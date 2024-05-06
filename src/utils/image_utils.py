import sys

import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


### Define function
def read_image_file(image):
    return cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)


def resize_image(cvImage, max_width, max_height):
    # Load the image
    image = cvImage

    # Get the original image dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = float(orig_width) / float(orig_height)

    # Calculate the new dimensions based on the maximum width and height
    new_width = min(orig_width, max_width)
    new_height = min(orig_height, max_height)

    # Adjust the dimensions while maintaining the aspect ratio
    if new_width / aspect_ratio > new_height:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def show_image(winname, cvImage):
    height, width, _ = cvImage.shape
    image_show = resize_image(cvImage, 1280, 720)
    cv2.imshow(winname + " (%s, %s)" % (width, height), image_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_legend(cvImage, boxes, texts=None, color=(0, 255, 0)):
    draw_image = cvImage.copy()
    height, width, _ = draw_image.shape

    if texts:
        for box, text in zip((boxes), texts):
            ymin = int(max(1, box[0]))
            xmin = int(max(1, box[1]))
            ymax = int(min(height, box[2]))
            xmax = int(min(width, box[3]))
            cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), color, 2)

            text_size, baseLine = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, text_size[1] + 10)
            cv2.rectangle(
                draw_image,
                (xmin, label_ymin - text_size[1] - 10),
                (xmin + text_size[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                draw_image,
                text,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )  # Draw label text
    else:
        for box in boxes:
            ymin = int(max(1, box[0]))
            xmin = int(max(1, box[1]))
            ymax = int(min(height, box[2]))
            xmax = int(min(width, box[3]))
            cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), color, 2)

    return draw_image
