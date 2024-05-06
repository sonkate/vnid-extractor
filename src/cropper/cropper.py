### Import packages
import sys

import cv2

if __name__ == "__main__":
    sys.path.append("src")
    from crop_utils import align_image
else:
    from cropper.crop_utils import align_image

from configs import configs
from utils.image_utils import draw_legend, show_image
from utils.tflite_utils import detect_objects, load_label, load_model


class Cropper:
    def __init__(self, config: dict) -> None:
        self.interpreter = load_model(config["path_to_model"])
        self.labels = load_label(config["path_to_labels"])
        self.min_conf = config["min_conf"]
        self.nms_ths = config["nms_ths"]
        self.debug = config["debug"]

    def crop(self, image):
        # if self.debug:
        #     show_image("Original", image)

        boxes, classes, scores = detect_objects(
            image, self.interpreter, self.nms_ths, self.min_conf
        )

        coordinate_dict = dict()
        height, width, _ = image.shape

        for i in range(len(classes)):
            label = str(self.labels[classes[i]])
            ymin = int(max(1, boxes[i][0]))
            xmin = int(max(1, boxes[i][1]))
            ymax = int(min(height, boxes[i][2]))
            xmax = int(min(width, boxes[i][3]))
            coordinate_dict[label] = (ymin, xmin, ymax, xmax)

        if self.debug:
            # show bounding boxes
            colors = (
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255)
            )
            image_show = image.copy()
            for box, label, color in zip(boxes, coordinate_dict.keys(), colors):
                text = "%s: %d%%" % (label, int(scores[i] * 100))
                image_show = draw_legend(image_show, (box,), (text,), color)
            # show_image("Detect corner", image_show)

        # align image
        cropped_img = align_image(image, coordinate_dict)

        # if self.debug:
            # show_image("Aligned", cropped_img)

        return cropped_img


if __name__ == "__main__":
    img = cv2.imread("1.jpg")
    config = configs.cropper
    Cropper(config).crop(img)
