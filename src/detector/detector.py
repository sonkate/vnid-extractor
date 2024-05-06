### Import packages
import sys

import cv2

if __name__ == "__main__":
    sys.path.append("src")
    from detect_utils import sort_text
else:
    from detector.detect_utils import sort_text

from configs import configs
from utils.image_utils import draw_legend, show_image
from utils.tflite_utils import detect_objects, load_label, load_model


class Detector:
    def __init__(self, config: dict) -> None:
        self.interpreter = load_model(config["path_to_model"])
        self.labels = load_label(config["path_to_labels"])
        self.min_conf = config["min_conf"]
        self.nms_ths = config["nms_ths"]
        self.debug = config["debug"]

    def detect(self, image):
        boxes, classes, _ = detect_objects(
            image, self.interpreter, self.nms_ths, self.min_conf
        )

        result_dict = sort_text(boxes, classes, self.labels)

        if self.debug:
            colors = (
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (255, 144, 0),
                (0, 159, 255),
            )
            image_show = image.copy()
            for k, color in zip(result_dict.keys(), colors):
                image_show = draw_legend(image_show, result_dict[k], color=color)

            # show = draw_legend(image, boxes)
            # show_image("Detect info", image_show)

        return result_dict


if __name__ == "__main__":
    img = cv2.imread("sample/cropped.jpg")
    config = configs.dectector
    Detector(config).detect(img)
