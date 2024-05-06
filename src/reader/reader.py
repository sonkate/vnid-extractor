if __name__ == "__main__":
    import sys

    sys.path.append("src")

import cv2
import yaml
from PIL import Image
from vietocr.tool.predictor import Predictor

from configs import configs


class Reader:
    def __init__(self, config: dict) -> None:
        self.config = self.load_config(config)
        self.detector = Predictor(self.config)

    def load_config(self, config):
        # load base config
        ocr_config = self.read_from_config(config["base_config"])
        # load vgg transformer config
        vgg_config = self.read_from_config(config["vgg_config"])
        # update base config
        ocr_config.update(vgg_config)
        # weight
        ocr_config["weights"] = config["model_weight"]
        ocr_config["predictor"]["beamsearch"] = False
        ocr_config["cnn"]["pretrained"] = False
        ocr_config["device"] = config["device"]
        return ocr_config

    def read_from_config(self, file_yml):
        with open(file_yml, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def predict(self, image):
        image = Image.fromarray(image)
        return self.detector.predict(image)


if __name__ == "__main__":
    img = cv2.imread("sample/test_reader.jpg")
    config = configs.reader
    output = Reader(config).predict(img)
    print(output)
