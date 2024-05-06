if __name__ == '__main__':
    import sys
    sys.path.append('src')
import cv2
from cropper.cropper import Cropper
from detector.detector import Detector
from reader.reader import Reader
from configs import configs

class IdExtractor:
    def __init__(self) -> None:
        self.cropper = Cropper(configs.cropper)
        self.detector = Detector(configs.dectector)
        self.reader = Reader(configs.reader)

    def extract(self, image):
        # response = {}
        # cropped_image = self.cropper.crop(image)
        # boxes_dict = self.detector.detect(cropped_image)

        # for k in boxes_dict.keys():
        #     response[k] = ''
        # for k in boxes_dict.keys():
        #     for box in boxes_dict[k]:
        #         ymin, xmin, ymax, xmax = box
        #         temp_image = cropped_image[ymin:ymax, xmin:xmax]
        #         response[k] += self.reader.predict(temp_image) + ' '
        #     # remove leading and trailing whitespace characters
        #     response[k] = response[k].strip()
        try:
            response = {}
            cropped_image = self.cropper.crop(image)
            boxes_dict = self.detector.detect(cropped_image)
            
            for k in boxes_dict.keys():
                response[k] = ''
            for k in boxes_dict.keys():
                for box in boxes_dict[k]:
                    ymin, xmin, ymax, xmax = box
                    temp_image = cropped_image[ymin:ymax, xmin:xmax]
                    response[k] += self.reader.predict(temp_image) + ' '
                # remove leading and trailing whitespace characters
                response[k] = response[k].strip()
        except Exception as e:
            response['exception'] = str(e)
        
        return response

if __name__ == '__main__':
    image = cv2.imread('sample/1.jpg')
    response = IdExtractor().extract(image)
    print(response)